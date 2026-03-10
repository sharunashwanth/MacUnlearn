"""NPO + SAM trainer.

Implements Sharpness-Aware Minimization (SAM) on top of NPO unlearning.
Based on: "Towards LLM Unlearning Resilient to Relearning Attacks" (Fan et al., ICML 2025)
Reference: https://github.com/OPTML-Group/Unlearn-Smooth

SAM training loop per step:
  1. Forward-backward on forget set -> get gradients for perturbation direction
  2. Perturb weights: w' = w + rho * grad / ||grad||
  3. Forward-backward on forget set at perturbed point -> store forget gradients
  4. Restore weights: w' -> w
  5. Forward-backward on retain set -> store retain gradients
  6. Combined update: final_grad = gamma * forget_grad + alpha * retain_grad
"""

import torch
from torch import nn
from trainer.unlearn.npo import NPO
from trainer.utils import compute_dpo_loss


class NPO_SAM(NPO):
    def __init__(self, sam_rho=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_rho = sam_rho

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        param_list = [p for p in model.parameters() if p.requires_grad]

        # 1. Store accumulated gradients on CPU to extremely save VRAM
        accum_grads_cpu = []
        for p in param_list:
            if p.grad is not None:
                accum_grads_cpu.append(p.grad.cpu())
                p.grad = None
            else:
                accum_grads_cpu.append(None)

        # 2. Compute perturbation direction
        forget_loss_1 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_1)

        # 3. Calculate perturbation scale and apply it
        grad_norm_sq = torch.tensor(0.0, device=self.args.device)
        for p in param_list:
            if p.grad is not None:
                grad_norm_sq += p.grad.detach().norm(2).pow(2)
        grad_norm = grad_norm_sq.sqrt()
        scale = self.sam_rho / (grad_norm + 1e-12)

        eps_cpu = []
        with torch.no_grad():
            for p in param_list:
                if p.grad is not None:
                    p.grad.mul_(scale)
                    p.data.add_(p.grad)
                    eps_cpu.append(p.grad.cpu())
                    p.grad = None
                else:
                    eps_cpu.append(None)

        accum_scale = 1.0 / self.args.gradient_accumulation_steps if self.args.gradient_accumulation_steps > 1 else 1.0

        # 4. Backward pass on forget set at perturbed point
        forget_loss_2 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_2 * self.gamma * accum_scale)

        # 5. Restore weights immediately (loads eps to GPU individually to avoid bulk allocation)
        with torch.no_grad():
            for p, eps in zip(param_list, eps_cpu):
                if eps is not None:
                    p.data.sub_(eps.to(p.device, non_blocking=True))
        del eps_cpu

        # 6. Backward pass on retain set at original weights (accumulates in p.grad)
        retain_loss = self._compute_retain_loss(model, inputs)
        self.accelerator.backward(retain_loss * self.alpha * accum_scale)

        # 7. Merge stored accumulated gradients back into p.grad parameter-by-parameter
        with torch.no_grad():
            for p, acc in zip(param_list, accum_grads_cpu):
                if acc is not None:
                    acc_gpu = acc.to(p.device, non_blocking=True)
                    if p.grad is not None:
                        p.grad.add_(acc_gpu)
                    else:
                        p.grad = acc_gpu

        # Let HF Trainer's optimizer.step() handle the actual update
        total_loss = (forget_loss_2 + self.alpha * retain_loss).detach()
        return total_loss * accum_scale

    def _compute_forget_loss(self, model, inputs):
        forget_inputs = inputs["forget"]
        forget_loss, _ = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        return forget_loss

    def _compute_retain_loss(self, model, inputs):
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
