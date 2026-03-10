"""SimNPO + SAM trainer (Proposed Method).

Implements Sharpness-Aware Minimization (SAM) on top of SimNPO unlearning.
Based on: "Towards LLM Unlearning Resilient to Relearning Attacks" (Fan et al., ICML 2025)

Same SAM training loop as NPO_SAM but uses SimNPO's reference-free forget loss.
"""

import torch
import torch.nn.functional as F
from torch import nn
from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll


class SimNPO_SAM(SimNPO):
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

        total_loss = (forget_loss_2 + self.alpha * retain_loss).detach()
        return total_loss * accum_scale

    def _compute_forget_loss(self, model, inputs):
        forget_inputs = inputs["forget"]
        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        forget_loss, _ = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
        return forget_loss

    def _compute_retain_loss(self, model, inputs):
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
