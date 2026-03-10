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

        # 1. Compute perturbation direction using autograd.grad
        forget_loss_1 = self._compute_forget_loss(model, inputs)

        grads = torch.autograd.grad(
            forget_loss_1,
            param_list,
            allow_unused=True
        )

        # 2. Calculate perturbation scale and perturb weights in-place
        grad_norm_sq = torch.tensor(0.0, device=self.args.device)
        for g in grads:
            if g is not None:
                grad_norm_sq += g.detach().norm(2).pow(2)
        grad_norm = grad_norm_sq.sqrt()
        scale = self.sam_rho / (grad_norm + 1e-12)

        # Build eps, perturb weights, then move eps to CPU to free GPU VRAM
        eps_cpu = []
        with torch.no_grad():
            for p, g in zip(param_list, grads):
                if g is not None:
                    eps = g.detach().mul_(scale)
                    p.data.add_(eps)
                    eps_cpu.append(eps.to("cpu"))
                    del eps
                else:
                    eps_cpu.append(None)
        del grads
        torch.cuda.empty_cache()

        accum_scale = 1.0 / self.args.gradient_accumulation_steps if self.args.gradient_accumulation_steps > 1 else 1.0

        # 3. Backward pass on forget set at perturbed point
        forget_loss_2 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_2 * self.gamma * accum_scale)

        # 4. Restore weights using CPU-stored eps (one param at a time)
        with torch.no_grad():
            for p, eps_c in zip(param_list, eps_cpu):
                if eps_c is not None:
                    p.data.sub_(eps_c.to(p.data.device))
        del eps_cpu

        # 5. Backward pass on retain set at original weights
        retain_loss = self._compute_retain_loss(model, inputs)
        self.accelerator.backward(retain_loss * self.alpha * accum_scale)

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
