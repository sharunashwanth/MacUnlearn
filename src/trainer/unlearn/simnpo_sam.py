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

        # 1. Compute perturbation direction via autograd.grad (doesn't touch p.grad)
        forget_loss_1 = self._compute_forget_loss(model, inputs)
        grads = torch.autograd.grad(forget_loss_1, param_list, allow_unused=True)

        # 2. Compute perturbation scale
        grad_norm_sq = sum(
            g.detach().norm(2).pow(2).item() for g in grads if g is not None
        )
        scale = self.sam_rho / (grad_norm_sq ** 0.5 + 1e-12)

        # 3. Perturb weights, store eps for restoration
        eps_list = []
        with torch.no_grad():
            for p, g in zip(param_list, grads):
                if g is not None:
                    eps = g.detach().mul_(scale)
                    p.data.add_(eps)
                    eps_list.append(eps)
                else:
                    eps_list.append(None)
        del grads

        accum_scale = 1.0 / self.args.gradient_accumulation_steps if self.args.gradient_accumulation_steps > 1 else 1.0

        # 4. Backward at perturbed point
        forget_loss_2 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_2 * self.gamma * accum_scale)

        # 5. Restore original weights
        with torch.no_grad():
            for p, eps in zip(param_list, eps_list):
                if eps is not None:
                    p.data.sub_(eps)
        del eps_list

        # 6. Retain backward at original weights
        retain_loss = self._compute_retain_loss(model, inputs)
        self.accelerator.backward(retain_loss * self.alpha * accum_scale)

        return (forget_loss_2 + self.alpha * retain_loss).detach() * accum_scale

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
