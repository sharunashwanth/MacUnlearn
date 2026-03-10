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

        # Step 1: Compute forget loss and backward for perturbation direction
        forget_loss_1 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_1)

        perturb_grads = []
        for p in param_list:
            if p.grad is not None:
                perturb_grads.append(p.grad.detach().clone())
            else:
                perturb_grads.append(None)

        # Compute perturbation: eps = rho * grad / ||grad||
        norm_list = [g.norm(2) for g in perturb_grads if g is not None]
        grad_norm = torch.stack(norm_list).norm(2) if norm_list else torch.tensor(0.0, device=self.args.device)

        eps_list = []
        for g in perturb_grads:
            if g is not None:
                eps_list.append(g * (self.sam_rho / (grad_norm.to(g.device) + 1e-12)))
            else:
                eps_list.append(None)

        # Step 2: Perturb weights
        model.zero_grad()
        with torch.no_grad():
            for p, eps in zip(param_list, eps_list):
                if eps is not None:
                    p.data.add_(eps)

        # Step 3: Forward-backward on forget set at perturbed point
        forget_loss_2 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_2)

        forget_grads = []
        for p in param_list:
            if p.grad is not None:
                forget_grads.append(p.grad.detach().clone())
            else:
                forget_grads.append(None)

        # Step 4: Restore weights
        with torch.no_grad():
            for p, eps in zip(param_list, eps_list):
                if eps is not None:
                    p.data.sub_(eps)

        # Step 5: Forward-backward on retain set
        model.zero_grad()
        retain_loss = self._compute_retain_loss(model, inputs)
        self.accelerator.backward(retain_loss)

        retain_grads = []
        for p in param_list:
            if p.grad is not None:
                retain_grads.append(p.grad.detach().clone())
            else:
                retain_grads.append(None)

        # Step 6: Combine gradients and set
        model.zero_grad()
        with torch.no_grad():
            for p, fg, rg in zip(param_list, forget_grads, retain_grads):
                f_grad = fg if fg is not None else torch.zeros_like(p.data)
                r_grad = rg if rg is not None else torch.zeros_like(p.data)
                p.grad = self.gamma * f_grad + self.alpha * r_grad

        total_loss = forget_loss_2 + self.alpha * retain_loss
        return total_loss.detach() / self.args.gradient_accumulation_steps

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
