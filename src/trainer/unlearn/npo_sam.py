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

        # Save accumulated grads for gradient accumulation
        accum_grads = [p.grad for p in param_list]
        for p in param_list:
            p.grad = None

        # Step 1: Compute forget loss and backward for perturbation direction
        forget_loss_1 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_1)

        # Step 2: Perturb weights (Store eps directly in eps_list reusing p.grad memory)
        eps_list = []
        with torch.no_grad():
            grad_tensors = [p.grad for p in param_list if p.grad is not None]
            if grad_tensors:
                grad_norm = torch.stack([g.norm(2) for g in grad_tensors]).norm(2)
            else:
                grad_norm = torch.tensor(0.0, device=self.args.device)
            
            scale = self.sam_rho / (grad_norm + 1e-12)

            for p in param_list:
                if p.grad is not None:
                    eps = p.grad.detach().mul_(scale)
                    p.data.add_(eps)
                    eps_list.append(eps)
                else:
                    eps_list.append(None)
                p.grad = None

        # Step 3: Forward-backward on forget set at perturbed point
        forget_loss_2 = self._compute_forget_loss(model, inputs)
        self.accelerator.backward(forget_loss_2 * self.gamma)

        # Step 4: Restore weights and free eps memory immediately
        with torch.no_grad():
            for p, eps in zip(param_list, eps_list):
                if eps is not None:
                    p.data.sub_(eps)
        del eps_list

        # Step 5: Forward-backward on retain set at original weights
        retain_loss = self._compute_retain_loss(model, inputs)
        self.accelerator.backward(retain_loss * self.alpha)

        # Step 6: Add back accumulated gradients
        with torch.no_grad():
            for p, acc in zip(param_list, accum_grads):
                if acc is not None:
                    if p.grad is not None:
                        p.grad.add_(acc)
                    else:
                        p.grad = acc

        # Let HF Trainer's optimizer.step() handle the actual update
        total_loss = forget_loss_2 + self.alpha * retain_loss
        return total_loss.detach() / self.args.gradient_accumulation_steps

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
