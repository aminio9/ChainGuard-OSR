"""Dynamic Global Anchor Regularization with Shannon Entropy Gating (Equations 5-8).

Prevents client drift in severe Non-IID settings by distilling soft label consensus
from the frozen global anchor model, gated by anchor certainty and a sigmoid time ramp.
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_shannon_certainty(
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Shannon entropy-based certainty gate U_anc(x) (Equation 5).

    Mathematical Formulation (Equation 5):
        p_anc^{(tau_g)}(x) = softmax(teacher_logits / tau_g)
        H(p_anc) = - sum_{c in C_K} p_anc,c * log(p_anc,c)
        U_anc(x) = 1.0 - ( H(p_anc) / log(|C_K|) )

    Args:
        teacher_logits: Tensor of shape (batch_size, num_classes) from frozen global anchor.
        temperature: Distillation temperature tau_g (default: 2.0).
        eps: Numerical stability constant.

    Returns:
        Tensor of shape (batch_size,) with certainty values in [0, 1].
    """
    num_classes = teacher_logits.size(-1)
    if num_classes <= 1:
        return torch.ones(teacher_logits.size(0), device=teacher_logits.device)

    # Softmax at temperature tau_g
    scaled_logits = teacher_logits / max(temperature, 1e-4)
    probs = F.softmax(scaled_logits, dim=-1)

    # Shannon Entropy in nats: H(p) = - sum(p * log(p))
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Normalized entropy in [0, 1] using log(|C_K|)
    max_entropy = math.log(float(num_classes))
    normalized_entropy = torch.clamp(entropy / max_entropy, min=0.0, max=1.0)

    # Certainty gate: U_anc(x) = 1 - (H / log|C_K|)
    u_anc = 1.0 - normalized_entropy
    return torch.clamp(u_anc, min=0.0, max=1.0)


def compute_anchor_weight(
    u_anc: torch.Tensor,
    t_ratio: float,
    max_lambda: float = 1.0,
    kappa: float = 10.0,
    rho_warm: float = 0.15,
    t0: Optional[float] = None,
) -> torch.Tensor:
    """Compute dynamic sample-wise anchor weight lambda_anc(x, t) (LaTeX Equation 73).

    Mathematical Formulation (LaTeX Equation 73):
        lambda_anc(x, t) = lambda_anc^{max} * ( 1 / (1 + exp(-kappa * (t/T - rho_warm))) ) * U_anc(x)

    Args:
        u_anc: Tensor of shape (batch_size,) representing certainty values.
        t_ratio: Progression ratio t / T in [0, 1].
        max_lambda: Peak penalty ceiling lambda_anc^{max} (default: 1.0).
        kappa: Sigmoid steepness parameter kappa (default: 10.0).
        rho_warm: Warm-up fraction parameter rho_warm (default: 0.15).
        t0: Optional legacy alias for rho_warm.

    Returns:
        Tensor of shape (batch_size,) with adaptive anchor weights.
    """
    warmup = t0 if t0 is not None else rho_warm
    if float(t_ratio) < warmup:
        # Strictly reserve first rho_warm of rounds for unconstrained local empirical exploration (LaTeX Eq. 66)
        return torch.zeros_like(u_anc)

    # Sigmoid progression factor following warm-up
    sigmoid_arg = -kappa * (float(t_ratio) - warmup)
    sigmoid_arg = max(min(sigmoid_arg, 50.0), -50.0)
    time_ramp = 1.0 / (1.0 + math.exp(sigmoid_arg))

    return max_lambda * time_ramp * u_anc


class DynamicGlobalAnchorLoss(nn.Module):
    """Dynamic Global Anchor Regularization with Shannon Entropy Gating (Equations 5-8 & LaTeX Eq. 66-78).

    Enforces non-zero gradients for missing local classes:
        d [lambda_anc * L_anchor] / d ell_{k, c}(x) = lambda_anc(x, t) * tau_g * (p_{k,c}^{(tau_g)} - p_{anc,c}^{(tau_g)}) != 0
    """

    def __init__(
        self,
        tau_g: float = 2.0,
        max_lambda: float = 1.0,
        kappa: float = 10.0,
        rho_warm: float = 0.15,
        t0: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.tau_g = tau_g
        self.max_lambda = max_lambda
        self.kappa = kappa
        self.rho_warm = t0 if t0 is not None else rho_warm
        self.t0 = self.rho_warm
        self.eps = eps

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        t_ratio: float,
    ) -> torch.Tensor:
        """Compute the entropy-gated anchor distillation loss (Equation 7).

        Args:
            student_logits: Active local client logits of shape (batch_size, num_classes).
            teacher_logits: Frozen global anchor logits of shape (batch_size, num_classes).
            t_ratio: Current round ratio t / T in [0.0, 1.0].

        Returns:
            Scalar tensor representing the regularized anchor distillation loss.
        """
        if student_logits.size(-1) <= 1:
            return torch.tensor(0.0, device=student_logits.device)

        with torch.no_grad():
            u_anc = compute_shannon_certainty(teacher_logits, temperature=self.tau_g, eps=self.eps)
            lambda_anc = compute_anchor_weight(
                u_anc=u_anc,
                t_ratio=t_ratio,
                max_lambda=self.max_lambda,
                kappa=self.kappa,
                t0=self.t0,
            )  # Shape: (batch_size,)

            p_teacher = F.softmax(teacher_logits / self.tau_g, dim=-1)

        # Student log-probabilities at temperature tau_g
        log_p_student = F.log_softmax(student_logits / self.tau_g, dim=-1)

        # Pointwise KL divergence: D_KL(p_teacher || p_student) = sum(p_teacher * (log(p_teacher) - log(p_student)))
        log_p_teacher = torch.log(p_teacher + self.eps)
        kl_pointwise = torch.sum(p_teacher * (log_p_teacher - log_p_student), dim=-1)  # (batch_size,)

        # Scaled by tau_g^2 as per Hinton KD and Equation 7
        scaled_anchor_loss = (self.tau_g ** 2) * kl_pointwise

        # Gated by dynamic weight lambda_anc(x, t)
        loss = torch.mean(lambda_anc * scaled_anchor_loss)
        return loss
