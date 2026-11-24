import torch
from torch import nn

PERSONALIZED_PARAM_PREFIX = "personalized_prompt."


def is_personalized_param_name(name: str) -> bool:
    return name.startswith(PERSONALIZED_PARAM_PREFIX)


class LowRankPersonalizedPrompt(nn.Module):
    """Client-persistent low-rank prompt adapter: Delta_P_i = (alpha / r) * A_i @ B_i."""

    def __init__(self, num_tokens: int, embed_dim: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(num_tokens, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, embed_dim))

    def forward(self) -> torch.Tensor:
        return (self.alpha / self.rank) * (self.A @ self.B)


class FullRankPersonalizedPrompt(nn.Module):
    """Client-persistent full-rank prompt adapter, for ablation against the low-rank version."""

    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(num_tokens, embed_dim))

    def forward(self) -> torch.Tensor:
        return self.delta


def build_personalized_prompt(personalization_type: str, num_tokens: int, embed_dim: int,
                               rank: int, alpha: float):
    if personalization_type == "none":
        return None
    if personalization_type == "low_rank":
        return LowRankPersonalizedPrompt(num_tokens, embed_dim, rank, alpha)
    if personalization_type == "full_rank":
        return FullRankPersonalizedPrompt(num_tokens, embed_dim)
    raise ValueError(f"unknown personalization_type: {personalization_type}")
