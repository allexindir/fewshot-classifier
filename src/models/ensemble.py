from typing import Sequence

import torch


class ProbEnsemble:
    def __init__(self, weights: Sequence[float]):
        if not weights:
            raise ValueError("ProbEnsemble needs at least one weight")
        self.weights = [float(w) for w in weights]

    @torch.no_grad()
    def combine(self, member_probs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(member_probs) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} members, got {len(member_probs)}"
            )
        total = sum(self.weights)
        combined = sum(w * p for w, p in zip(self.weights, member_probs))
        return combined / total
