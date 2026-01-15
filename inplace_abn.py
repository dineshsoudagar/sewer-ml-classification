# inplace_abn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class InPlaceABN(nn.BatchNorm2d):
    """
    Minimal inference-friendly substitute for the inplace_abn.InPlaceABN layer.

    This keeps state_dict compatibility by inheriting BatchNorm2d directly,
    so parameter names match: weight, bias, running_mean, running_var, etc.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.activation = activation
        self.activation_param = activation_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)

        act = (self.activation or "identity").lower()
        if act in ("identity", "none", ""):
            return x
        if act in ("relu",):
            return F.relu(x, inplace=False)
        if act in ("leaky_relu", "leakyrelu"):
            return F.leaky_relu(x, negative_slope=float(self.activation_param), inplace=False)
        if act in ("elu",):
            return F.elu(x, inplace=False)

        raise ValueError(f"Unsupported activation in InPlaceABN stub: {self.activation}")
