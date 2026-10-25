import torch
import torch.nn as nn

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    # x.shape[-1] gives the feature dimension (d)
    norm_layer = nn.RMSNorm(x.shape[-1], eps=epsilon).to(x.device, x.dtype)

    norm_layer.weight.data = g.to(x.dtype)

    with torch.no_grad():
        return norm_layer(x)