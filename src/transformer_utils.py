import torch
import torch.nn as nn

def to_2tuple(x):
    return tuple([x] * 2)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output