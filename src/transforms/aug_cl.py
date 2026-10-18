import random

import torch.nn as nn
from torch import Tensor


class InstanceTransform(nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p

    def forward(self, data: Tensor) -> Tensor:
        if random.random() >= self.p:
            return data
        return self.apply_transform(data)

    def apply_transform(self, data: Tensor) -> Tensor:
        raise NotImplementedError
