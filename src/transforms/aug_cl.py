import random

import torch.nn as nn
from torch import Tensor


class InstanceTransform(nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("wrong p value")
        self.p = p

    def forward(self, data: Tensor) -> Tensor:
        if random.random() >= self.p:
            return data
        return self.apply_transform(data)

    def apply_transform(self, data: Tensor) -> Tensor:
        raise NotImplementedError
