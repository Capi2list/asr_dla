import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.transforms.aug_cl import InstanceTransform


class TimeMasking(InstanceTransform):
    def __init__(self, p=1.0, *args, **kwargs):
        super().__init__(p=p)
        self.aug = T.TimeMasking(*args, **kwargs)

    def apply_transform(self, data: torch.Tensor):
        return self.aug(data.transpose(-2, -1)).transpose(-2, -1)


class FrequencyMasking(InstanceTransform):
    def __init__(self, p=1.0, *args, **kwargs):
        super().__init__(p=p)
        self.aug = T.FrequencyMasking(*args, **kwargs)

    def apply_transform(self, data: torch.Tensor):
        return self.aug(data.transpose(-2, -1)).transpose(-2, -1)
