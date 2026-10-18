import torch_audiomentations
from torch import Tensor, nn

from src.transforms.aug_cl import InstanceTransform


class Gain(InstanceTransform):
    def __init__(self, p=1.0, *args, **kwargs):
        super().__init__(p=p)
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
