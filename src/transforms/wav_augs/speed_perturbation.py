import random

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from src.transforms.aug_cl import InstanceTransform


class SpeedPerturbation(InstanceTransform):
    def __init__(self, sr: int, min_rate: float, max_rate: float, p=1.0):
        super().__init__(p=p)
        self.sr = sr
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply_transform(self, data: Tensor):
        rate = random.uniform(self.min_rate, self.max_rate)
        wav, _ = torchaudio.functional.speed(
            data.unsqueeze(0), orig_freq=self.sr, factor=rate
        )
        return wav.squeeze(0)
