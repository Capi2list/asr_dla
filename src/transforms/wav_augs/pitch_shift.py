import random

import torch
import torch.nn as nn
import torch_audiomentations
import torchaudio.transforms as T

from src.transforms.aug_cl import InstanceTransform


class PitchShift(InstanceTransform):
    def __init__(self, sr: int, min_steps=-2.0, max_steps=2.0, p=1.0, *args, **kwargs):
        super().__init__(p=p)
        self.aug = torch_audiomentations.PitchShift(
            sample_rate=sr,
            min_transpose_semitones=min_steps,
            max_transpose_semitones=max_steps,
            p=1.0,
            output_type="tensor",
        )

    def apply_transform(self, data: torch.Tensor):
        x = data.unsqueeze(0)
        return self.aug(x).squeeze(0)
