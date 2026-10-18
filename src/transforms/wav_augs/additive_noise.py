import random

import torch
import torch.nn as nn
import torchaudio

from src.transforms.aug_cl import InstanceTransform


class AdditiveNoise(InstanceTransform):
    def __init__(self, min_snr_db=5.0, max_snr_db=30.0, p=1.0):
        super().__init__(p=p)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def apply_transform(self, data):
        noise = torch.randn_like(data)
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)

        snr = torch.tensor(
            [snr_db],
            dtype=data.dtype,
            device=data.device,
        )

        return torchaudio.functional.add_noise(data, noise, snr)
