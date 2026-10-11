from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.metrics.wer import SpeechMetric

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class CERMetric(SpeechMetric):
    def _calculate_error(self, target_text: str, pred_text: str) -> float:
        return calc_cer(target_text, pred_text)
