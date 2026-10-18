from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class SpeechMetric(BaseMetric):
    def __init__(self, text_encoder, argmax=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.is_argmax = argmax

    def _calculate_error(self, target_text: str, pred_text: str) -> float:
        raise NotImplementedError

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        errors = []
        lengths = log_probs_length.detach().cpu().numpy().flatten()
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            log_prob = log_prob_vec[: int(length)]
            if self.is_argmax:
                pred_text = self.text_encoder.argmax_decode(log_prob)
            else:
                pred_text = self.text_encoder.custom_decode(log_prob)
            errors.append(self._calculate_error(target_text, pred_text))
        return sum(errors) / len(errors)


class WERMetric(SpeechMetric):
    def _calculate_error(self, target_text: str, pred_text: str) -> float:
        return calc_wer(target_text, pred_text)
