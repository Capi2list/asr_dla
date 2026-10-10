from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class SpeechMetric(BaseMetric):
    def __init__(self, text_encoder, decoder="argmax", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.decoder = decoder

    def _calculate_error(self, target_text: str, pred_text: str) -> float:
        raise NotImplementedError

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        errors = []
        lengths = log_probs_length.detach().cpu().numpy()
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            log_prob = log_prob_vec[: int(length)]
            if self.decoder == "argmax":
                pred_text = self.text_encoder.ctc_decode(
                    log_prob.argmax(dim=-1).tolist()
                )
            elif self.decoder == "beam_search":
                pred_text = self.text_encoder.ctc_beam_search(log_prob)
            else:
                raise ValueError(f"Unknown decoder: {self.decoder}")
            errors.append(self._calculate_error(target_text, pred_text))
        return sum(errors) / len(errors)


class WERMetric(BaseMetric):
    def _calculate_error(self, target_text: str, pred_text: str) -> float:
        return calc_wer(target_text, pred_text)


class CERMetric(BaseMetric):
    def _calculate_error(self, target_text: str, pred_text: str) -> float:
        return calc_cer(target_text, pred_text)
