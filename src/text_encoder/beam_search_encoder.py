import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class BeamSearchEncoder(CTCTextEncoder):

    """
    Custom hand-made beam search class.
    """

    def __init__(self, alphabet=None, beam_size=5, **kwargs):
        super().__init__(alphabet, **kwargs)
        self.beam_size = beam_size

    def expand_beams(self, current_log_probs, current_beams) -> Dict[str, List[float]]:
        new_beams = defaultdict(lambda: [-np.inf, -np.inf])
        for prefix, (p_empty, p_non_empty) in current_beams.items():
            for next_char in self.vocab:
                if next_char == self.EMPTY_TOK:
                    new_beams[prefix][0] = np.logaddexp(
                        new_beams[prefix][0],
                        current_log_probs[self.char2ind[next_char]]
                        + np.logaddexp(p_empty, p_non_empty),
                    )
                else:
                    if prefix and prefix[-1] == next_char:
                        new_beams[prefix][1] = np.logaddexp(
                            new_beams[prefix][1],
                            current_log_probs[self.char2ind[next_char]] + p_non_empty,
                        )
                        new_beams[prefix + next_char][1] = np.logaddexp(
                            new_beams[prefix + next_char][1],
                            current_log_probs[self.char2ind[next_char]] + p_empty,
                        )
                    else:
                        new_beams[prefix + next_char][1] = np.logaddexp(
                            new_beams[prefix + next_char][1],
                            current_log_probs[self.char2ind[next_char]]
                            + np.logaddexp(p_empty, p_non_empty),
                        )
        return new_beams

    def custom_decode(self, log_probs: torch.Tensor) -> str:
        beams = {"": [0.0, -np.inf]}  # (p_empty, p_non_empty)
        log_probs = log_probs.detach().cpu().numpy()
        for cur_log_probs in log_probs:
            beams = self.expand_beams(cur_log_probs, beams)
            beams = dict(
                sorted(
                    beams.items(),
                    key=lambda x: np.logaddexp(x[1][1], x[1][0]),
                    reverse=True,
                )[: self.beam_size]
            )
        result = [
            (prefix, np.logaddexp(scores[0], scores[1]))
            for prefix, scores in beams.items()
        ]
        return sorted(result, key=lambda x: x[1])[-1][0]
