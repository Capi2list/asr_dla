import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pyctcdecode import build_ctcdecoder

# TODO add BPE, LM
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result_inds = []
        prev_id = None
        for curr_id in inds:
            if prev_id == curr_id:
                continue
            if curr_id != self.char2ind[self.EMPTY_TOK]:
                result_inds.append(curr_id)
            prev_id = curr_id
        return self.decode(result_inds)

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

    def ctc_beam_search(self, log_probs: torch.Tensor, beam_size=5) -> str:
        beams = {"": [0.0, -np.inf]}  # (p_empty, p_non_empty)
        log_probs = log_probs.detach().cpu().numpy()
        for cur_log_probs in log_probs:
            beams = self.expand_beams(cur_log_probs, beams)
            beams = dict(
                sorted(
                    beams.items(),
                    key=lambda x: np.logaddexp(x[1][1], x[1][0]),
                    reverse=True,
                )[:beam_size]
            )
        result = [
            (prefix, np.logaddexp(scores[0], scores[1]))
            for prefix, scores in beams.items()
        ]
        return sorted(result, key=lambda x: x[1])[-1][0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class LMTextEncoder(CTCTextEncoder):
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, alpha=0.5, beta=1.0, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        self.alphabet = alphabet
        self.blank_token = self.EMPTY_TOK
        vocab = [self.blank_token] + list(ascii_lowercase.upper() + " ")
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path="/home/marat/Desktop/sound/asr_dla/data/4-gram.arpa",
            alpha=alpha,
            beta=beta,
        )
        self.vocab = [self.blank_token] + list(ascii_lowercase.lower() + " ")
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_beam_search(self, log_probs: torch.Tensor, beam_size=None) -> str:
        if log_probs.ndim == 3:
            log_probs = log_probs.squeeze(0)
        log_probs = log_probs.detach().cpu().numpy()
        return self.decoder.decode(log_probs, beam_width=50).lower()
