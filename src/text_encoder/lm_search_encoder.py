import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ctc_text_encoder import CTCTextEncoder
from pyctcdecode import build_ctcdecoder


class LMTextEncoder(CTCTextEncoder):
    def __init__(self, alphabet=None, alpha=0.5, beta=1, beam_size=50, **kwargs):
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
        self.beam_size = beam_size
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path="/home/marat/Desktop/sound/asr_dla/data/4-gram.arpa",
            alpha=alpha,
            beta=beta,
        )
        self.vocab = [self.blank_token] + list(ascii_lowercase.lower() + " ")
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def custom_decode(self, log_probs: torch.Tensor, beam_size=None) -> str:
        if log_probs.ndim == 3:
            log_probs = log_probs.squeeze(0)
        log_probs = log_probs.detach().cpu().numpy()
        return self.decoder.decode(log_probs, beam_width=self.beam_size).lower()
