from collections.abc import Sequence
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BNRnn(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.LN = nn.LayerNorm(2 * hidden_size)

    def forward(self, x):
        x = F.gelu(x)
        x, _ = self.gru(x)
        x = self.LN(x)
        return x


class DeepSpeechConv(nn.Module):
    """
    Convolutional layers of Deep Speech model
    """

    def __init__(self, conv_configs, dropout=0.1):
        """
        Args:
            conv_configs (List[Dict]): configs for Convs
        """
        super().__init__()
        layers = []
        in_channels = 1
        self.dropout = dropout
        for cfg in conv_configs:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=cfg["out_channels"],
                    kernel_size=cfg["kernel_size"],
                    stride=cfg["stride"],
                    padding=cfg["padding"],
                )
            )
            # layers.append(nn.LayerNorm(cfg["out_channels"]))
            # layers.append(nn.Dropout(self.dropout))
            layers.append(nn.GELU())
            in_channels = cfg["out_channels"]
        self.conv_layers = nn.Sequential(*layers)
        self.conv_configs = conv_configs

    def transform_input_lengths(self, input_lengths: torch.Tensor, dim=1):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        for cfg in self.conv_configs:
            pad = (
                cfg["padding"][dim]
                if isinstance(cfg["padding"], Sequence)
                else cfg["padding"]
            )
            stride = (
                cfg["stride"][dim]
                if isinstance(cfg["stride"], Sequence)
                else cfg["stride"]
            )
            kernel = (
                cfg["kernel_size"][dim]
                if isinstance(cfg["kernel_size"], Sequence)
                else cfg["kernel_size"]
            )
            input_lengths = (
                torch.floor_divide(input_lengths + 2 * pad - (kernel - 1) - 1, stride)
                + 1
            )
        return input_lengths

    def forward(self, x: torch.Tensor, input_lengths):
        """
        formula from https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
        """

        # X : [batch, n_mels ,T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        b, c, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, time, freq * c)
        return x, self.transform_input_lengths(input_lengths)


class DeepSpeech(nn.Module):
    """

    Deep speech 2 model from
    https://proceedings.mlr.press/v48/amodei16.pdf

    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        hidden_size,
        num_layers,
        dropout,
        conv_layer: nn.Module,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            hidden_size (int): GRU's hidden size.
            num_layers (int): GRU's num layers.
            dropout (int): GRU's dropout
            conv_layer : first conv layers of the model.


        """
        super().__init__()
        # self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)
        # n_feats = n_feats // 2
        self.conv_layer = conv_layer
        n_feats_transformed = self.conv_layer.transform_input_lengths(
            torch.tensor([n_feats], dtype=torch.long), dim=0
        ).item()
        self.fc_1 = nn.Linear(
            conv_layer.conv_configs[-1]["out_channels"] * n_feats_transformed,
            hidden_size,
        )
        self.gru = nn.Sequential(
            *[
                BNRnn(hidden_size if i == 0 else hidden_size * 2, hidden_size, dropout)
                for i in range(num_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x, lengths = self.conv_layer(spectrogram, spectrogram_length)
        x = self.fc_1(x)

        x = self.gru(x)

        output = self.fc(x)

        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": lengths}
