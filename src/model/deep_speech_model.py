from typing import List

import torch
import torch.nn as nn


class DeepSpeechConv(nn.Module):
    """
    Convolutional layers of Deep Speech model
    """

    def __init__(self, conv_configs):
        """
        Args:
            conv_configs (List[Dict]): configs for Convs
        """
        super().__init__()
        layers = []
        in_channels = 1
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
            layers.append(nn.BatchNorm2d(num_features=cfg["out_channels"]))
            layers.append(nn.Hardtanh(0, 20, inplace=True))
        self.conv_layers = nn.Sequential(*layers)
        self.conv_configs = conv_configs

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
        for cfg in self.conv_layers:
            pad = (
                cfg["padding"][1]
                if isinstance(cfg["padding"], tuple)
                else cfg["padding"]
            )
            stride = (
                cfg["stride"][1] if isinstance(cfg["stride"], tuple) else cfg["stride"]
            )
            kernel = (
                cfg["kernel"][1] if isinstance(cfg["kernel"], tuple) else cfg["kernel"]
            )
            input_lengths = (
                torch.floor_divide(input_lengths + 2 * pad - (kernel - 1) - 1, stride)
                + 1
            )
        return x, input_lengths


class DeepSpeech(nn.Module):
    """

    Deep speech 2 model from
    https://proceedings.mlr.press/v48/amodei16.pdf

    """

    def __init__(
        self,
        n_feats,
        n_tokens,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            .

        """
        super().__init__()
