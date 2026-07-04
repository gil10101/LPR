"""CRNN recognizer — a CNN + BiLSTM sequence model trained from scratch with CTC.

This is the heart of the project: a neural network trained end-to-end to read the
plate string directly from pixels.

Architecture (Shi et al., 2015, "An End-to-End Trainable Neural Network for
Image-based Sequence Recognition"):

    input crop (C x 32 x 128)
        -> CNN backbone: 7 conv blocks, pooling collapses height 32 -> 1
        -> feature map (512 x 1 x W')  reinterpreted as a length-W' sequence
        -> BiLSTM x2 over the width axis (context in both directions)
        -> linear head -> per-timestep class logits (alphabet + blank)
        -> CTC loss aligns the W'-length prediction to the shorter label

CTC removes the need for per-character bounding boxes or segmentation: the
network learns the alignment itself. Greedy decoding at inference merges repeated
predictions and drops blanks (see ``charset.CharsetCodec.ctc_greedy_decode``).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, the repeated unit of the backbone."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, batch_norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """Convolutional-recurrent network for fixed-height plate crops.

    Parameters
    ----------
    num_classes : alphabet size + 1 (CTC blank). Comes from ``CharsetCodec``.
    in_channels : 1 (grayscale) or 3 (RGB).
    rnn_hidden  : BiLSTM hidden units per direction.
    rnn_layers  : number of stacked BiLSTM layers.
    img_height  : input height; the backbone is designed for 32.
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 rnn_hidden: int = 256, rnn_layers: int = 2,
                 img_height: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.img_height = img_height

        # Backbone. Pooling schedule reduces a 32-px-tall input to height 1 while
        # keeping width resolution (each surviving column = one CTC timestep).
        self.cnn = nn.Sequential(
            _ConvBlock(in_channels, 64),          # 32 x W
            nn.MaxPool2d(2, 2),                    # 16 x W/2
            _ConvBlock(64, 128),                   # 16 x W/2
            nn.MaxPool2d(2, 2),                    # 8  x W/4
            _ConvBlock(128, 256),                  # 8  x W/4
            _ConvBlock(256, 256),                  # 8  x W/4
            nn.MaxPool2d((2, 1), (2, 1)),          # 4  x W/4  (pool height only)
            _ConvBlock(256, 512),                  # 4  x W/4
            _ConvBlock(512, 512),                  # 4  x W/4
            nn.MaxPool2d((2, 1), (2, 1)),          # 2  x W/4
            _ConvBlock(512, 512, kernel=2, stride=1, padding=0),  # 1 x (W/4-1)
        )
        self.dropout = nn.Dropout(dropout)

        # Recurrent head: sees the sequence of column features in both
        # directions, which matters because plate glyphs give left/right context.
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities shaped (T, B, num_classes) for CTC.

        CTCLoss expects time-major log-softmax inputs, so we emit exactly that.
        """
        feats = self.cnn(x)                     # (B, 512, H'=1, W')
        b, c, h, w = feats.size()
        assert h == 1, (
            f"CNN output height must be 1 for the sequence view, got {h}. "
            f"Ensure input height is {self.img_height}."
        )
        feats = feats.squeeze(2)                # (B, 512, W')
        feats = feats.permute(2, 0, 1)          # (W'=T, B, 512)
        feats = self.dropout(feats)

        rnn_out, _ = self.rnn(feats)            # (T, B, 2*hidden)
        logits = self.classifier(rnn_out)       # (T, B, num_classes)
        return logits.log_softmax(dim=2)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_crnn_from_config(cfg, num_classes: int) -> CRNN:
    """Instantiate a CRNN from a :class:`lpr.config.RecognizerConfig`."""
    r = cfg.recognizer
    return CRNN(
        num_classes=num_classes,
        in_channels=r.channels,
        rnn_hidden=r.rnn_hidden,
        rnn_layers=r.rnn_layers,
        img_height=r.img_height,
        dropout=r.cnn_dropout,
    )
