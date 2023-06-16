import math
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from . import commons, modules
from .commons import get_padding
from .modules import ConvNext2d, ISTFTHead, LoRALinear1d, WaveBlock

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

sr2sr = {
    "24k": 24000,
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

class GeneratorVocos(torch.nn.Module):
    def __init__(
        self,
        emb_channels,
        inter_channels,
        gin_channels,
        n_layers,
        n_fft,
        hop_length,
    ):
        super(GeneratorVocos, self).__init__()
        self.n_layers = n_layers

        self.resblocks = nn.ModuleList()
        self.init_conv = LoRALinear1d(emb_channels, inter_channels, gin_channels, r=4)
        for _ in range(self.n_layers):
            self.resblocks.append(WaveBlock(inter_channels, gin_channels, [11] * 3, [1] * 3, [1, 2, 4], 2, r=4))
        self.head = ISTFTHead(inter_channels, n_fft=n_fft, hop_length=hop_length)

    def forward(self, x, x_mask, g):
        x = self.init_conv(x, g)
        for i in range(self.n_layers):
            x = self.resblocks[i](x, x_mask, g)
        x = self.head(x)
        return x

    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()
        self.init_conv.remove_weight_norm()


class SynthesizerTrnMs256NSFSid(nn.Module):
    def __init__(
        self,
        segment_size,
        n_fft,
        hop_length,
        inter_channels,
        n_layers,
        spk_embed_dim,
        gin_channels,
        emb_channels,
        sr,
        **kwargs
    ):
        super().__init__()
        if type(sr) == type("strr"):
            sr = sr2sr[sr]
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.inter_channels = inter_channels
        self.n_layers = n_layers
        self.spk_embed_dim = spk_embed_dim
        self.gin_channels = gin_channels
        self.emb_channels = emb_channels
        self.sr = sr

        self.emb_pitch = nn.Embedding(256, emb_channels)  # pitch 256
        self.dec = GeneratorVocos(
            emb_channels,
            inter_channels,
            gin_channels,
            n_layers,
            n_fft,
            hop_length
        )

        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        print(
            "gin_channels:",
            gin_channels,
            "self.spk_embed_dim:",
            self.spk_embed_dim,
            "emb_channels:",
            emb_channels,
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()

    def forward(
        self, phone, phone_lengths, pitch, pitchf, ds
        ):
        g = self.emb_g(ds).unsqueeze(-1)
        x = phone + self.emb_pitch(pitch)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        m_p_slice, ids_slice = commons.rand_slice_segments(
            x, phone_lengths, self.segment_size
        )
        mask_slice = commons.slice_segments(x_mask, ids_slice, self.segment_size)
        o = self.dec(m_p_slice, mask_slice, g)
        return o, ids_slice, x_mask, g

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        x = phone + self.emb_pitch(pitch)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        o = self.dec((x * x_mask)[:, :, :max_len], x_mask, g)
        return o, x_mask, (None, None, None, None)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, gin_channels, upsample_rates, final_dim=256, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        self.init_kernel_size = upsample_rates[-1] * 3
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        N = len(upsample_rates)
        self.init_conv = norm_f(Conv2d(1, final_dim // (2 ** (N - 1)), (upsample_rates[-1] * 3, 1), (upsample_rates[-1], 1)))
        self.convs = nn.ModuleList()
        for i, u in enumerate(upsample_rates[::-1][1:], start=1):
            self.convs.append(
                ConvNext2d(
                    final_dim // (2 ** (N - i)),
                    final_dim // (2 ** (N - i - 1)),
                    gin_channels,
                    (u*3, 1),
                    (u, 1),
                    4,
                    r=2 + i//2
                )
            )
        self.conv_post = weight_norm(Conv2d(final_dim, 1, (3, 1), (1, 1)))

    def forward(self, x, g):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (n_pad, 0), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        x = torch.flip(x, dims=[2])
        x = F.pad(x, [0, 0, 0, self.init_kernel_size - 1], mode="constant")
        x = self.init_conv(x)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = torch.flip(x, dims=[2])
        fmap.append(x)

        for i, l in enumerate(self.convs):
            x = l(x, g)
            fmap.append(x)

        x = F.pad(x, [0, 0, 2, 0], mode="constant")
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, upsample_rates, gin_channels, periods=[2, 3, 5, 7, 11, 17], **kwargs):
        super(MultiPeriodDiscriminator, self).__init__()

        discs = [
            DiscriminatorP(i, gin_channels, upsample_rates, use_spectral_norm=False) for i in periods
        ]
        self.ups = np.prod(upsample_rates)
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, g):
        fmap_rs = []
        fmap_gs = []
        y_d_rs = []
        y_d_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y, g)
            y_d_g, fmap_g = d(y_hat, g)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
