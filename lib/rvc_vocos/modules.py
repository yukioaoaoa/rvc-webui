import math

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import commons, modules
from .commons import get_padding, init_weights
from .transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1, bias=True):
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, dilation=dilation, bias=bias))

    def forward(self, x):
        x = torch.flip(x, [2])
        x = F.pad(x, [0, (self.kernel_size - 1) * self.dilation], mode="constant", value=0.)
        size = x.shape[2] // self.stride
        x = self.conv(x)[:, :, :size]
        x = torch.flip(x, [2])
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)


class CausalConvTranspose1d(nn.Module):
    """
    padding = 0, dilation = 1のとき

    Lout = (Lin - 1) * stride + kernel_rate * stride + output_padding
    Lout = Lin * stride + (kernel_rate - 1) * stride + output_padding
    output_paddingいらないね
    """
    def __init__(self, in_channels, out_channels, kernel_rate=3, stride=1, groups=1):
        super(CausalConvTranspose1d, self).__init__()
        kernel_size = kernel_rate * stride
        self.trim_size = (kernel_rate - 1) * stride
        self.conv = weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups))

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.trim_size]

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)


class LoRALinear1d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = weight_norm(nn.Conv1d(in_channels, out_channels, 1))
        self.adapter_in = nn.Conv1d(info_channels, in_channels * r, 1)
        self.adapter_out = nn.Conv1d(info_channels, out_channels * r, 1)
        nn.init.normal_(self.adapter_in.weight.data, 0, 0.01)
        nn.init.constant_(self.adapter_out.weight.data, 1e-6)
        self.adapter_in = weight_norm(self.adapter_in)
        self.adapter_out = weight_norm(self.adapter_out)

    def forward(self, x, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        x = self.main_fc(x) + torch.einsum("brl,brc->bcl", torch.einsum("bcl,bcr->brl", x, a_in), a_out)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.main_fc)
        remove_weight_norm(self.adapter_in)
        remove_weight_norm(self.adapter_out)


class LoRALinear2d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = weight_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1)))
        self.adapter_in = nn.Conv1d(info_channels, in_channels * r, 1)
        self.adapter_out = nn.Conv1d(info_channels, out_channels * r, 1)
        nn.init.normal_(self.adapter_in.weight.data, 0, 0.01)
        nn.init.constant_(self.adapter_out.weight.data, 1e-6)
        self.adapter_in = weight_norm(self.adapter_in)
        self.adapter_out = weight_norm(self.adapter_out)

    def forward(self, x, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        x = self.main_fc(x) + torch.einsum("brhw,brc->bchw", torch.einsum("bchw,bcr->brhw", x, a_in), a_out)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.main_fc)
        remove_weight_norm(self.adapter_in)
        remove_weight_norm(self.adapter_out)


class ConvNext2d(torch.nn.Module):
    """
    Causal ConvNext Block
    stride = 1 only
    """
    def __init__(self, in_channels, out_channels, gin_channels, kernel_size, stride, extend_ratio, r, use_spectral_norm=False):
        super(ConvNext2d, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.kernel_size = kernel_size
        self.dwconv = norm_f(Conv2d(in_channels, in_channels, kernel_size, stride, groups=in_channels))
        self.pwconv1 = LoRALinear2d(in_channels, inner_channels, gin_channels, r=r)
        self.pwconv2 = LoRALinear2d(inner_channels, out_channels, gin_channels, r=r)
        self.act = nn.GELU()
        self.norm = LayerNorm(in_channels)

    def forward(self, x, g):
        x = F.pad(x, [0, 0, self.kernel_size[0] - 1, 0], mode="constant")
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x, g)
        x = self.act(x)
        x = self.pwconv2(x, g)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.dwconv)


class WaveBlock(torch.nn.Module):
    def __init__(self, inner_channels, gin_channels, kernel_sizes, strides, dilations, extend_ratio, r):
        super(WaveBlock, self).__init__()
        norm_f = weight_norm 
        extend_channels = int(inner_channels * extend_ratio)
        self.dconvs = nn.ModuleList()
        self.p1convs = nn.ModuleList()
        self.p2convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # self.ses = nn.ModuleList()
        # self.norms = []
        for i, (k, s, d) in enumerate(zip(kernel_sizes, strides, dilations)):
            self.dconvs.append(DilatedCausalConv1d(inner_channels, inner_channels, k, stride=s, dilation=d, groups=inner_channels))
            self.p1convs.append(LoRALinear1d(inner_channels, extend_channels, gin_channels, r))
            self.p2convs.append(LoRALinear1d(extend_channels, inner_channels, gin_channels, r))
            self.norms.append(LayerNorm(inner_channels))
        self.act = nn.GELU()

    def forward(self, x, x_mask, g):
        x *= x_mask
        for i in range(len(self.dconvs)):
            residual = x
            x = self.dconvs[i](x)
            x = self.norms[i](x)
            x *= x_mask
            x = self.p1convs[i](x, g)
            x = self.act(x)
            x = self.p2convs[i](x, g)
            x = residual + x
        return x

    def remove_weight_norm(self):
        for c in self.dconvs:
            c.remove_weight_norm()
        for c in self.p1convs:
            c.remove_weight_norm()
        for c in self.p2convs:
            c.remove_weight_norm()


"""
https://github.com/charactr-platform/vocos/blob/main/vocos/heads.py
"""
class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y

class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = Conv1d(dim, out_dim, 1)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        x = torch.cos(p)
        y = torch.sin(p)
        phase = torch.atan2(y, x)
        S = mag * torch.exp(phase * 1j)
        audio = self.istft(S)
        return audio.unsqueeze(1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)