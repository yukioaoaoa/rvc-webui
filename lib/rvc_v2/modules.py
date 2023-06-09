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


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


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
        init_weights(self.main_fc)
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


class WaveConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, gin_channels, kernel_sizes, strides, dilations, extend_ratio, r, use_spectral_norm=False):
        super(WaveConv1D, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.convs = []
        # self.norms = []
        self.convs.append(LoRALinear1d(in_channels, inner_channels, gin_channels, r))
        for i, (k, s, d) in enumerate(zip(kernel_sizes, strides, dilations), start=1):
            self.convs.append(norm_f(Conv1d(inner_channels, inner_channels, k, s, dilation=d, groups=inner_channels, padding=get_padding(k, d))))
            if i < len(kernel_sizes):
                self.convs.append(norm_f(Conv1d(inner_channels, inner_channels, 1, 1)))
            else:
                self.convs.append(norm_f(Conv1d(inner_channels, out_channels, 1, 1)))
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x, g, x_mask=None):
        for i, l in enumerate(self.convs):
            if i % 2:
                x_ = l(x)
            else:
                x_ = l(x, g)
            x = F.leaky_relu(x_, modules.LRELU_SLOPE)
            if x_mask is not None:
                x *= x_mask
        return x

    def remove_weight_norm(self):
        for i, c in enumerate(self.convs):
            if i % 2:
                remove_weight_norm(c)
            else:
                c.remove_weight_norm()


class FusedMBConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, gin_channels, kernel_size, stride, extend_ratio, r=1, use_spectral_norm=False):
        super(FusedMBConv2D, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.layers = nn.ModuleList(
            [
                norm_f(Conv2d(in_channels, inner_channels, kernel_size, stride, padding=(get_padding(kernel_size[0], 1), 0))),
                LoRALinear2d(inner_channels, out_channels, gin_channels, r=r),
            ]
        )

    def forward(self, x, g):
        for i, l in enumerate(self.layers):
            if i:
                x = l(x, g)
            else:
                x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
        return x


class MBConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, gin_channels, kernel_size, stride, extend_ratio, r, use_spectral_norm=False):
        super(MBConv2D, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.layers = nn.ModuleList(
            [
                LoRALinear2d(in_channels, inner_channels, gin_channels, r=r),
                norm_f(Conv2d(inner_channels, inner_channels, kernel_size, stride, groups=inner_channels, padding=(get_padding(kernel_size[0], 1), 0))),
                LoRALinear2d(inner_channels, out_channels, gin_channels, r=r),
            ]
        )

    def forward(self, x, g):
        for i, l in enumerate(self.layers):
            if i%2 == 0:
                x = l(x, g)
            else:
                x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
        return x


class SqueezeExcitation1D(torch.nn.Module):
    def __init__(self, input_channels, squeeze_channels, use_spectral_norm=False):
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        super(SqueezeExcitation1D, self).__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = norm_f(torch.nn.Conv1d(input_channels, squeeze_channels, 1))
        self.fc2 = norm_f(torch.nn.Conv1d(squeeze_channels, input_channels, 1))

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = F.leaky_relu(scale, modules.LRELU_SLOPE)
        scale = self.fc2(scale)
        return torch.sigmoid(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input

    def remove_weight_norm(self):
        remove_weight_norm(self.fc1)
        remove_weight_norm(self.fc2)


class ResBlock1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, gin_channels, kernel_sizes, strides, dilations, extend_ratio, r):
        super(ResBlock1, self).__init__()
        norm_f = weight_norm 
        inner_channels = int(in_channels * extend_ratio)
        self.dconvs = nn.ModuleList()
        self.pconvs = nn.ModuleList()
        # self.norms = []
        self.init_conv = LoRALinear1d(in_channels, inner_channels, gin_channels, r)
        for i, (k, s, d) in enumerate(zip(kernel_sizes, strides, dilations)):
            self.dconvs.append(norm_f(Conv1d(inner_channels, inner_channels, k, s, dilation=d, groups=inner_channels, padding=get_padding(k, d))))
            if i < len(kernel_sizes) - 1:
                self.pconvs.append(LoRALinear1d(inner_channels, inner_channels, gin_channels, r))
        self.out_conv = LoRALinear1d(inner_channels, out_channels, gin_channels, r)
        init_weights(self.init_conv)
        self.dconvs.apply(init_weights)
        self.pconvs.apply(init_weights)
        init_weights(self.out_conv)

    def forward(self, x, g, x_mask=None):
        if x_mask is not None:
            x *= x_mask
        x_ = self.init_conv(x, g)
        x = F.leaky_relu(x_, modules.LRELU_SLOPE)
        for i in range(len(self.dconvs)):
            if x_mask is not None:
                x *= x_mask
            x_ = self.dconvs[i](x)
            x = x + F.leaky_relu(x_, modules.LRELU_SLOPE)
            if i < len(self.dconvs) - 1:
                x_ = self.pconvs[i](x, g)
                x = x + F.leaky_relu(x_, modules.LRELU_SLOPE)
        if x_mask is not None:
            x *= x_mask
        x_ = self.out_conv(x, g)
        x = F.leaky_relu(x_, modules.LRELU_SLOPE)
        return x

    def remove_weight_norm(self):
        for c in self.dconvs:
            remove_weight_norm(c)
        for c in self.pconvs:
            c.remove_weight_norm()
        self.init_conv.remove_weight_norm()
        self.out_conv.remove_weight_norm()


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x
