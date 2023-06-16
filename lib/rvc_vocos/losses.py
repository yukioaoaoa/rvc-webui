import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F


class MelLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max, eps=1e-7, device="cuda"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            window_fn=torch.hann_window,
            center=True,
            power=1,
            norm="slaney",
            mel_scale="slaney"
        )

    def forward(self, x_true, x_pred):
        x_true = torch.nn.functional.pad(
            x_true,
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
        )

        x_pred = torch.nn.functional.pad(
            x_pred,
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
        )
        x_pred = x_pred

        S_true = torch.log(torch.clamp(self.melspec(x_true), min=self.eps))
        S_pred = torch.log(torch.clamp(self.melspec(x_pred), min=self.eps))

        loss = F.l1_loss(S_true, S_pred)
        return loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean(torch.clamp(1 - dr, min=0))
        g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean(torch.clamp(1 - dg, min=0))
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

