import os
from collections import OrderedDict
from typing import *

import torch


def write_config(state_dict: Dict[str, Any], cfg: Dict[str, Any]):
    state_dict["config"] = []
    for key, x in cfg.items():
        state_dict["config"].append(x)
    state_dict["params"] = cfg


def create_trained_model(
    weights: Dict[str, Any],
    version: Literal["v1", "v2", "v3"],
    sr: str,
    f0: bool,
    emb_name: str,
    emb_ch: int,
    emb_output_layer: int,
    epoch: int,
):
    state_dict = OrderedDict()
    state_dict["weight"] = {}
    for key in weights.keys():
        if "enc_q" in key:
            continue
        state_dict["weight"][key] = weights[key].half()
    if version == "v3":
        write_config(
            state_dict,
            {
                "spec_channels": 1025,
                "segment_size": 32,
                "inter_channels": 768,
                "hidden_channels": 768,
                "filter_channels": 768,
                "n_heads": 6,
                "n_layers": 6,
                "kernel_size": 5,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [
                3,
                7,
                11
                ],
                "resblock_dilation_sizes": [
                [
                    1,
                    3,
                    5
                ],
                [
                    1,
                    3,
                    5
                ],
                [
                    1,
                    3,
                    5
                ]
                ],
                "upsample_rates": [
                5,
                6,
                4,
                4
                ],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [
                16,
                16,
                4,
                4,
                4
                ],
                "use_spectral_norm": False,
                "gin_channels": 256,
                "emb_channels": 768,
                "spk_embed_dim": 109,
                "sr": 48000,
            },
        )
    else:
        if sr == "40k":
            write_config(
                state_dict,
                {
                    "spec_channels": 1025,
                    "segment_size": 32,
                    "inter_channels": 192,
                    "hidden_channels": 192,
                    "filter_channels": 768,
                    "n_heads": 2,
                    "n_layers": 6,
                    "kernel_size": 3,
                    "p_dropout": 0,
                    "resblock": "1",
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [10, 10, 2, 2],
                    "upsample_initial_channel": 512,
                    "upsample_kernel_sizes": [16, 16, 4, 4],
                    "spk_embed_dim": 109,
                    "gin_channels": 256,
                    "emb_channels": emb_ch,
                    "sr": 40000,
                },
            )
        elif sr == "48k":
            print(version, "48k saved")
            write_config(
                state_dict,
                {
                    "spec_channels": 1025,
                    "segment_size": 32,
                    "inter_channels": 192,
                    "hidden_channels": 192,
                    "filter_channels": 768,
                    "n_heads": 2,
                    "n_layers": 6,
                    "kernel_size": 3,
                    "p_dropout": 0,
                    "resblock": "1",
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [10, 6, 2, 2, 2],
                    "upsample_initial_channel": 512,
                    "upsample_kernel_sizes": [16, 16, 4, 4, 4],
                    "spk_embed_dim": 109,
                    "gin_channels": 256,
                    "emb_channels": emb_ch,
                    "sr": 48000,
                },
            )
        elif sr == "32k":
            write_config(
                state_dict,
                {
                    "spec_channels": 513,
                    "segment_size": 32,
                    "inter_channels": 192,
                    "hidden_channels": 192,
                    "filter_channels": 768,
                    "n_heads": 2,
                    "n_layers": 6,
                    "kernel_size": 3,
                    "p_dropout": 0,
                    "resblock": "1",
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [10, 4, 2, 2, 2],
                    "upsample_initial_channel": 512,
                    "upsample_kernel_sizes": [16, 16, 4, 4, 4],
                    "spk_embed_dim": 109,
                    "gin_channels": 256,
                    "emb_channels": emb_ch,
                    "sr": 32000,
                },
            )
    state_dict["version"] = version
    state_dict["info"] = f"{epoch}epoch"
    state_dict["sr"] = sr
    state_dict["f0"] = 1 if f0 else 0
    state_dict["embedder_name"] = emb_name
    state_dict["embedder_output_layer"] = emb_output_layer
    return state_dict


def save(
    model,
    version: Literal["v1", "v2", "v3"],
    sr: str,
    f0: bool,
    emb_name: str,
    emb_ch: int,
    emb_output_layer: int,
    filepath: str,
    epoch: int,
):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    print(f"save: emb_name: {emb_name} {emb_ch}")

    state_dict = create_trained_model(
        state_dict,
        version,
        sr,
        f0,
        emb_name,
        emb_ch,
        emb_output_layer,
        epoch,
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state_dict, filepath)

