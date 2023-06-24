
# VoRAS (Vocos with Retrieval and self-Augmentation for Speechs)
(This repository is currently in alpha. After completing the training of the pre-learning model, it will be a beta version.)

Welcome to VoRAS. VoRAS is a model derived from RVC for fast and lightweight real-time voice change in Japanese.

<br >

VoRAS is based on RVC, replacing the vocoder with Vocos and rewriting the overall model architecture into a modern structure.

<br >

## Note
VoRAS is currently in beta version and will be ported to latopia in the future. In the future, the structure of the model will change and compatibility will be lost, and until the official version, it is assumed that only the RTX series or higher GPU will be used for learning in a local Windows environment. Please understand.


# Launch

## Windows
Double click `webui-user.bat` to start the webui.

```
Tested environment: Windows 10, Python 3.10.9, torch 2.0.0+cu118
```

<br >

# Troubleshooting

## `error: Microsoft Visual C++ 14.0 or greater is required.`

Microsoft C++ Build Tools must be installed.

### Step 1: Download the installer
[Download](https://visualstudio.microsoft.com/ja/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)

### Step 2: Install `C++ Build Tools`
Run the installer and select `C++ Build Tools` in the `Workloads` tab.

<br >

# Credits
- [`liujing04/Retrieval-based-Voice-Conversion-WebUI`](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
- [`charactr-platform/Vocos`](https://github.com/charactr-platform/vocos)
- [`ddPn08/rvc-webui`](https://github.com/ddPn08/rvc-webui/tree/main)
- [`NVIDIA/BigVGAN`](https://github.com/NVIDIA/BigVGAN)
- [`/rinna/japanese-hubert-base`](https://huggingface.co/rinna/japanese-hubert-base)
- [`teftef6220/Voice_Separation_and_Selection`](https://github.com/teftef6220/Voice_Separation_and_Selection)
