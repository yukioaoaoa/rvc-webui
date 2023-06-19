これはv3モデルの試作用リポジトリです。
それ以外の用途の方はフォーク元に移動してください。

v3の学習済みの重みは以下で提供しています(VCTKで20エポックのみの事前学習のため品質は不十分です)
https://huggingface.co/datasets/nadare/rvc-v3j

vocos-versionは現在開発中です。しばらくお待ちください

# Launch

## Windows
Double click `webui-user.bat` to start the webui.

## Linux or Mac
Run `webui.sh` to start the webui.

<br >

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
- [`teftef6220/Voice_Separation_and_Selection`](https://github.com/teftef6220/Voice_Separation_and_Selection)
