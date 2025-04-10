---
title: 2025-04-01-mt-implementation-log
date: 2025-04-01 12:58:02
tags:
---
123
1. Folder directory problem
The structure of tornike's repo (derive from config):

MusicLDM-Ext
- lightning_logs
    - multichannel_slakh (project check points)
    - musicldm_checkpoints
        - vae-ckpt.ckpt
        - hifigan-ckpt.ckpt
        - clap-ckpt.pt
- data
    - slakh2100 
        - train (https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md)
        - validation
        - test
    <!-- - Audiostock-10k-16khz 
        - test_split_audio_content_analysis.json
        -label -->
- src
- config
    - multichannel_LDM
    - musicldm_audiostock10k
    - musicldm_soundcloud
- train_musicldm.py


2. env
the given musicldm 
 mamba is a good tool for env management (like conda)


nohup mamba env create -f musicldm_env.yml -y --no-builds > install_log.txt 2>&1 &
nohup使用方法
ps aux | grep mamba

torch安装
https://pytorch.org/get-started/previous-versions/

（1）musicldm_env是没有版本号的，只有torch等和原来相同。问题是transformers之类的一些包过于新，不兼容。
（2）目前是去掉哈希值micromamba 安装
nohup micromamba env create -f musicldm_env_clean.yml -y > install_log.txt 2>&1 &
nohup micromamba env update -f musicldm_env_clean.yml -y > install_log.txt 2>&1 &
error 1: torchlibrosa==0.9.2 无法安装
从yaml中去掉，手动装
- 正在 micromamba update
- 手动装 torchlibrosa==0.9.2








































