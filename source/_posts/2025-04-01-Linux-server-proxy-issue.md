---
title: 2025-04-01-Linux-server-proxy-issue
date: 2025-04-01 15:23:11
tags:
---

今天试图解决国内服务器网络连接国外网站失败问题。

我是mac m3 pro芯片。

首先在linux服务器上安装aria2。

## aria2 下载
如果有sudo权限，就很简单。
```
sudo apt install aria2
```

如果没有，
aria2最新版本可查看：https://github.com/aria2/aria2/releases/  可以将release-1.37.0换成最新版本

太长不看简略版
```
cd ~
wget https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0.tar.gz
tar -xzvf aria2-1.37.0.tar.gz
cd aria2-1.37.0

./configure --prefix=$HOME/.local

make -j$(nproc)       # 使用多核加速编译
make install          # 安装到 ~/.local

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc     # 立即生效

aria2c --version

```




## m3 microsocks 下载
(~~我也不知道为什么只能ssh~~)

```
git clone git@github.com:rofl0r/microsocks.git
cd microsocks
```

因为我是m3芯片所以make会不同
``` make CC=clang ```
如果是x86之类，这样就可以
```make```

运行
```
./microsocks -p 1080
```