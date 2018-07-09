# DAWNBench Analysis

## Introduction

[DAWNBench](https://dawn.cs.stanford.edu/benchmark/) is an end-to-end deep learning training and inference benchmark. Computation time and cost are critical resources in building deep models, yet many existing benchmarks focus solely on model accuracy.

This repository contains the analysis of DAWNBench results. We will analyze two common deep learning workloads, CIFAR-10 and ImageNet time-to-accuracy and training cost across different optimization strategies, model architectures, software frameworks, clouds, and hardware.

For the start, we will analyze only CIFAR-10 workload across two model architectures:

- Custom Wide ResNet
- ResNet18 (TODO)

|Rank | Time to 94% Accuracy | Model | Framework | Hardware |
| --- | --- | --- | --- | --- |
| 1 | 00:52:11 | Custom Wide ResNet + AdamW + modified 1 cycle policy <br /> Cedric Chee <br /> [source](/src/models/wrn) | fastai / PyTorch  0.4.0 | 1 K80 (AWS p2.xlarge) |
| 2 | 00:55:51 | Custom Wide ResNet + modified 1 cycle policy <br /> Cedric Chee <br /> [source](/src/models/wrn) | fastai / PyTorch  0.4.0 | 1 K80 (AWS p2.xlarge) |
| 3 | 0:59:38 | Custom DarkNet <br /> Cedric Chee <br /> [source](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/cifar10-darknet.ipynb) | PyTorch 0.3.1 | 1 K80 (AWS p2.xlarge) |
| 4 | 1:18:22 | Custom Wide ResNet <br /> fast.ai students team <br /> [source](https://github.com/fastai/imagenet-fast/commit/1bc5aeec765572397c0ebe4a5d616d03beeeeec1) | fastai / PyTorch 0.4.0 | 1 K80 (AWS p2.xlarge) |

## Analysis

### 1. Custom Wide ResNet fast.ai students team DAWNBench submission (Baseline)

[Jupyter Notebook / Source](https://nbviewer.jupyter.org/github/cedrickchee/dawnbench-analysis/blob/master/src/cifar10_custom_wrn_dawnbench.ipynb#fastai-DAWN-bench-submission)

![Training](/images/cifar10_fastai_dawnbench_submission_training.png)

![Loss and learning rate plot](/images/cifar10_fastai_dawnbench_submission_loss_lr_plot.png)

### 2. My Custom Wide ResNet + AdamW + modified 1 cycle policy hyper-parameters

[Jupyter Notebook / Source](https://nbviewer.jupyter.org/github/cedrickchee/dawnbench-analysis/blob/master/src/cifar10_custom_wrn_adamw.ipynb)

![Training](/images/cifar10_fastai_adamw_training.png)

![Loss and learning rate plot](/images/cifar10_fastai_adamw_loss_lr_plot.png)

### 3. My Custom Wide ResNet + modified 1 cycle policy hyper-parameters

[Jupyter Notebook / Source](https://nbviewer.jupyter.org/github/cedrickchee/dawnbench-analysis/blob/master/src/cifar10_custom_wrn_dawnbench.ipynb#With-tweaks-for-training-on-AWS-p2.xlarge-K80-GPU)

![Training](/images/cifar10_custom_wrn_training.png)

![Loss and learning rate plot](/images/cifar10_custom_wrn_loss_lr_plot.png)

## Code to replicate all analyses

### Instructions for building and running the container.

You will need to have [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed in order to run this.

1. cd into cloned repo.
2. `docker build -t cifar10 .`
3. `./run_container.sh cifar10`

Once the Docker container started, all you have to do is access Jupyter Notebook through this url, https://localhost:8888 in your web browser. Then, enter `jupyter` as password. Open the notebook and click 'run all cells'.