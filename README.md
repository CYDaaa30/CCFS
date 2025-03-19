# Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation

<div align="center">
    <img width="80%" alt="CCFS-Architecture" src="./figures/architecture.png">
</div>

Official PyTorch implementation of the paper **Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation (CVPR 2025)**. This repo contains code for conducting CCFS on CIFAR-10/100 and Tiny-ImageNet based on the already distilled data.

## Abstract
Dataset distillation (DD) excels in synthesizing a small number of images per class (IPC) but struggles to maintain its effectiveness in high-IPC settings.
Recent works on dataset distillation demonstrate that combining distilled and real data can mitigate the effectiveness decay. 
However, our analysis of the combination paradigm reveals that the current one-shot and independent selection mechanism induces an incompatibility issue between distilled and real images. 
To address this issue, we introduce a novel curriculum coarse-to-fine selection (CCFS) method for efficient high-IPC dataset distillation.
CCFS employs a curriculum selection framework for real data selection, where we leverage a coarse-to-fine strategy  to select appropriate real data based on the current synthetic dataset in each curriculum.
Extensive experiments validate CCFS, surpassing the state-of-the-art by +6.6\% on CIFAR-10, +5.8\% on CIFAR-100, and +3.4\% on Tiny-ImageNet under high-IPC settings.
Notably, CCFS achieves 60.2\% test accuracy on ResNet-18 with a 20\% compression ratio of Tiny-ImageNet, closely matching full-dataset training with only 0.3\% degradation.

## Usage

### Requirements

```
pandas==2.2.3
torch==2.2.1
torchvision==0.17.1
tqdm==4.66.2
```
### How to Run

To conduct a curriculum coarse-to-fine selection based on the distilled data, you need to prepare a distilled images folder, a relabel teacher checkpoint, and difficulty scores for corresponding dataset.

Difficulty scores for the 3 datasets are provided in scores/.

The teacher checkpoints can be download here:

CCFS can be extended to almost all dataset distillation methods, as long as you have a copy of the already distilled data and organize it into an image folder structure.
In the main table of our paper, we used the distilled data by the CDA method, you can download the data here. 
We encourage adopting different distilled data by other DD methods and configuring corresponding data augmentation and training settings to verify the scalability of CCFS.

## How to Run
Since the ß

# CCFS
Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation

We provide the experimental procedures for CIFAR-10 with IPC=500, CIFAR-100 with IPC=50, and Tiny-ImageNet with IPC=100 in the form of Jupyter Notebook files.
- ccfs_cifar10_ipc500.ipynb: CCFS on CIFAR-10 with IPC=500 (compression ratio=10%)
- ccfs_cifar100_ipc50.ipynb: CCFS on CIFAR-100 with IPC=50 (compression ratio=10%)
- ccfs_tiny_ipc100.ipynb: CCFS on Tiny-ImageNet with IPC=100 (compression ratio=20%)

![Architecture](./figures/architecture.png)

**Architecture of our curriculum coarse-to-fine selection method for high-IPC dataset distillation, CCFS.** CCFS adopts a combination of distilled and real data to construct the final synthetic dataset. We apply a curriculum framework and select the optimal real data for the current synthetic dataset in each curriculum. (a) **Curriculum selection framework**: CCFS begins the curriculum with the already distilled data as the initial synthetic dataset. Then continuously incorporates real data into the current synthetic dataset through the coarse-to-fine selection within each curriculum phase. (b) **Coarse-to-fine selection strategy**: In the coarse stage, CCFS trains a filter model on the current synthetic dataset and evaluates it on the original dataset excluding already selected data to filter out all correctly classified samples. In the fine stage, CCFS selects the simplest misclassified samples and incorporates them into the current synthetic dataset for the next curriculum.

![Results](./figures/results.png)
