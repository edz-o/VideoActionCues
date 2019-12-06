# EN.601.661 Final Project: Exploring Multiple Visual Cues for Human Action Recognition

## Introduction
This is the code repository of the final project for the 19Fall Computer Vision Course (EN.601.661) at JHU. 
The team members are Heather Han, Zili Huang, Yingda Xia and Yi Zhang. The code is based on [MMAction](https://github.com/open-mmlab/mmaction).

## Installation
Please refer to [INSTALL.md](https://github.com/open-mmlab/mmaction/blob/master/INSTALL.md) for installation.

## Data Preparation

We use a subset of [NTU RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset. We provide a script to process the dataset 
and generate necessary files for training and testing.

```shell
bash prepare_nturgbd.sh
```

## Test Pretrained Model

We provide pretrained models for testing. Download them to `modelzoo/`,

```shell
bash download_models.sh
```

Test models on the testset,

```shell
bash test_rgb.sh
```

## Training

We provide a script for training RGB network.

```shell
bash train_rgb.sh
```
