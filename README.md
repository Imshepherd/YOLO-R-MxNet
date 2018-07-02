# YOLO v1 with R language ( MxNet library )
(Version 0.1, Last updated :2018.07.02)

#### [MxNet](https://mxnet.apache.org/)：A flexible and efficient library for deep learning.



### 1. Introduction

This is mxnet implementation of the YOLO:Real-Time Object Detection.
YOLO is an unified framework for object detection with a single network. 

It has been originally introduced in this research [article](https://pjreddie.com/media/files/papers/yolo.pdf).

This repository contains a MxNet implementation of a MobileNets_V2-based YOLO networks.

For details with Google's MobileNets, please read the following papers:
- [v1] [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [v2] [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

### 2. Pretrained Models on ImageNet

See: https://github.com/yuantangliang/MobileNet-v2-Mxnet

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):

Network|Top-1|Top-5|sha256sum|Architecture
:---:|:---:|:---:|:---:|:---:
MobileNet v2| 71.90| 90.49| a3124ce7 (13.5 MB)| [netscope](http://ethereon.github.io/netscope/#/gist/d01b5b8783b4582a42fe07bd46243986)

### 3. Pikachu data

For testing model purposes, we’ll train our model to detect Pikachu in the wild. We use a synthetic toy dataset by rendering images from open-sourced 3D Pikachu models. 

For more detail. Please see：
-. https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html.
-. or http://zh.gluon.ai/chapter_computer-vision/pikachu.html.
                         

The dataset consists of 1000 pikachus with random pose/scale/position in random background images. The exact locations are recorded as ground-truth for training and validation.



