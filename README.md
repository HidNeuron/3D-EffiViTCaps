# 3D-EffiViTCaps: 3D Efffcient Vision Transformer with Capsule for Medical Image Segmentation
Table of Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Trained models](#trained-models)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

# Introduction
![image](network_imgs/3D-EffiViTCaps.jpg)

The figure above illustrates our 3D-EffiViTCaps architecture. Details about it are described in our paper [here](https://arxiv.org/abs/2205.09299). The main implementation of this whole network can be found [here](module/effiViTcaps.py). In addition, the implementation of 3D Patch Merging block and 3D EfficientViT block can be find [here](main_block/efficientViT3D.py). A visualization example is shown below.

![image](network_imgs/3D-EffiViTCaps.jpg)

## Usage

### Installation
- Clone the repository:
```
git clone https://github.com/HidNeuron/3D-EffiViTCaps.git
```

- Install dependencies depends on your cuda version (CUDA 10 or CUDA 11)
```
conda env create -f environment_cuda11.yml
or
conda env create -f environment_cuda10.yml
```

### Data preparation
Our method is evaluated on three datasets:
* iSeg-2017 challenge (infant brain MRI segmentation): <https://iseg2017.web.unc.edu/download/>
* Cardiac and Hippocampus dataset from Medical Segmentation Decathlon: <http://medicaldecathlon.com/>

See this [repository](https://github.com/VinAIResearch/3D-UCaps) for more details on data preparation.

### Training
The training example script is available [here](scripts/train_3d_effiViTcaps.sh)

### Validation
The evaluating example script is available [here](scripts/eval_3d_effiViTcaps.sh)

See this [repository](https://github.com/VinAIResearch/3D-UCaps) for more details on training and evaluating parameters.

## Trained models
Our trained 3DConvCaps models on three datasets can be downloaded as follows: 

- [iSeg-2017](https://uark-my.sharepoint.com/:u:/g/personal/minht_uark_edu/EcXhqKrOfp9BiAhru2x0vwABSeew_qcQ-RSTA8NYmYO0xg?e=ZPI0Nj)
- [Hippocampus](https://uark-my.sharepoint.com/:u:/g/personal/minht_uark_edu/Eag8cZNDQ7FMietkIa4RodMB8dMXcaMS9eXzJfnrubbZTw?e=fWqP3h)
- [Cardiac](https://uark-my.sharepoint.com/:u:/g/personal/minht_uark_edu/EeqQ4YJ9LSZDhpJ8eDehfLMBBGTnY4ovlvkBAHODATe4Lg?e=6NZA8G)

## Acknowledgement
The implementation makes liberal use of code from [3D-UCaps](https://github.com/VinAIResearch/3D-UCaps) and [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT).

## Citation
```
@article{tran20223dconvcaps,
  title={3DConvCaps: 3DUnet with Convolutional Capsule Encoder for Medical Image Segmentation},
  author={Tran, Minh and Vo-Ho, Viet-Khoa and Le, Ngan TH},
  journal={arXiv preprint arXiv:2205.09299},
  year={2022}
}
```
## Contacts
We are honored to helping you if you have any questions. Please feel free to open an issue or contact us directly. Hope our code helps and look forward to your citations.
