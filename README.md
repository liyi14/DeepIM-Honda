# DeepIM: Deep Iterative Matching for 6D Pose Estimation

PyTorch implementation of the DeepIM framework.

### Introduction

We propose a novel deep neural network for 6D pose matching named DeepIM. Given an initial pose estimation, our network is able to iteratively refine the pose by matching the rendered image against the observed image. The network is trained to predict a relative pose transformation using an untangled representation of 3D location and 3D orientation and an iterative training process. [pdf](https://yuxng.github.io/yili_eccv18.pdf), [Project](https://rse-lab.cs.washington.edu/projects/deepim/)

[![DeepIM](http://yuxng.github.io/DeepIM.png)](https://youtu.be/61DM_WsigY4)

### License

DeepIM is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find DeepIM useful in your research, please consider citing:

    @inproceedings{li2017deepim,
        Author = {Yi Li and Gu Wang and Xiangyang Ji and Yu Xiang and Dieter Fox},
        Title = {DeepIM: Deep Iterative Matching for 6D Pose Estimation},
        booktitle = {European Conference Computer Vision (ECCV)},
        Year = {2018}
    }

### Installation

1. Build conda environment
```
conda env create -f environment.yml
conda activate deepim
```

2. install cupy suitable for your cuda version
```commandline
pip install cupy-cu101
```

3. Build YCB_Renderer
```
cd ycb_render
sudo apt-get install libassimp-dev
pip install -r requirement.txt
python setup.py develop 
```
### Tested environment
- Ubuntu 16.04
- PyTorch 1.7.1
- CUDA 10.1

