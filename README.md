# Regularizing Neural Networks via Minimizing Hyperspherical Energy

By Rongmei Lin, Weiyang Liu, Zhen Liu, Chen Feng, Zhiding Yu, James Rehg, Li Xiong, Le Song

### License 
*CoMHE* is released under the MIT License (refer to the LICENSE file for details).

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Short Video Introduction](#short-video-introduction)
0. [Requirements](#requirements)
0. [Usage](#usage)
0. [Results](#results)

### Introduction
Inspired by the Thomson problem in physics where the distribution of multiple propelling electrons on a unit sphere can be modeled via minimizing some potential energy, hyperspherical energy minimization has demonstrated its potential in regularizing neural networks and improving their generalization power. See our previous work -- [MHE](https://wyliu.com/papers/LiuNIPS18_MHE.pdf) for an in-depth introduction.

Here we propose the compressive minimum hyperspherical energy (CoMHE) as a more effective regularization for neural networks (compared to [the original MHE](https://github.com/wy1iu/MHE)). Specifically, CoMHE utilizes projection mappings to reduce the dimensionality of neurons and minimizes their hyperspherical energy. According to different designs for the projection mapping, we consider several well-performing variants. Welcome to try our CoMHE in your work!

**Our CoMHE is accepted to [CVPR 2020](http://cvpr2020.thecvf.com/) and the full paper is available on [arXiv](https://arxiv.org/abs/1906.04892) and [here](https://wyliu.com/papers/LiuCVPR20.pdf).**

<img src="asserts/teaser.png" width="55%" height="55%">

### Citation
If you find our work useful in your research, please consider to cite:

    @InProceedings{Lin20CoMHE,
        title={Regularizing Neural Networks via Minimizing Hyperspherical Energy},
        author={Lin, Rongmei and Liu, Weiyang and Liu, Zhen and Feng, Chen and Yu, Zhiding 
         and Rehg, James M. and Xiong, Li and Song, Le},
        booktitle={CVPR},
        year={2020}
    }

### Short Video Introduction
We also provide a short video introduction to help interested readers quickly go over our work and understand the essence of CoMHE. Please click the following figure to watch the Youtube video.

[![DCNet_talk](https://img.youtube.com/vi/vXxt_ggWW8s/0.jpg)](https://youtu.be/vXxt_ggWW8s)


### Requirements
1. `Python 3.6` 
2. `TensorFlow 1.14.0`

### Usage

#### Part 1: Clone the repositary
```Shell  
git clone https://github.com/rmlin/CoMHE.git
```
#### Part 2: Download official CIFAR-100 training and testing data (python version)
```Shell  
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

#### Part 3: Train and test with the following code in different folder. 
```Shell
# run random projection CoMHE
cd random_projection
python train.py
```
```Shell
# run angle-preserving projection CoMHE
cd angle_projection
python train.py
```

```Shell
# run adversarial projection CoMHE
cd adversarial_projection
python train.py
```
If you want to change the hyperparameter settings of CoMHE, please refer to the file [train.py](https://github.com/rmlin/CoMHE/blob/master/random_projection/train.py) for different input arguments such as dimension and number of projections. 

### Results
The expected training and testing dynamics (loss and accuracy) can be found in the corresponding log folder. 

- Random projection CoMHE: [log](random_projection/log/pd40_pn20)
- Angle-preserving projection CoMHE: [log](angle_projection/log/pd30_pn1_iter1)
- Adversarial projection CoMHE: [log](adversarial_projection/log/pd30_pn1_iter10/)

### Contact

- [Rongmei Lin](mailto:rongmei.lin@emory.edu)
- [Weiyang Liu](https://wyliu.com)
