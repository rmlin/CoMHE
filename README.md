# CoMHE
Implementation for &lt;Regularizing Neural Networks via Minimizing Hyperspherical Energy> in CVPR'20.

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

#### Part 3: Train and test with the following code
```Shell  
python train.py
```
please refer to the code for details of parameters such as dimension and number of projections

### Citation

If you find our work useful in your research, please consider to cite:

    @InProceedings{Lin20CoMHE,
        title={Regularizing Neural Networks via Minimizing Hyperspherical Energy},
        author={Lin, Rongmei and Liu, Weiyang and Liu, Zhen and Feng, Chen and Yu, Zhiding 
         and Rehg, James M. and Xiong, Li and Song, Le},
        booktitle={CVPR},
        year={2020}}
