# Efficient Single Image Super-Resolution Using Dual Path Connections with Multiple Scale Learning

This repository is for EMSRDPN introduced in the following paper

Bin-Cheng Yang and [Gangshan Wu](http://mcg.nju.edu.cn), "Efficient Single Image Super-Resolution Using Dual Path Connections with Multiple Scale Learning", [[arxiv]](http://arxiv.org/abs/2112.15386)

It's an extension to a conference paper

Bin-Cheng Yang. 2019. Super Resolution Using Dual Path Connections. In Proceedings of the 27th ACM International Conference on Multimedia (MM ’19), October 21–25, 2019, Nice, France. ACM, NewYork, NY, USA, 9 pages. https://doi.org/10.1145/3343031.3350878

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 16.04 environment (Python3.7, PyTorch_1.1.0, CUDA9.0) with Titan X/Xp/V100 GPUs.

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

Deep convolutional neural networks have been demonstrated to be effective for SISR in recent years. On the one hand, residual connections and dense connections have been used widely to ease forward information and backward gradient flows to boost performance. However, current methods use residual connections and dense connections separately in most network layers in a sub-optimal way. On the other hand, although various networks and methods have been designed to improve computation efficiency, save parameters, or utilize training data of multiple scale factors for each other to boost performance, it either do super-resolution in HR space to have a high computation cost or can not share parameters between models of different scale factors to save parameters and inference time. To tackle these challenges, we propose an efficient single image super-resolution network using dual path connections with multiple scale learning named as EMSRDPN. By introducing dual path connections inspired by Dual Path Networks into EMSRDPN, it uses residual connections and dense connections in an integrated way in most network layers. Dual path connections have the benefits of both reusing common features of residual connections and exploring new features of dense connections to learn a good representation for SISR. To utilize the feature correlation of multiple scale factors, EMSRDPN shares all network units in LR space between different scale factors to learn shared features and only uses a separate reconstruction unit for each scale factor, which can utilize training data of multiple scale factors to help each other to boost performance, meanwhile which can save parameters and support shared inference for multiple scale factors to improve efficiency. Experiments show EMSRDPN achieves better performance and comparable or even better parameter and inference efficiency over SOTA methods.

## Train
### Prepare training data

1. Download DIV2K training data (800 training images for x2, x3, x4 and x8) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and Flickr2K training data (2650 training images) from [Flickr2K dataset](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar). 

2. Untar the download files.

3. Using src/generate_LR_x8.m to generate x8 LR data for Flickr2K dataset, you need to modify 'folder' in src/generate_LR_x8.m to your directory to place Flickr2K dataset.

4. Specify '--dir_data' in src/option.py to your directory to place DIV2K and Flickr2K datasets.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).


### Begin to train

1. Cd to 'src', run the following scripts to train models.

    **You can use scripts in file 'demo.sh' to train models for our paper.**

    To train a fresh model using DIV2K dataset
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train DIV2K
    ```

    To train a fresh model using Flickr2K dataset
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K
    ```

    To train a fresh model using both DIV2K and Flickr2K datasets to reproduce results in the paper, you need copy all the files in DIV2K_HR/ to Flickr2K_HR/, copy all the directories in DIV2K_LR_bicubic/ to Flickr2K_LR_bicubic/, then using the following script
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train Flickr2K
    ```

    To continue a unfinished model using DIV2K dataset, the processes for other datasets are similiar
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348 --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 2 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --resume -1 --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train DIV2K --load EMSRDPN_BIx2348
    ```
## Test
### Quick start
1. Download benchmark dataset from [BaiduYun](https://pan.baidu.com/s/1Bl8TUHywC1HUHoamUFdCew)(提取码：20v5), place them in directory specified by '--dir_data' in src/option.py, untar it.

2. Download model for our paper from [BaiduYun](https://pan.baidu.com/s/1SeHnbEupTBrYdWDSGMJ4dQ)(提取码：d2ov) and place them in 'experiment/'.

3. Cd to 'src', run the following scripts to test downloaded model.

    **You can use scripts in file 'demo.sh' to produce results for our paper.**

    To test a trained model
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348.pt --test_only --save_results
    ```

    To test a trained model using self ensemble
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test+ --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5+Set14+B100+Urban100+Manga109 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348.pt --test_only --save_results --self_ensemble
    ```

    To test a trained model using multiple scale infer
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3.7 main.py --scale 2+3+4+8 --test_scale 2+3+4+8 --save EMSRDPN_BIx2348_test_multi_scale_infer --model EMSRDPN --epochs 5000 --batch_size 16 --patch_size 48 --n_GPUs 1 --n_threads 16 --SRDPNconfig A --ext sep --data_test Set5 --reset --decay 1000-2000-3000-4000-5000 --lr_patch_size --data_range 1-3450 --data_train DIV2K --pre_train ../experiment/EMSRDPN_BIx2348.pt --test_only --save_results --multi_scale_infer
    ```

## Results
All the test results can be download from [BaiduYun](https://pan.baidu.com/s/11YpqTTE076ns7yMTrkxoNQ)(提取码：oawz).

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

@inproceedings{2019Super,
  title={Super Resolution Using Dual Path Connections},
  author={ Yang, Bin Cheng },
  booktitle={the 27th ACM International Conference},
  year={2019},
}

@misc{yang2021efficient,
      title={Efficient Single Image Super-Resolution Using Dual Path Connections with Multiple Scale Learning}, 
      author={Bin-Cheng Yang and Gangshan Wu},
      year={2021},
      eprint={2112.15386},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their code.
