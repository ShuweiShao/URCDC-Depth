# URCDC-Depth: Uncertainty Rectified Cross-Distillation with CutFlip for Monocular Depth Estimation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/urcdc-depth-uncertainty-rectified-cross/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=urcdc-depth-uncertainty-rectified-cross) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/urcdc-depth-uncertainty-rectified-cross/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=urcdc-depth-uncertainty-rectified-cross)

This is the official PyTorch implementation of the method described in

> **URCDC-Depth: Uncertainty Rectified Cross-Distillation with CutFlip for Monocular Depth Estimation** 

[[Link to paper]](https://arxiv.org/abs/2302.08149)
>
> [Shuwei Shao](https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ), Zhongcai Pei, [Weihai Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ), Ran Li, Zhong Liu and [Zhengguo Li](https://scholar.google.com.hk/citations?hl=zh-CN&user=LiUX7WQAAAAJ)
>

## Overview

We have released the code of CutFlip, which has been incorporated into the dataloader.py.  Apart from the results shown in the article, we apply the CutFip to different monocular depth estimation algorithms on the KITTI dataset, such as BTS, TransDepth and Adabins,

<p align="center">
<img src='images/additional_results.png' width=800/> 
</p>

The complete source code will be available upon the acceptance.

## ✏️ 📄 Citation

If you find our work useful in your research please consider citing our paper:

```
@article{shao2023urcdc,
  title={URCDC-Depth: Uncertainty Rectified Cross-Distillation with CutFlip for Monocular Depth Estimatione},
  author={Shao, Shuwei and Pei, Zhongcai and Chen, Weihai and Li, Ran and Liu, Zhong and Li, Zhengguo},
  journal={https://arxiv.org/abs/2302.08149},
  year={2023},
}
```

## Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Models](#models)
6. [Demo](#demo)

## Installation
```
conda create -n newcrfs python=3.8
conda activate newcrfs
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib, tqdm, tensorboardX, timm, mmcv
```


## Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts), and then modify the data path in the config files to your dataset locations.


## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python newcrfs/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python newcrfs/train.py configs/arguments_train_kittieigen.txt
```

## Evaluation
Evaluate the NYUv2 model:
```
python newcrfs/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the KITTI model:
```
python newcrfs/eval.py configs/arguments_eval_kittieigen.txt
```

## Models
| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | a1 | a2 | a3| SILog| 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|[NYUv2](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/models/model_nyu.ckpt) | 0.0952 | 0.0443 | 0.3310 | 0.1185 | 0.923 | 0.992 | 0.998 | 9.1023 |
|[KITTI_Eigen](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/models/model_kittieigen.ckpt) | 0.0520 | 0.1482 | 2.0716 | 0.0780 | 0.975 | 0.997 | 0.999 | 6.9859 |


## Demo
Test images with the indoor model:
```
python newcrfs/test.py --data_path datasets/test_data --dataset nyu --filenames_file data_splits/test_list.txt --checkpoint_path model_nyu.ckpt --max_depth 10 --save_viz
```

Play with the live demo from a video or your webcam:
```
python newcrfs/demo.py --dataset nyu --checkpoint_path model_zoo/model_nyu.ckpt --max_depth 10 --video video.mp4
```

![Output1](files/output_nyu1_compressed.gif)

[Demo video1](https://www.youtube.com/watch?v=RrWQIpXoP2Y)

[Demo video2](https://www.youtube.com/watch?v=fD3sWH_54cg)

[Demo video3](https://www.youtube.com/watch?v=IztmOYZNirM)

## Contact

If you have any questions, please feel free to contact swshao@buaa.edu.cn.


## Acknowledgement

Our code is based on the implementation of [BTS](https://github.com/cleinc/bts) and [NewCRFs](https://github.com/aliyun/NeWCRFs). We thank their excellent works.