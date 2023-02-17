# URCDC-Depth
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/paper/urcdc-depth-uncertainty-rectified-cross)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=urcdc-depth-uncertainty-rectified-cross) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/paper/urcdc-depth-uncertainty-rectified-cross)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=urcdc-depth-uncertainty-rectified-cross)

This is the official PyTorch implementation of the method described in

> **URCDC-Depth: Uncertainty Rectified Cross-Distillation with CutFlip for Monocular Depth Estimation** [[Link]](https://arxiv.org/abs/2302.08149)
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

## Contact

If you have any questions, please feel free to contact swshao@buaa.edu.cn.


## Acknowledgement

Our code is based on the implementation of [BTS](https://github.com/cleinc/bts) and [NewCRFs](https://github.com/aliyun/NeWCRFs). We thank their excellent works.
