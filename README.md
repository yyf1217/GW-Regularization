# GW-Regularization
This repository contains the code implementation for the paper titled "Improving Hyperbolic Representations via Gromov-Wasserstein Regularization".


## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Overview

This repository provides the implementation for the algorithms presented in the paper "Improving Hyperbolic Representations via Gromov-Wasserstein Regularization".
Our approach demonstrates consistent enhancements over current state-of-the-art methods across various tasks, including few-shot image classification, as well as semi-supervised graph link prediction and node classification. This code includes:

- **protonet_gw** for few-shot image classification. We apply our GW regularization on the hyperbolic ProtoNet model [[1]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf).
- **hnn_gw** for semi-supervised graph link prediction and node classification. We consider the following three HNNs: the vanilla HNN [[2]](https://proceedings.neurips.cc/paper/2018/file/dbab2adc8f9d078009ee3fa810bea142-Paper.pdf), HGCN [[3]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf), and HyboNet [[4]](https://arxiv.org/pdf/2105.14686).

## Setup

### Installation

1. **Create a virtual environment**:

    ```sh
    conda create -n gw_regularization python=3.8.13
    ```

2. **Install dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running Experiments

1. **Dataset**
   
    Few-shot image classification task:
    To replicate the results on CUB and MiniImageNet, follow the instructions [here](https://github.com/leymir/hyperbolic-image-embeddings/tree/master/examples/fewshot).
    
    Semi-supervised graph link prediction and node classification task:
    download dataset from /hnn_gw/data.

2. **Training**: Use the training script to train the model.

    Few-shot image classification task:
   
    1 shot 5 way
    ```sh
    run /protonet_gw/reproduce_1s_5w.sh
    ```
   
    1 shot 5 way
    ```sh
    run /protonet_gw/reproduce_5s_5w.sh
    ```

    Semi-supervised graph link prediction and node classification task:
   
    link prediction
    ```sh
    run /hnn_gw/reproduce_lp.sh
    ```

    node classification
    ```sh
    run /hnn_gw/reproduce_nc.sh
    ```
   

## Acknowledgements

YY and DZ acknowledge funding from National Natural Science Foundation of China (NSFC) under award number 12301117, WL acknowledges funding from the National Institute of Standards and Technology (NIST) under award number 70NANB22H021, and GL acknowledges funding from NSF award DMS 2124913.


## References

[1] [Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., Lempitsky, V.: Hyperbolic image embeddings. In: IEEE Conf. Comput. Vis. Pattern Recog. (2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf)

[2] [Ganea, O., Bécigneul, G., Hofmann, T.: Hyperbolic neural networks. Adv. Neural Inform. Process. Syst. 31, 5345–5355 (2018)](https://proceedings.neurips.cc/paper/2018/file/dbab2adc8f9d078009ee3fa810bea142-Paper.pdf)

[3] [Chami, I., Ying, Z., Ré, C., Leskovec, J.: Hyperbolic graph convolutional neural networks. Adv. Neural Inform. Process. Syst. 32, 4868–4879 (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf)

[4] [Chen, W., Han, X., Lin, Y., Zhao, H., Liu, Z., Li, P., Sun, M., Zhou, J.: Fully hyperbolic neural networks. In: Annu. Meeting Assoc. Comput. Linguistics. pp.5672–5686 (2022)](https://arxiv.org/pdf/2105.14686)

