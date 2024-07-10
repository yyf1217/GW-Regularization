# GW-Regularization
This repository contains the code implementation for the paper titled "Improving Hyperbolic Representations via Gromov-Wasserstein Regularization".


## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

This repository provides the implementation for the algorithms presented in the paper "Improving Hyperbolic Representations via Gromov-Wasserstein Regularization".
Our approach demonstrates consistent enhancements over current state-of-the-art methods across various tasks, including few-shot image classification, as well as semi-supervised graph link prediction and node classification. This code includes:

- **protonet_gw** for few-shot image classification.
- **hnn_gw** for semi-supervised graph link prediction and node classification.

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
