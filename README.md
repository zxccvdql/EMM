# Efficient Multi-Task Modeling through Automated Fusion of Trained Models

This repository is for EMM method.

## Introduction

Multi-task learning plays a pivotal role in the field of intelligent services. Although multi-task learning has been widely applied in numerous terminal services, traditional multi-task modeling methods often require custom design for specific task combinations, resulting in a cumbersome and complex process. Inspired by the rapid development and outstanding performance of single-task models, this paper proposes an efficient multi-task modeling method that can automatically fuse trained single-task models. This method serves as a general framework, allowing designers to simply select the required tasks for multi-task modeling. This not only simplifies the modeling process but also enables the direct utilization of trained single-task models, eliminating the need to overly focus on task interrelations and model structure design.
To achieve this goal, we deeply consider the structural and task differences among various trained models and employ model decomposition techniques to break these models down into multiple operable components. Furthermore, we design a Transformer-based Adaptive Knowledge Fusion (AKF) module, which can adaptively integrate the knowledge of similar tasks within the components. Through the proposed method, we achieve efficient construction of multi-task models and validate the effectiveness of this method through extensive experiments on three datasets.

<div align="center">
  <img src="Figs/fig1.png" width="100%">
  Overview of our method.
</div>

## Prerequisites

### Experiment Environment

Our implementation is in Pytorch.
We use `python3.11` and  please refer to [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) to create a `python3.11` conda environment.
Install the listed packages in the virual environment:

```
pip install torch-rechub
```

### Datasets

A total of three multi-task datasets were used, namely Ali-CCP, AliExpress, and Census-Income.

#### Ali-CCP

This dataset is collected from the recommendation system logs of Taobao’s mobile app. It consists of 23 sparse features and 8 dense features, including two labels: “click” and “purchase”.
The dataset covers information from 250,000 users and 500,000 products, totaling 80 million data entries.

- Notes
  - The original data has already been divided into a training set and a test set. During preprocessing, half of the original test set is randomly allocated as a validation set. The ratio of the preprocessed training set, validation set, and test set is 2:1:1.
  - The sparse features of the preprocessed dataset have undergone Label Encoding, while the dense features have been normalized. We also provide a script named `data\ali-ccp\preprocess_ali_ccp.py` for processing the original data.
- Original data address: [This Link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408).
- Download address for the preprocessed full dataset: [This Link](https://cowtransfer.com/s/1903cab699fa49).

#### AliExpress

AliExpress data contains 16 sparse features and 63 dense features, labeled with "exposure", "click", and "conversion". The dense features in the original data have already undergone normalization preprocessing. The original data consists of 5 CSV files, but only the data from the US region is used for testing here.

- Original data address: [This Link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690&lang=en-us).
- Download address for the preprocessed full dataset: [This Link](https://cowtransfer.com/s/7080e52e5f4f4a).

#### Census-Income

This dataset consists of US census income data, which is used to predict income levels (below 50k or above 50k). After preprocessing, it contains a total of 41 columns, including 7 dense features, 33 sparse features, and 1 label column.

- Notes
  - In Experiment of the MMOE paper and in the PLE approach, income prediction is treated as the main task, while marital status prediction serves as an auxiliary task.
  - To consistently test all multitask models, we follow the ESMM setup, where income prediction is considered a CTR (Click-Through Rate) task, and marital status prediction is treated as a CVR (Conversion Rate) task.
  - For reference on how to process the original data, please see `data\census-income\preprocess_census.py`.
- Original data address: [This Link](http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)).
- Download address for the preprocessed full dataset: [This Link](https://cowtransfer.com/s/e8b67418ce044c).

## Train Pre-trained Model

To train a single-task pre-trained model as the foundation for model fusion, you can refer to `single_datasetname.ipynb` for experimentation.

## Train Baseline and Our Method

Please refer to the details in `multi_task_datasetname.ipynb` for training both the Baseline model and the model of our proposed method.

## Acknowledgement
Some dataloading and evaluation code is from:
https://github.com/datawhalechina/torch-rechub