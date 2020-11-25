# Mirai: A mammography-based model for breast cancer risk

# Introduction
This repository was used to develop Mirai, the risk model described in: [Towards Robust Mammography-Based Models for Breast Cancer Risk](). Mirai was designed to predict risk at multiple time points, leverage potentially missing risk-factor information, and produce predictions that are consistent across mammography machines. Mirai was trained on a large dataset from Massachusetts General Hospital (MGH) in the US and was tested on held-out test sets from MGH, Karolinska in Sweden and Chang Gung Memorial Hospital in Taiwan, obtaining C-indices of 0.76 (0.74, 0.80), 0.81 (0.79, 0.82), 0.79 (0.79, 0.83), respectively. Mirai obtained significantly higher five-year ROC AUCs than the Tyrer-Cuzick model (p<0.001) and prior deep learning models, Hybrid DL (p<0.001) and ImageOnly DL (p<0.001), trained on the same MGH dataset. In our paper, we also demonstrate that Mirai was more significantly accurate in identifying high risk patients than prior methods across all datasets. On the MGH test set, 41.5% (34.4, 48.5) of patients who would develop cancer within five-years were identified as high risk, compared to 36.1% (29.1, 42.9) by Hybrid DL (p=0.02) and 22.9% (15.9, 29.6) by Tyrer-Cuzick lifetime risk (p<0.001).

This code base is meant to achieve a few goals:
- Provide exact implementation details for the development of Mirai
- Help researchers to install Mirai for clinical workflows
- Help researchers to validate Mirai on large datasets
- Help researchers to fine-tune Mirai on large datasets

We note that this code-base is an extension of [OncoNet](https://github.com/yala/OncoNet_Public), which we used to develop Hybrid DL and ImageOnly DL.

## Aside on Software Depedencies
This code assumes python3.6 and a Linux environment.
The package requirements can be install with pip:
`pip install -r requirements.txt`

If you are familiar with docker, you can also directly leverage the OncoServe docker container which has all the depedencies preinstalled(see below).

## Preprocessing
Our code-base operates on PNG images. We converted presentation view dicoms to PNG16 files using the DCMTK library. We used the dcmj2pnm program (v3.6.1, 2015) with +on2 and–min-max-window flags. To this, you can use DCMTK directly or [OncoData](https://github.com/yala/OncoData_Public), our python wrapper for converting dicoms.

## Inclusion Criteria
Mirai expects all four standard “For Presentation” views of the mammogram.  Specifically, it requires an “L CC”, “L MLO”, “R CC”, “R MLO”. It will not work without all four views or with “For Processing” mammograms.  As a result, we unfortunately cannot provide risk assessments for patients with only non-standard views or unilateral mammograms.  All of the mammograms used in our study were captured using either the Hologic Selenia or Selenia Dimensions mammography devices. We have yet tested Mirai on other machines.

# Reproducing Mirai
As described in the supplementary material of the paper, Mirai was trained in two phases; first, we trained the image encoder in conjunction with the risk factor predictor and additive hazard layer to predict breast cancer independently from each view without using conditional adversarial training. In this stage, we intialialized our image encoder with weights from ImageNet, and augmented our training set with random flips and rotations of the original images. We found that adding an adversarial loss at this stage or training the whole architecture end-to-end prevented the model from converging. In the second stage of training, we froze our image encoder, and trained the image aggregation module, the risk factor prediction module, the additive hazard layer, and the device discriminator in a conditional adversarial training regime. We trained our adversary for three steps for every one step of training Mirai. In each stage, we performed small hyperparameter searches and chose the model that obtained the highest C-index on the development set.

The grid searches are shown in :
`configs/mirai_base.json` and `configs/mirai_full.json`

The grid searches were run using our job-dispatcher, as shown bellow.
`python scripts/dispatcher.py --alert_config_path /path/to/secret_for_sms.json --experiment_config_path configs/mirai_base.json --result_path mirai_base_sweep.csv`
We selected the image encoder with the highest C-index on the development set, and leveraged it for the second stage hyper-parameter sweep.
`python scripts/dispatcher.py --alert_config_path /path/to/secret_for_sms.json --experiment_config_path configs/mirai_full.json --result_path mirai_full_sweep.csv`

We note that this command run relies on integrations that were specific to the MGH data, and so the exact line above will not run on your system. The configs above are meant to specify exact implementation details and our experimental procedure.

# Installing Mirai for clinical use
Please see [OncoServe](https://github.com/yala/OncoServe_Public), our framework for deploying mammography-based models in the clinic. OncoServe can be easily installed on premise using Docker, and it provides a simple HTTP interface to get risk assessments for a given patient's dicom files.

# How validate the model on a large dataset



# How to fine-tune the model

