##### Table of contents
1. [Installation](#installation)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)
4. [Results](#results)
5. [Evaluation](#evaluation)
6. [Acknowledgments](#acknowledgments)
7. [Contacts](#contacts)

# Official PyTorch implementation of "On Inference Stability for Diffusion Models" [(AAAI'24)](https://arxiv.org/)

## Installation ##
Python `3.8.0` and Pytorch `2.0.0` are used in this implementation.

<!-- It is recommended to create `conda` env from our provided [environment.yml](./environment.yml):
```
conda env create -f environment.yml
conda activate sadpm
``` -->

<!-- Or you can install neccessary libraries as follows:
```bash
pip install -r requirements.txt
``` -->

## Dataset preparation ##
We trained on five datasets, including CIFAR10, CelebA 64, FFHQ 64, AFHQ 64 and CelebA-HQ 256. 

For CIFAR10, they will be automatically downloaded in the first time execution. 

For FFHQ 64 and AFHQ 64, please download files [here](https://drive.google.com/drive/folders/1QvhF8wfPtnoZY8YMGGEdRlNDUhb0kV3E)

For CelebA 64 and CelebA HQ 256, please check out [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [here](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) for dataset preparation.

Once a dataset is downloaded, please put it in `exp/datasets/` directory as follows:
```
exp/datasets/
├── cifar10
├── celeba
├── celebahq
├── ffhq64
├── afhq
```

## How to run ##
We provide a bash script for our experiments on different datasets. The syntax is following:
```
python train_ddp.py --config <DATASET>.yml \
    --doc <MODEL_NAME> \
    --num_consecutive_steps <K> --lamda <lamda> \
    --ni --num_process_per_node <#GPUS>
```
where:
- `<DATASET>`: `cifar10`, `celeba`, `celebahq`, `afhq`, and `ffhq64`.
- `<K>`: the number of consecutive steps (K in the paper) (e.g. 2, 3, 4).
- `<lamda>`: the coefficient of SA loss (e.g. 0.2, 0.5).
- `<#GPUS>`: the number of used gpus (e.g. 1, 2, 4, 8).

## Results ##
Avaiable soon!

## Evaluation ##
### FID ###
```
python generate.py --config <DATASET>.yml --doc <MODEL_NAME> \
    --image_folder </path/to/save/images> --ckpt_id <ckpt_id> --num_samples 50000 \
    --timesteps <#STEPS> --eta <ETA> --ni --seeds=0-49999 \
    --model_ema --num_process_per_node <#GPUS> \
    --fid_log </path/to/file/saving/log>
```
where:
- `<DATASET>`: `cifar10`, `celeba`, `celebahq`, `afhq`, and `ffhq64`.
- `<#STEPS>`: the number of sampling steps (e.g. 10, 50, 100, 200, 1000).
- `<ckpt_id>`: the id of the best checkpoint  (e.g. 800, 850, 900, ...).
- `<ETA>`: controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM).
- `<#GPUS>`: the number of used gpus (e.g. 1, 2, 4, 8).

## Acknowledgments
This implementation is based on:
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) 
- [https://github.com/NVlabs/denoising-diffusion-gan](https://github.com/NVlabs/denoising-diffusion-gan)
- [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)

## Contacts ##
If you have any problems, please open an issue in this repository or ping an email to [viettmab123@gmail.com](mailto:viettmab123@gmail.com).
