##### Table of contents
1. [Installation](#installation)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)
4. [Models and Hyperparameters](#models-and-hyperparameters)
5. [Evaluation](#evaluation)
6. [Acknowledgments](#acknowledgments)
7. [Contacts](#contacts)

# Official PyTorch implementation of "On Inference Stability for Diffusion Models" [(AAAI'24)](https://arxiv.org/abs/2312.12431)

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

For FID score statistics of the datasets, please download files [here](https://drive.google.com/drive/folders/1_T6Sp1rC8LqqOjhMo9uDX2KHg6beWcBQ) and put it in `pytorch_fid/` directory

## How to run ##
We provide a bash script for our experiments on different datasets. The syntax is following:
```
python train_ddp.py --config <DATASET>.yml \
    --doc <MODEL_NAME> --ni \
    --num_consecutive_steps <K> --lamda <lamda> \
    --num_process_per_node <#GPUS>
```
where:
- `<DATASET>`: `cifar10`, `celeba`, `celebahq`, `afhq`, and `ffhq64`.
- `<K>`: the number of consecutive steps (e.g. 2, 3, 4).
- `<lamda>`: the coefficient of SA loss (e.g. 0.2, 0.5).
- `<#GPUS>`: the number of used gpus (e.g. 1, 2, 4, 8).

## Models and Hyperparameters ##

CelebA 64x64 with the SA_DPM loss [[checkpoint](https://drive.google.com/drive/folders/1L-2EPsD5cZ07XA55ke1gzx38NLOSoOyF)]
```bash
python train_ddp.py --config celeba.yml --doc celeba_sa --ni --num_consecutive_steps 2 --lamda 1 --num_process_per_node 4
```

AFHQv2 64x64 with the base loss [[checkpoint](https://drive.google.com/drive/folders/1P85qx4PjhDbN10ke2OeJeZlTH4Ls6AFn)]
```bash
python train_ddp.py --config afhq.yml --doc afhq_simple --ni --num_consecutive_steps 0 --lamda 0 --num_process_per_node 4
```

AFHQv2 64x64 with the SA_DPM loss [[checkpoint](https://drive.google.com/drive/folders/1Tj0HiA0qBJ3k5_ot2FI9jBPK75oJdcty)]
```bash
python train_ddp.py --config afhq.yml --doc afhq_sa --ni --num_consecutive_steps 2 --lamda 0.2 --num_process_per_node 4
```

FFHQ 64x64 with the base loss [[checkpoint](https://drive.google.com/drive/folders/1V2s3MFXgT4kVsZMjrtmiKBnZ2ALKePkz)]
```bash
python train_ddp.py --config ffhq64.yml --doc ffhq_simple --ni --num_consecutive_steps 0 --lamda 0 --num_process_per_node 4
```

FFHQ 64x64 with the SA_DPM loss [[checkpoint](https://drive.google.com/drive/folders/1ET8SXPBh-3OPk3oOEPMhimqce4M2KKxk)]
```bash
python train_ddp.py --config ffhq64.yml --doc ffhq_sa --ni --num_consecutive_steps 2 --lamda 0.5 --num_process_per_node 4
```

CelebA-HQ 256x256 with the base loss [[checkpoint](https://drive.google.com/drive/folders/15T9CvC1rfok1ky4m7LZiI0qQvm5ARw8W)]
```bash
python train_ddp.py --config celebahq.yml --doc celebahq_simple --ni --num_consecutive_steps 0 --lamda 0 --num_process_per_node 4
```

CelebA-HQ 256x256 with the SA_DPM loss [[checkpoint](https://drive.google.com/drive/folders/1Ce9TFx4lb57eyY121Y4-aIfx-3xZ2bah)]
```bash
python train_ddp.py --config celebahq.yml --doc celebahq_sa --ni --num_consecutive_steps 2 --lamda 0.1 --num_process_per_node 4
```

Downloaded pre-trained models should be put in should be put in `exp/logs/<MODEL_NAME>` (e.g. `<MODEL_NAME>`:  celeba_sa, afhq_simple, ...)

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

Example: 
```bash
# Generate 50000 images of AFHQv2 dataset using DDIM sampling with 200 steps (4 GPUs)
python generate.py --config afhq.yml --doc afhq_sa --image_folder ./afhq_sa/1500_DDIM_T200 --ckpt_id 1500 --num_samples 50000 \
    --timesteps 200 --eta 0 --ni --seeds=0-49999 --model_ema --num_process_per_node 4 --fid_log fid_afhq_sa.txt

# Generate 10000 images of CelebA-HQ dataset using DDPM sampling with 100 steps (4 GPUs)
python generate.py --config celebahq.yml --doc celebahq_simple --image_folder ./celebahq_simple/700_DDPM_T100 --ckpt_id 700 --num_samples 10000 \
    --timesteps 100 --eta 1 --ni --seeds=0-9999 --model_ema --num_process_per_node 4 --fid_log fid_celebahq_simple.txt
```


## Acknowledgments
This implementation is based on:
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) 
- [https://github.com/NVlabs/denoising-diffusion-gan](https://github.com/NVlabs/denoising-diffusion-gan)
- [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)

## Contacts ##
If you have any problems, please open an issue in this repository or ping an email to [viettmab123@gmail.com](mailto:viettmab123@gmail.com).
