# MRCert

This is the code for MRCert.

 We will upload our checkpoints/metadata after the acceptance of the paper.

## Environment

The code is implemented in Python==3.10, timm==0.9.16, torch==1.13.1.

## Datasets

- [ImageNet](https://image-net.org/download.php) (ILSVRC2012)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNette](https://github.com/fastai/imagenette)

## Files

├── [pcure_attack.py](https://github.com/mr-cert/mrcert/blob/main/pcure_attack.py)    #call the attacker to perform patch attacks

├── [patch_attacker.py](https://github.com/mr-cert/mrcert/blob/main/patch_attacker.py)   #The patch attacker

├── [main.py](https://github.com/mr-cert/mrcert/blob/main/main.py)              #obtain the eva results and save the metadata

├── [prediction_map_analysis.py](https://github.com/mr-cert/mrcert/blob/main/prediction_map_analysis.py)    #data analysis for certification

├── checkpoints

|   ├── [build_model_training.py](https://github.com/mr-cert/mrcert/blob/main/checkpoints/build_model_training.py)                    #build VIT-SRF for training

|   ├── [train.py](https://github.com/mr-cert/mrcert/blob/main/checkpoints/train.py)                  #the training script from timm following PC



## Demo

0. You may need to configure the location of datasets and checkpoints ([build_model_training.py](https://github.com/mr-cert/mrcert/blob/main/checkpoints/build_model_training.py)  needs to be configured manually).

1. First, finetune base DL models. You may check [PatchCURE/checkpoints/README.md at main · inspire-group/PatchCURE](https://github.com/inspire-group/PatchCURE/blob/main/checkpoints/README.md) for further information, from which we adopt the finetuning process. 

  ```python
python -u train.py   --model vitsrf14x1_split6 --dataset imagenet  --opt adamw     --batch-size 64   --layer-decay 0.65   --weight-decay 0.05 --drop-path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --amp --epochs 100 --lr-base 5e-5 --num-classes 1000
  ```

2. Then, evaluate the finetuned ViT-SRF with mutants.

  ```python
python -u main.py --dataset imagenet --model vitsrf14x1_split6_masked --patch_size 32
  ```



3. Finally, perform the data analysis for evaluation on certification.

  ```python
python -u prediction_map_analysis.py --dataset imagenet --model vitsrf14x1_split6_masked --patch_size 32
  ```



4. For attacks, use pcure_attack.py like Step 2 and repeat Step 3 for analysis.

   

5. For VOT, please refer to [kio-cs/CrossCert](https://github.com/kio-cs/CrossCert), using [train_drs.py](https://github.com/kio-cs/CrossCert/blob/main/train_drs.py) for finetuning, [certification_drs.py](https://github.com/kio-cs/CrossCert/blob/main/certification_drs.py) for evaluating the mutants, and the function certified_drs in [CrossCert/utils/new.py at main · kio-cs/CrossCert](https://github.com/kio-cs/CrossCert/blob/main/utils/new.py#L81) for analysis, where in line 92, "gap > 2 * delta" for certification, "gap > 4 * delta" for round-trip certification.

## Acknowledgement

This package is partly built on [inspire-group/PatchCURE](https://github.com/inspire-group/PatchCURE) and [alevine0/patchSmoothing](https://github.com/alevine0/patchSmoothing).
