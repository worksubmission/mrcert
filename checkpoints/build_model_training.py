from math import ceil

import sys
sys.path.append('..')
sys.path.append('../..')

print(sys.path)
from utils.pcure import PatchCURE,SecureLayer,SecurePooling,PatchGuardPooling,CBNPooling,MRPooling,MRPC
from utils.bagnet import BAGNET_FUNC
from utils import bagnet
from utils.vit_srf import vit_base_patch16_224_srf,vit_large_patch16_224_srf
from utils.split import split_resnet50_like,split_vit_like
from timm.models import load_checkpoint

import argparse
import copy
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
import timm

def build_init_pcure_model(args):
    if 'split' in args.model:
        #vitsrf14x2_split3 # 3,6,9
        if 'vitsrf' in args.model or 'vitlsrf' in args.model:
            model_name = args.model.split('.')[0]
            model_name = model_name.split('_')
            i = model_name[0].find('srf')+3
            window_size = model_name[0][i:].split('x')
            window_size = [int(x) for x in window_size]
            split_point = int(model_name[1][5:])
            print('window_size',window_size,'split_point',split_point)
            if 'vitsrf' in args.model:
                vit = create_model('vit_base_patch16_224',global_pool='avg') #the MAE setup
                vitsrf = vit_base_patch16_224_srf(window_size=window_size)
                load_checkpoint(vit,'')
            elif 'vitlsrf' in args.model:
                vit = create_model('vit_large_patch16_224',global_pool='avg') #the MAE setup
                vitsrf = vit_large_patch16_224_srf(window_size=window_size)
                load_checkpoint(vit,'')
            vitsrf.reset_classifier(num_classes=args.num_classes)
            vit.reset_classifier(num_classes=args.num_classes)
            load_checkpoint(vitsrf,''.format(args.dataset, model_name[0])) ######vanilla not masked yet
            print(''.format(args.dataset, model_name[0]))
            # add

            # add
            vitsrf,_ = split_vit_like(vitsrf,split_point,True)
            _,vit = split_vit_like(vit,split_point)
            vit.num_window = vitsrf.num_window
            vit.num_patch = vitsrf.num_patch
            vit.window_size = vitsrf.window_size
            model = nn.Sequential(vitsrf,vit)          
        elif 'bagnet' in args.model: #bagnet33_split1 # 1,2,3 # not used in the paper 
            model_name = args.model.split('.')[0]
            model_name = model_name.split('_')
            model_func = BAGNET_FUNC[model_name[0]]
            bn = model_func(pretrained=False,avg_pool=False)
            load_checkpoint(bn,'checkpoints/{}_vanilla.pth.tar'.format(model_name[0])) ######vanilla not masked yet
            rn = create_model('resnet50',pretrained=True)
            print('model loaded!!!!!!!!!!')
            split_point = int(model_name[1][5:])
            print('split_point',split_point)
            bn,_ = split_resnet50_like(bn,split_point)
            _,rn = split_resnet50_like(rn,split_point)
            model = nn.Sequential(bn,rn)           
    elif 'bagnet' in args.model:
        # if args.num_classes==1000:
        if 'bagnet33' in args.model:
            model = bagnet.bagnet33()
        elif "bagnet17" in args.model:
            model = bagnet.bagnet17()
        elif "bagnet45" in args.model:
            model = bagnet.bagnet45()
        # else:
        #     print("args.num_classes: "+str(args.num_classes))
        #     if 'bagnet33' in args.model:
        #         model = bagnet.bagnet33(num_classes=args.num_classes)
        #     elif "bagnet17" in args.model:
        #         model = bagnet.bagnet17(num_classes=args.num_classes)
        #     elif "bagnet45" in args.model:
        #         model = bagnet.bagnet45(num_classes=args.num_classes)
        # args.num_classes = 1000
        if args.initial_checkpoint:
            print()
            print("load checkpoint "+str(args.initial_checkpoint))
            load_checkpoint(model,args.initial_checkpoint)
            if not args.num_classes==1000:
                model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
            # model.reset_classifier(num_classes=args.num_classes)
            # print("reset: args.num_classes "+str(args.num_classes))
    elif 'mae' in args.model:
        model = create_model('vit_base_patch16_224',global_pool='avg') #the MAE setup
        load_checkpoint(model,'')
        model.reset_classifier(num_classes=args.num_classes)
    elif  'vitsrf' in args.model or 'vitlsrf' in args.model:
        #vitsrf14x2_vanilla
        model_name = args.model.split('.')[0]
        model_name = model_name.split('_')
        i = model_name[0].find('srf')+3
        window_size = model_name[0][i:].split('x')
        window_size = [int(x) for x in window_size]
        if 'vitsrf' in args.model:
            vitsrf = vit_base_patch16_224_srf(window_size=window_size)
            if 'masked' in args.model:
                load_checkpoint(vitsrf,'checkpoints/{}_vanilla.pth.tar'.format(model_name[0])) ######vanilla not masked yet
            else:
                if 'mae' in args.initial_checkpoint:
                    load_checkpoint(vitsrf,'')
                # elif 'original_vit' in args.initial_checkpoint:
                #     vitsrf_ = create_model('vit_base_patch16_224', pretrained=True)
                #     msg = vitsrf.load_state_dict(vitsrf_['model'], strict=True)
                #     print(msg)
                else:
                    NotImplementedError()
        elif 'vitlsrf' in args.model:
            vitsrf = vit_large_patch16_224_srf(window_size=window_size)
            if 'masked' in args.model:
                load_checkpoint(vitsrf,''.format(model_name[0])) ######vanilla not masked yet
            else:
                if 'mae' in args.initial_checkpoint:
                    load_checkpoint(vitsrf, '')
                    print("large!!!")
                # elif 'original_vit' in args.initial_checkpoint:
                #     vitsrf = create_model('vit_large_patch16_224', pretrained=True)
                else:
                    NotImplementedError()
        vitsrf.reset_classifier(num_classes=args.num_classes)
        model = vitsrf
    # model.reset_classifier(num_classes=args.num_classes)
    return model


def build_pcure_model_attacked(args):
    # build and initialize model
    # examples of model name
    # hybrid (SRF+LRF): vitsrf14x2_split9_vanilla
    # SRF-only: vitsrf14x2_masked
    # LRF-only: mae_vanilla
    MODEL_DIR = os.path.join('.', args.model_dir)
    split_point = -1  # default value --> SRF-only or LRF-only mode
    MODEL_NAME = args.model.split('_')[:-1]
    if args.dataset != 'imagenet':
        MODEL_NAME = MODEL_NAME[:-1]  # remove the suffix for dataset name
    if len(MODEL_NAME) == 1:  # LRF-only or SRF-only
        MODEL_NAME = MODEL_NAME[0]
        split_point = -1
    elif len(MODEL_NAME) == 2:
        MODEL_NAME, split_point = MODEL_NAME[0], int(MODEL_NAME[1][5:])
    else:
        raise NotImplementedError('model name not supported')

    lrf = None
    srf = None

    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'cifar':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'gtsrb':
        num_classes = 43
    elif args.dataset == 'imagenette':
        num_classes = 10
    elif args.dataset == 'flowers102':
        num_classes = 102

    else:
        print("error, not supportted datasets")

    # init SRF if needed
    if 'bagnet' in MODEL_NAME:  # bagnet
        model_func = BAGNET_FUNC[MODEL_NAME]
        srf = model_func(pretrained=False, avg_pool=False)
        if split_point < 0:  #
            rf_size = int(MODEL_NAME[6:])
            rf_size = (rf_size, rf_size)
            rf_stride = (8, 8)
        else:
            raise NotImplementedError(
                'only support SRF-only mode for BagNet')  # for hybrid, bagnet RF needs to be changed accordingly
    elif 'vitsrf' in MODEL_NAME or 'vitlsrf' in MODEL_NAME:  # ViT-SRF -- vitsrf for ViT-Base; vitlsrf for ViT-Large
        # parse window size
        i = MODEL_NAME.find('srf') + 3
        window_size = MODEL_NAME[i:].split('x')
        window_size = [int(x) for x in window_size]
        # build vit_srf model
        srf = vit_base_patch16_224_srf(window_size=window_size,
                                       return_features=True) if 'vitsrf' in MODEL_NAME else vit_large_patch16_224_srf(
            window_size=window_size, return_features=True)
        rf_size = (window_size[0] * 16, window_size[1] * 16)
        rf_stride = rf_size

    # init LRF if needed
    if 'resnet50' in MODEL_NAME or ('bagnet' in MODEL_NAME and split_point > 0):
        lrf = timm.create_model('resnet50')
        lrf.reset_classifier(num_classes=num_classes)
    elif 'mae' in MODEL_NAME or ('vitsrf' in MODEL_NAME and split_point >= 0):
        lrf = timm.create_model('vit_base_patch16_224', global_pool='avg')  # the MAE setup
        lrf.reset_classifier(num_classes=num_classes)

    # calculate the corruption size in the secure layer
    patch_size = (args.patch_size, args.patch_size)
    # corruption_size: the "patch size" in the SRF feature map (for secure operation)
    # feature_size: the size of the SRF feature map (for secure operation)
    # mask_stride, mask_size: mask strides and sizes in the feature map
    # Note: for LRF-only (PatchCleanser), the secure operation layer is the input image
    # for now patch size and mask stride are **manually** set.

    data_cfg = {'input_size': (3, 224, 224)}  # hard-coded image size for ImageNet

    if srf:
        corruption_size = ceil((patch_size[0] + rf_size[0] - 1) / rf_stride[0]), ceil(
            (patch_size[1] + rf_size[1] - 1) / rf_stride[1])  # "patch size" in the feature space

        # dry run to get the feature map size
        srf.eval()
        dummy_img = torch.zeros(data_cfg['input_size']).unsqueeze(0)
        feature_shape = srf(dummy_img).shape

        feature_size = (feature_shape[-2], feature_shape[-1])
        mask_stride = (args.mask_stride, args.mask_stride)

    else:  # pure patchcleanser
        corruption_size = patch_size
        feature_size = (data_cfg['input_size'][-2], data_cfg['input_size'][-1])
        mask_stride = (
        args.mask_stride, args.mask_stride)  # num_mask -> mask_stride mapping: {6:33,5:39,4:49,3:65,2:97}

    # calculate mask size
    mask_size = (min(corruption_size[0] + mask_stride[0] - 1, feature_size[0]),
                 min(corruption_size[1] + mask_stride[1] - 1, feature_size[1]))

    print('patch_size', patch_size)
    print('corruption_size', corruption_size)
    print('feature_size', feature_size)
    print('mask_size', mask_size)
    print('mask_stride', mask_stride)

    # construct PatchCURE model and load weights
    checkpoint_name = args.model + '.pth.tar'
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)


    if split_point < 0:  # SRF-only or LRF-only
        model = srf or lrf
        if lrf:
            load_checkpoint(lrf, checkpoint_path)  # ,remap=False)
            secure_layer = SecureLayer(lrf, input_size=feature_size, mask_size=mask_size, mask_stride=mask_stride)
            model = PatchCURE(nn.Identity(), secure_layer)  # no SRF --> use nn.Indentify for the SRF sub-model
        elif srf:
            load_checkpoint(srf, checkpoint_path)  # ,remap=False)
            if args.alg == 'cbn':  # clipped bagnet
                secure_layer = CBNPooling()
            elif args.alg == 'pg':  # patchguard
                secure_layer = PatchGuardPooling(mask_size=mask_size)
            elif args.alg == 'pcure':  # pcure srf-only
                secure_layer = SecurePooling(input_size=feature_size, mask_size=mask_size, mask_stride=mask_stride)
            elif args.alg == 'mr':
                secure_layer = MRPooling(input_size=feature_size, mask_size=mask_size, mask_stride=mask_stride)
            else:
                raise NotImplementedError
            model = PatchCURE(srf, secure_layer)
    else:  # hybrid
        if 'vitsrf' in args.model:
            srf, _ = split_vit_like(srf, split_point, True)  # get the first half of SRF
            _, lrf = split_vit_like(lrf, split_point)  # get the second half of LRF
            lrf.num_window = srf.num_window
            lrf.num_patch = srf.num_patch
            lrf.window_size = srf.window_size
        elif 'bagnet' in args.model:
            raise NotImplementedError('only support SRF-only mode for BagNet')
        # combine SRF and LRF together to instantiate PatchCURE
        if args.alg == 'pcure':
            secure_layer = SecureLayer(lrf, input_size=feature_size, mask_size=mask_size, mask_stride=mask_stride)
        elif args.alg == 'mr':
            secure_layer = MRPC(lrf, input_size=feature_size, mask_size=mask_size, mask_stride=mask_stride)
        else:
            raise NotImplementedError

        model = PatchCURE(srf, lrf)
        # need to create PatchCURE instance before load_checkpoint.
        # need to remap the state_dict *key*. (the names of weight tensors are a bit different; I used nn.Sequential(srf,lrf) during training)
        load_checkpoint(model, checkpoint_path, remap=True)
    return model