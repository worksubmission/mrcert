import copy
from collections import Counter

import joblib as joblib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse

from tqdm import tqdm
import time

from utils.builder import get_data_loader, build_pcure_model, build_undefended_model
from fvcore.nn import FlopCountAnalysis

parser = argparse.ArgumentParser()

parser.add_argument("--model-dir", default='checkpoints', type=str, help="checkpoints directory")
parser.add_argument('--data-dir', default='', type=str, help="data directory")

parser.add_argument('--dataset', default='imagenet', type=str, help="dataset name")
parser.add_argument("--model", default='vitsrf14x1_split6_masked', type=str,
                    help="model name; see checkpoints/readme.md for more details")
parser.add_argument("--patch-size", default=32, type=int, help="size of the adversarial patch")
parser.add_argument("--batch-size", default=2, type=int, help="batch size for inference")
parser.add_argument("--num-img", default=-1, type=int,
                    help="the number of images for experiments; use the entire dataset if -1")
parser.add_argument("--mask-stride", default=1, type=int, help="the mask stride (double-masking parameter)")
parser.add_argument("--alg", default='pcure', choices=['pcure', 'pg', 'cbn', 'mr'],
                    help="algorithm to use. set to pcure to obtain main results")
parser.add_argument("--certify", default=True, type=bool, help="do certification")

parser.add_argument("--runtime", action='store_true', help="analyze runtime")
parser.add_argument("--flops", action='store_true', help="analyze flops")
parser.add_argument("--memory", action='store_true', help="analyze memory usage")
parser.add_argument("--undefended", action='store_true', help="experiment with undefended models")
parser.add_argument("--gpu", default=0, type=int, help="gpu")

parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
DUMP_DIR = os.path.join('.', args.dump_dir)
if "attack" in args.dump_dir:
    SUFFIX = '_{}_{}_p{}_attacked_500.z'.format(args.dataset, args.model, args.patch_size)
else:
    SUFFIX = '_{}_{}_p{}.z'.format(args.dataset, args.model, args.patch_size)


print(args)

prediction_map_list = joblib.load(os.path.join(DUMP_DIR, 'prediction_map_list' + SUFFIX))
label_list = joblib.load(os.path.join(DUMP_DIR, 'label_list' + SUFFIX))
original_pre_list = joblib.load(os.path.join(DUMP_DIR, 'original_pre_list' + SUFFIX))

prediction_map_list_full=[]
prediction_map_list_copy=copy.deepcopy(prediction_map_list)

for prediction_map_idx in range(len(prediction_map_list)):
    prediction_map=prediction_map_list[prediction_map_idx]
    # pred_map = torch.where(prediction_map == -1, y[img_i], pred_map)
    #
    # # pred_map = pred_map + pred_map.T - torch.diag(torch.diag(pred_map))
    # results[img_i] = torch.all(pred_map == y[img_i])

    print("running")
    n=prediction_map.shape[0]
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             for q in range(n):
    for i in range(n):
        for j in range(i,n):
            for k in range(j,n):
                for q in range(k,n):
                    if prediction_map[i,j,k,q] != -1:
                        value = prediction_map[i,j,k,q]

                        prediction_map[j, i, k, q] = value
                        prediction_map[j, i, q, k] = value
                        prediction_map[j, k, i, q] = value
                        prediction_map[j, k, q, i] = value
                        prediction_map[j, q, i, k] = value
                        prediction_map[j, q, k, i] = value
                        prediction_map[i, j, k, q] = value
                        prediction_map[i, j, q, k] = value
                        prediction_map[i, k, j, q] = value
                        prediction_map[i, k, q, j] = value
                        prediction_map[i, q, j, k] = value
                        prediction_map[i, q, k, j] = value
                        prediction_map[k, j, i, q] = value
                        prediction_map[k, j, q, i] = value
                        prediction_map[k, i, j, q] = value
                        prediction_map[k, i, q, j] = value
                        prediction_map[k, q, j, i] = value
                        prediction_map[k, q, i, j] = value
                        prediction_map[q, j, i, k] = value
                        prediction_map[q, j, k, i] = value
                        prediction_map[q, i, j, k] = value
                        prediction_map[q, i, k, j] = value
                        prediction_map[q, k, j, i] = value
                        prediction_map[q, k, i, j] = value
                    else:
                        AttributeError

    prediction_map_list_full.append(prediction_map)

# prediction_map_list=prediction_map_list_full
def prediction_label_produce(prediction_map_list,label_list):
    prediction_output_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        flag=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]
        # # if two mask results all the same
        if np.all(prediction_map_two_mask == prediction_map_two_mask.flat[0]):
            # output this label
            prediction_output_list.append(prediction_map_two_mask.flat[0])
            flag=True
            print("inference Case 1")
        #     if not
        else:
            # check three mask
            for j in range(len(prediction_map_three_mask)):
                if flag==False:
                # if exist one mask, all its additional two mask results is the same
                    if np.all(prediction_map_three_mask[j] == prediction_map_three_mask[j,j,j]):
                        # output this label
                        prediction_output_list.append(prediction_map_three_mask[j,j,j])
                        flag=True
                        print("inference Case 2")
                #     if not
            if flag==False:
                # if exist two mask, all its additional two mask results is the same
                for a in range(len(prediction_map)):
                    for b in range(a,len(prediction_map)):
                        if a == b:
                            continue
                        if np.all(prediction_map[a][b] == prediction_map[a,a,b,b]):
                            prediction_output_list.append(prediction_map[a,a,b,b])
                            flag = True
                            print("inference Case 3")
                            break
                        if flag==True:
                            break
                    if flag == True:
                        break
        if flag==False:
            most_common_element = np.bincount(prediction_map_one_mask.flat).argmax()
            prediction_output_list.append(most_common_element)
            print("inference Case 4")

    return prediction_output_list

def prediction_label_produce_with_original_pre(prediction_map_list,label_list, original_pre_list):
    prediction_output_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        original_pre=original_pre_list[i]
        flag=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]
        # # if two mask results all the same
        if np.all(prediction_map_two_mask == original_pre):
            # output this label
            prediction_output_list.append(original_pre)
            flag=True
            print("inference Case 1")
        #     if not
        else:
            # check three mask
            for j in range(len(prediction_map_three_mask)):
                if flag==False:
                # if exist one mask, all its additional two mask results is the same
                    if np.all(prediction_map_three_mask[j] == prediction_map_three_mask[j,j,j]):
                        # output this label
                        prediction_output_list.append(prediction_map_three_mask[j,j,j])
                        flag=True
                        print("inference Case 2")
                #     if not
            if flag==False:
                # if exist two mask, all its additional two mask results is the same
                for a in range(len(prediction_map)):
                    for b in range(a,len(prediction_map)):
                        if a == b:
                            continue
                        if np.all(prediction_map[a][b] == prediction_map[a,a,b,b]):
                            prediction_output_list.append(prediction_map[a,a,b,b])
                            flag = True
                            print("inference Case 3")
                            break
                        if flag==True:
                            break
                    if flag == True:
                        break
        if flag==False:
            # most_common_element = np.bincount(prediction_map_one_mask.flat).argmax()
            # prediction_output_list.append(most_common_element)
            prediction_output_list.append(original_pre)
            print("inference Case 4")

    return prediction_output_list

def prediction_label_produce_original_pc(prediction_map_list,label_list):
    prediction_output_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        flag=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        # prediction_map_three_mask = prediction_map[:, :, indices, indices]
        # prediction_map_two_mask= prediction_map_three_mask[:, indices, indices]
        # prediction_map_one_mask= prediction_map_two_mask[indices, indices]
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]
        # pc
        most_common_element = np.bincount(prediction_map_one_mask).argmax()
        flag=False
        if np.all(prediction_map_one_mask[0]==prediction_map_one_mask):
            prediction_output_list.append(prediction_map_one_mask[0])
        else:
            for i in range(len(prediction_map_two_mask)):
                if prediction_map_two_mask[i,i] == most_common_element:
                    continue
                if np.all(prediction_map_two_mask[i,i] == prediction_map_two_mask[i]) and flag==False:
                    prediction_output_list.append(prediction_map_two_mask[i,i])
                    flag=True
            if flag==False:
                prediction_output_list.append(most_common_element)


    return prediction_output_list

def prediction_label_produce_original_pc_with_original_pre(prediction_map_list,label_list, original_pre_list):
    prediction_output_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        original_pre=original_pre_list[i]
        flag=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        # prediction_map_three_mask = prediction_map[:, :, indices, indices]
        # prediction_map_two_mask= prediction_map_three_mask[:, indices, indices]
        # prediction_map_one_mask= prediction_map_two_mask[indices, indices]
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]
        # pc
        # most_common_element = np.bincount(prediction_map_one_mask).argmax()
        flag=False
        if np.all(original_pre==prediction_map_one_mask):
            prediction_output_list.append(original_pre)
        else:
            for i in range(len(prediction_map_two_mask)):
                if prediction_map_two_mask[i,i] == original_pre:
                    continue
                if np.all(prediction_map_two_mask[i,i] == prediction_map_two_mask[i]) and flag==False:
                    prediction_output_list.append(prediction_map_two_mask[i,i])
                    flag=True
            if flag==False:
                prediction_output_list.append(original_pre)


    return prediction_output_list

def prediction_label_produce_original_pc_with_original_pre_for_two_patches(prediction_map_list,label_list, original_pre_list):
    prediction_output_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        original_pre=original_pre_list[i]
        flag=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        # prediction_map_three_mask = prediction_map[:, :, indices, indices]
        # prediction_map_two_mask= prediction_map_three_mask[:, indices, indices]
        # prediction_map_one_mask= prediction_map_two_mask[indices, indices]
        prediction_map_four_mask=prediction_map
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]
        # pc
        # most_common_element = np.bincount(prediction_map_one_mask).argmax()
        flag=False
        if np.all(original_pre==prediction_map_two_mask):
            prediction_output_list.append(original_pre)
        else:
            for a in range(len(prediction_map_four_mask)):
                for b in range(len(prediction_map_four_mask)):
                    if prediction_map_four_mask[a,a,b,b] == original_pre:
                        continue
                    if np.all(prediction_map_four_mask[a,a,b,b] == prediction_map_four_mask[a,b]) and flag==False:
                        prediction_output_list.append(prediction_map_four_mask[a,a,b,b])
                        flag=True
            if flag==False:
                prediction_output_list.append(original_pre)


    return prediction_output_list

# def prediction_label_produce_original_pc_debugging(prediction_map):
#
#     # three/two-mask result
#     m, p = prediction_map.shape
#     indices = np.arange(m)
#     # prediction_map_three_mask = prediction_map[:, :, indices, indices]
#     # prediction_map_two_mask= prediction_map[:, indices, indices, indices]
#     # prediction_map_one_mask= prediction_map[indices, indices, indices, indices]
#     prediction_map = prediction_map + prediction_map.T - torch.diag(torch.diag(prediction_map))
#     prediction_map=prediction_map.cpu().numpy()
#     # pc
#     most_common_element = np.bincount(prediction_map[indices,indices]).argmax()
#     flag=False
#     if np.all(prediction_map[indices,indices][0]==prediction_map[indices,indices]):
#         return prediction_map[indices,indices][0]
#     else:
#         for i in range(len(prediction_map)):
#             if prediction_map[i,i] == most_common_element:
#                 continue
#             if np.all(prediction_map[i,i] == prediction_map[i]) and flag==False:
#                 return prediction_map[i,i]
#         if flag==False:
#             return most_common_element

def one_cert_produce_with_original_pre(prediction_map_list,label_list,original_pre_list):
    cert_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        original_pre=original_pre_list[i]
        cert=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]

        # if np.all(prediction_map_three_mask.flat[0]==prediction_map_three_mask):
        #     cert=True
        if np.all(original_pre==prediction_map_three_mask):
            cert=True
            print("cert Case 1")

        else:
            label_cert=None
            for j in range(len(prediction_map)):
                if np.all(prediction_map[j,j,j,j] == prediction_map[j]):
                    if label_cert==None:
                        label_cert=prediction_map[j,j,j,j]
                    # elif not label_cert==prediction_map[j,j,j,j]:
                    #     print("wrong!!!!")
                        print("cert Case 2")
                    cert=True
        cert_list.append(cert)

    return cert_list

def one_cert_produce(prediction_map_list,label_list):
    cert_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        cert=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]

        # if np.all(prediction_map_three_mask.flat[0]==prediction_map_three_mask):
        #     cert=True
        if np.all(prediction_map_three_mask.flat[0]==prediction_map_three_mask):
            cert=True
            print("cert Case 1")

        else:
            label_cert=None
            for j in range(len(prediction_map)):
                if np.all(prediction_map[j,j,j,j] == prediction_map[j]):
                    if label_cert==None:
                        label_cert=prediction_map[j,j,j,j]
                    # elif not label_cert==prediction_map[j,j,j,j]:
                    #     print("wrong!!!!")
                        print("cert Case 2")
                    cert=True
        cert_list.append(cert)

    return cert_list

def one_cert_produce_pc(prediction_map_list,label_list, original_pre_list):
    cert_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        original_pre=original_pre_list[i]
        cert=False

        # three/two-mask result
        d, n, m, p = prediction_map.shape
        indices = np.arange(m)
        prediction_map_three_mask = prediction_map[:, :, indices, indices]
        prediction_map_two_mask= prediction_map[:, indices, indices, indices]
        prediction_map_one_mask= prediction_map[indices, indices, indices, indices]

        # original pc
        # if np.all(label==prediction_map[:, :, indices, indices]):
        if np.all(original_pre==prediction_map_two_mask):
            cert = True
        cert_list.append(cert)

    return cert_list

def double_cert_produce(prediction_map_list,label_list):
    cert_list=[]
    for i in range(len(prediction_map_list)):
        prediction_map=prediction_map_list[i]
        label=label_list[i]
        cert=False

        if np.all(label==prediction_map):
            cert=True
        cert_list.append(cert)

    return cert_list


prediction_output_list=prediction_label_produce_with_original_pre(prediction_map_list_full,label_list, original_pre_list)
one_cert_list=one_cert_produce_with_original_pre(prediction_map_list_full,label_list, original_pre_list)
one_cert_list_pc=one_cert_produce_pc(prediction_map_list_full,label_list, original_pre_list)
prediction_output_list_pc=prediction_label_produce_original_pc_with_original_pre(prediction_map_list,label_list, original_pre_list)
double_cert_list=double_cert_produce(prediction_map_list_full,label_list)
prediction_output_list_pc_two_patches=prediction_label_produce_original_pc_with_original_pre_for_two_patches(prediction_map_list,label_list, original_pre_list)



def convert_array_to_int(lst):
    result = []
    for item in lst:
        if isinstance(item, np.ndarray):
            if item.size == 1:
                result.append(int(item.item()))
            else:
                raise ValueError(f"Array {item} contains multiple elements, cannot convert to single int")
        else:
            result.append(int(item) if isinstance(item, (float, np.floating)) else item)
    return result

prediction_output_list=convert_array_to_int(prediction_output_list)
prediction_output_list_pc=convert_array_to_int(prediction_output_list_pc)
label_list=convert_array_to_int(label_list)
prediction_output_list_pc_two_patches=convert_array_to_int(prediction_output_list_pc_two_patches)


correct=0
one_cert_correct=0
double_cert=0

correct_pc=0
one_cert_pc_correct=0
correct_pc_two_patches=0

one_cert=0
one_cert_pc=0

those_correct_inside_cert=0
those_correct_inside_cert_pc=0


for i in range(len(prediction_output_list)):
    if prediction_output_list[i]==label_list[i]:
        correct=correct+1
        if one_cert_list[i]==True:
            one_cert_correct=one_cert_correct+1
        if double_cert_list[i]==True:
            double_cert=double_cert+1
    if prediction_output_list_pc[i]==label_list[i]:
        correct_pc=correct_pc+1
        if one_cert_list_pc[i]==True:
            one_cert_pc_correct=one_cert_pc_correct+1
    if one_cert_list[i]:
        one_cert=one_cert+1
        if prediction_output_list[i]==label_list[i]:
            those_correct_inside_cert=those_correct_inside_cert+1

    if one_cert_list_pc[i]:
        one_cert_pc=one_cert_pc+1
        if prediction_output_list_pc[i]==label_list[i]:
            those_correct_inside_cert_pc=those_correct_inside_cert_pc+1
    # for tow pathes in pc
    if prediction_output_list_pc_two_patches[i]==label_list[i]:
        correct_pc_two_patches=correct_pc_two_patches+1


num=len(prediction_output_list)

print("correct/num "+str(correct/num))
print("one_cert_correct/num "+str(one_cert_correct/num))
print("double_cert/num "+str(double_cert/num))
print("those_correct_inside_cert/one_cert "+str(those_correct_inside_cert/one_cert))

print("\n")

print("correct_pc/num "+str(correct_pc/num))
print("correct_pc_two_patches/num "+str(correct_pc_two_patches/num))

print("one_cert_pc_correct/num "+str(one_cert_pc_correct/num))


print("one_cert_pc "+str(one_cert_pc))
print("those_correct_inside_cert_pc/one_cert_pc "+str(those_correct_inside_cert_pc/one_cert_pc))
# print("one_cert_pc "+str(one_cert_pc))
correct_original=0
for i in range(len(original_pre_list)):
    original_pre=original_pre_list[i]
    label=label_list[i]
    if original_pre==label:
        correct_original=correct_original+1

print("correct_original/num "+str(correct_original/num))








