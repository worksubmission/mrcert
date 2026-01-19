import sys

import joblib as joblib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse

from tqdm import tqdm
import time
from checkpoints.build_model_training import build_init_pcure_model, build_pcure_model_attacked
from utils.builder import get_data_loader, build_pcure_model, build_undefended_model
from fvcore.nn import FlopCountAnalysis
from patch_attacker import PatchAttacker

parser = argparse.ArgumentParser()
NUM_CLASSES_DICT = {'imagenette':10,'imagenet':1000,'flower102':102,'cifar':10,'cifar100':100,'svhn':10,'gtsrb':43}

# parser.add_argument("--model-dir",default='checkpoints',type=str,help="checkpoints directory")
parser.add_argument("--model-dir",
                    default='',
                    type=str, help="checkpoints directory")
#
# parser.add_argument('--data-dir', default='', type=str, help="data directory")
parser.add_argument('--data-dir', default='', type=str,help="data directory")

parser.add_argument('--dataset', default='cifar', type=str, help="dataset name")
# parser.add_argument("--model",default='vitsrf14x2_split11_masked',type=str,help="model name; see checkpoints/readme.md for more details")
parser.add_argument("--model", default='vitsrf14x1_split6_masked_cifar', type=str,
                    help="model name; see checkpoints/readme.md for more details")
# parser.add_argument("--attacked_model", default='mae_finetuned_vit_base', type=str,
#                     help="model name; see checkpoints/readme.md for more details")
parser.add_argument("--patch-size", default=32, type=int, help="size of the adversarial patch")
parser.add_argument("--batch-size", default=1, type=int, help="batch size for inference")
parser.add_argument("--num-img", default=-1, type=int,
                    help="the number of images for experiments; use the entire dataset if -1")
# parser.add_argument("--num-mask",default=-1,type=int,help="the number of mask used in double-masking")
parser.add_argument("--mask-stride", default=1, type=int, help="the mask stride (double-masking parameter)")
parser.add_argument("--alg", default='pcure', choices=['pcure', 'pg', 'cbn', 'mr'],
                    help="algorithm to use. set to pcure to obtain main results")
# parser.add_argument("--certify",action='store_true',help="do certification")
parser.add_argument("--certify", default=True, type=bool, help="do certification")

parser.add_argument("--runtime", action='store_true', help="analyze runtime")
parser.add_argument("--flops", action='store_true', help="analyze flops")
parser.add_argument("--memory", action='store_true', help="analyze memory usage")
parser.add_argument("--undefended", action='store_true', help="experiment with undefended models")
parser.add_argument("--gpu", default=0, type=int, help="gpu")

parser.add_argument("--steps", default=150, type=int, help='directory to dump two-mask predictions')
parser.add_argument("--step_size", default=0.05, type=float, help='directory to dump two-mask predictions')
parser.add_argument("--randomizations", default=80, type=int, help='directory to dump two-mask predictions')
# parser.add_argument("--steps", default=1, type=int, help='directory to dump two-mask predictions')
# parser.add_argument("--step_size", default=0.05, type=float, help='directory to dump two-mask predictions')
# parser.add_argument("--randomizations", default=1, type=int, help='directory to dump two-mask predictions')
parser.add_argument("--attack_num", default=500, type=int, help='directory to dump two-mask predictions')
parser.add_argument("--num_classes", default=10, type=int, help='directory to dump two-mask predictions')


parser.add_argument("--dump_dir", default='dump_attack', type=str, help='directory to dump two-mask predictions')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
DUMP_DIR = os.path.join('.', args.dump_dir)
SUFFIX = '_{}_{}_p{}_attacked_{}.z'.format(args.dataset, args.model, args.patch_size,args.attack_num)

print(args)

val_loader = get_data_loader(args)
model_attacked = build_pcure_model_attacked(args)
if args.undefended:
    model = build_undefended_model(args)
else:
    model = build_pcure_model(args)
device = 'cuda'
model = model.to(device)
model.eval()

model_attacked = model_attacked.to(device)
model_attacked.eval()
cudnn.benchmark = True
# cudnn.benchmark = False
# cudnn.deterministic = True
max_value = float('-inf')
min_value = float('inf')
for data, target in val_loader:
    max_value = max(max_value, torch.max(data).item())
    min_value = min(min_value, torch.min(data).item())
print(max_value)
print(min_value)
attacker = PatchAttacker(model_attacked, [0.,0.,0.],[1.,1.,1.], ub=[max_value, max_value, max_value], lb=[min_value, min_value, min_value],kwargs={
    'epsilon':1.0,
    'random_start':True,
    'steps':args.steps,
    'step_size':args.step_size,
    'num_classes':NUM_CLASSES_DICT[args.dataset],
    'patch_l':args.patch_size,
    'patch_w':args.patch_size
})

correct_undefended = 0
correct = 0
correct_original=0
total = 0
certify = 0
data_time = []
inference_time = []
certification_time = []
flops_total = 0
memory_allocated = []
memory_reserved = []
start_time = time.time()

prediction_map_list = []
label_list = []
original_pre_list=[]
attacked_sample_list=[]

if args.runtime:  # dry run
    data = torch.ones(args.batch_size, 3, 224, 224).cuda()
    for i in range(3):
        model(data)

np.random.seed(42)
random_idx_array=np.random.choice(len(val_loader), args.attack_num, replace=False)
print(len(np.unique(random_idx_array)))
random_idx_list=random_idx_array.tolist()
for idx, (data, labels) in enumerate(tqdm(val_loader)):
    if idx not in random_idx_list:
        continue
    else:
        print("hit "+str(idx))

    total += len(labels)
    # sys.stdout.flush()
    # if total==args.attack_num+1:
    #     total=total-1
    #     break
    # data loading
    if args.runtime:
        torch.cuda.synchronize()
        a = time.time()

    data = data.to(device)
    labels = labels.to(device)
    attacked = attacker.perturb(data, labels, float('inf'), random_count=args.randomizations)
    if args.runtime:
        torch.cuda.synchronize()
        data_time.append(time.time() - a)

    if args.flops:
        flops = FlopCountAnalysis(model, data)
        flops_total += flops.total() / 1e9
        continue

    # inference
    if args.runtime:
        torch.cuda.synchronize()
        a = time.time()
    if args.memory:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(attacked)
        print("attacked pre")
        if args.memory:
            memory_allocated.append(torch.cuda.max_memory_allocated() / 1048576)  # to MB
            memory_reserved.append(torch.cuda.max_memory_reserved() / 1048576)
        if args.runtime:
            torch.cuda.synchronize()
            inference_time.append(time.time() - a)

        if args.undefended:
            output = torch.argmax(output, dim=1)  # logits vector -> prediction label

        correct += torch.sum(output == labels).item()

        # certification
        if args.certify:
            if args.runtime:
                torch.cuda.synchronize()
                a = time.time()

            results, pred_map, original_pres = model.certify(attacked, labels)
            attacked_sample_list.append(attacked.cpu().numpy())

            print("eva attacked")
            for map in pred_map:
                prediction_map_list.append(map.detach().cpu().numpy())
            for label in labels:
                label_list.append(label.detach().cpu().numpy())
            for original_pre in original_pres:
                original_pre_list.append(original_pre.detach().cpu().numpy())
            correct_original += torch.sum(original_pre == labels).item()

            if args.runtime:
                torch.cuda.synchronize()
                certification_time.append(time.time() - a)

            certify += torch.sum(results).item()
            print(f'Clean Accuracy: {correct / total}')
            print(f'correct_original Accuracy: {correct_original / total}')
            print(f'Certified Robust Accuracy: {certify / total}')

joblib.dump(prediction_map_list, os.path.join(DUMP_DIR, 'prediction_map_list' + SUFFIX))
joblib.dump(label_list, os.path.join(DUMP_DIR, 'label_list' + SUFFIX))
joblib.dump(original_pre_list, os.path.join(DUMP_DIR, 'original_pre_list' + SUFFIX))
joblib.dump(attacked_sample_list, os.path.join(DUMP_DIR, 'attack_sample_list' + SUFFIX))


print(len(prediction_map_list))
print(len(random_idx_list))

print(random_idx_list)
print(f'Clean Accuracy: {correct / total}')
if args.certify:
    print(f'Certified Robust Accuracy: {certify / total}')
if args.runtime:
    data_time = np.sum(data_time) / total
    inference_time = np.sum(inference_time) / total
    certification_time = np.sum(certification_time) / total
    print(f'Throughput: {1 / (data_time + inference_time)} img/s')
# print(f'Data loading time: {data_time}; per-image inference time: {inference_time}; per-image certification time: {certification_time}')
if args.flops:
    print(f'Average per-image FLOP count: {flops_total / total}')

if args.memory:
    print(f'Memory allocated (MB): allocated {np.mean(memory_allocated[:-1])}')
# print(np.mean(memory_allocated[:-1]))
# print(np.mean(memory_allocated[1:-1]))
# print(np.mean(memory_reserved[:-1]))
# print(np.mean(memory_reserved[1:-1]))
# print(f'Experiment run time : {(time.time()-start_time)/60},{(time.time()-start_time)/3600}')