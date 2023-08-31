# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset_1 import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 
import humanize,psutil,GPUtil

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)


def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)   

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.3582489, 0.3582489, 0.3582489], std=[0.03951929, 0.03951929, 0.03951929])
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.3582489], std=[0.03951929])
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.3582489, 0.3582489, 0.3582489], std=[0.03951929, 0.03951929, 0.03951929])
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.3582489], std=[0.03951929])
])


if args.dataset == 'CryoPPP':
    '''Cryoppp data'''
    label_list, name_list = get_imagse_and_labels_path(args.data_path,mode='train')

    train_dataset = CryopppDataset(args, args.data_path,name_list[:5],label_list[:5], transform=transform_train,
                                   transform_msk=transform_train_seg, mode='train', prompt=args.prompt_approach)
    
    valid_dataset = CryopppDataset(args, args.data_path,name_list[5:6],label_list[5:6], transform = transform_test,
                                   transform_msk=transform_test_seg, mode ='valid', prompt=args.prompt_approach)

    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_valid_loader = DataLoader(valid_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''

elif args.dataset == 'decathlon':
    nice_train_loader, nice_valid_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)


'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
# TensorBoard 是一组用于数据可视化的工具。它包含在流行的开源机器学习库 Tensorflow 中。TensorBoard 的主要功能包括：
# 可视化模型的网络架构;跟踪模型指标，如损失和准确性等;检查机器学习工作流程中权重、偏差和其他组件的直方图;显示非表格数据，包括图像、文本和音频;将高维嵌入投影到低维空间
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begin training'''
best_acc = 0.0
best_tol = 1e6
best_iou = 1e10
best_dice = 1e10
for epoch in range(settings.EPOCH):
    if args.mod == 'sam_adpt':
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print("-------------------------------------------------------")
        print("Number of Trainable Parameters",count_parameters(net))
        print("-------------------------------------------------------")
        print('time_for_training ', time_end - time_start)

        net.eval()   # 验证的结果统计指标（IOU、...）
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:  # 什么时候eval
            tol, (eiou, edice) = function.validation_sam(args, nice_valid_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol or eiou > best_iou or edice > best_dice:
                best_iou = eiou
                best_dice = edice
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
                print("-------------sava best checkpoint------------------")
            else:
                is_best = False
mem_report()
writer.close()