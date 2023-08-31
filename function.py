
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
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim
import models.sam.utils.transforms as samtrans

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)


import torch


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # BCELoss和sigmoid融合  BECloss对输出向量的每个元素单独使用交叉熵损失函数，然后计算平均值
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print('total:{}'.format(total_num))
    print('trainable:{}'.format(trainable_num))
    # return {'Total': total_num, 'Trainable': trainable_num}

def get_ptsimgs(img):
    from dataset import build_all_layer_point_grids
    point_grids = build_all_layer_point_grids(
        n_per_side=32,
        n_layers=0,
        scale_per_layer=1,
    )
    points_scale = np.array(img.shape[1:])[None, ::-1]
    points_for_image = point_grids[0] * points_scale  # 1024 * 2
    in_points = torch.as_tensor(points_for_image)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
    # points = (in_points, in_labels)
    pt = points_for_image.tolist()
    point_label = in_labels.tolist()
    return pt,point_label


def postprocess_masks(
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_img_mask(image,masks_np,name):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks_np, plt.gca())
    plt.axis('off')
    plt.savefig(name)
    plt.show()

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')  # 也可以试试这个 针对seg任务
    else:
        # lossfunc = criterion_G
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')  # 也可以试试这个 针对seg任务

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:  # 调用ISIC2016.__getitem__
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            # del pack['pt']
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
                # pt, point_labels = get_ptsimgs(imgs)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != None:   # != -1
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if args.prompt_approach == 'random_click':
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                elif args.prompt_approach == 'points_grids':
                    pass
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            sam_no_freeze_block = [f"block.{idx}." for idx,i in enumerate(args.sam_block_freeze) if i == 1 ]

            
            '''Train'''
            for n, value in net.image_encoder.named_parameters():  # named_parameters() 方法可以对一个nn.Module中所有注册的参数进行迭代
                # print("named parameter image_encoder",n)
                # print("-----------------------------------------------")
                if "Adapter" not in n:  # 当前不是adapter的时候
                    if 1 in [1 for val in sam_no_freeze_block if val not in n]: 
                        value.requires_grad = False  # 当前量不需要在计算中保留对应的梯度信息（只训练image encoder的adapter ？）

            # for p in net.image_encoder.parameters():
            #     p.requires_grad = True

            imge = net.image_encoder(imgs)  # image embeddings

            # with torch.no_grad():
            #     # imge= net.image_encoder(imgs)
            #     se, de = net.prompt_encoder( # sparse embeddings for the points and boxes; dense embeddings for the masks
            #         points=pt,
            #         boxes=None,
            #         masks=None,
            #     )

            # use requires_grad = False to freeze prompt encoder rather than torch.no_grad()
            # Because the latter method will make the gradient not return,
            # the gradient of image encoder will not be updated.
            for p in net.prompt_encoder.parameters():
                p.requires_grad = False

            se, de = net.prompt_encoder(
                    points=pt,  # 用随机点作为prompt
                    boxes=None,
                    masks=None,
            )
            pred, _ = net.mask_decoder( # batched predicted masks
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),  # get_dense_pe() get positional encoding used to encode point prompts
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
              )

            # get_parameter_number(net)
            # get_parameter_number(net.image_encoder)
            # get_parameter_number(net.mask_decoder)

            loss = lossfunc(pred, masks)  # pred -> mask  masks -> label

            pbar.set_postfix(**{'loss (batch)': loss.item()}) # 显示指标
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step() # 更新权重
            optimizer.zero_grad() # 清除梯度

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'

                    if args.prompt_approach == 'random_click':
                        vis_image(imgs,pred,masks, os.path.join(args.path_helper['train_sample_path'],
                                                                namecat+'epoch+' + str(epoch) + '.jpg'), reverse=False, points=showp)
                    elif args.prompt_approach == 'points_grids':
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['train_sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'), reverse=False, points=None)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        # lossfunc = criterion_G
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:  # 改   1：当前情况的结果   2：用默认的情况看效果
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != None:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if args.prompt_approach == 'random_click':
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                    elif args.prompt_approach == 'points_grids':
                        pass
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)

                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                
                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Valid'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'

                        if args.prompt_approach == 'random_click':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=showp)
                        elif args.prompt_approach == 'points_grids':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=None)

                    mask_old = masks
                    # mask_threshold = 0.5
                    # return_logits = False
                    #
                    # masks = postprocess_masks(pred, (1024, 1024), (1024, 1024))
                    # if not return_logits:
                    #     masks = masks > mask_threshold
                    #
                    # masks_np = masks[0].detach().cpu().numpy()
                    # # true_point = random_click(masks_np[0], inout=True)
                    #
                    # image = torch.squeeze(imgs, dim=0).permute(1, 2, 0)
                    # image = image.detach().cpu().numpy()
                    # show_img_mask(image, masks_np)

                    temp = eval_seg(pred, mask_old, threshold)
                    print(temp)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def Test_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0, 0, 0, 0)
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    metrics = []

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        # lossfunc = criterion_G
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:  # 改   1：当前情况的结果   2：用默认的情况看效果
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:, :, buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size, c, w, h = imgs.size()
                longsize = w if w >= h else h

                if point_labels[0] != None:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if args.prompt_approach == 'random_click':
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                    elif args.prompt_approach == 'points_grids':
                        pass
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype=mask_type, device=GPUdevice)

                '''test'''
                with torch.no_grad():
                    imge = net.image_encoder(imgs)

                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )

                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'

                        if args.prompt_approach == 'random_click':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['test_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=showp)
                        elif args.prompt_approach == 'points_grids':
                            print(namecat + 'epoch+' + str(epoch) + '.jpg')
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['test_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=None)

                    mask_old = masks
                    mask_threshold = 0.5
                    return_logits = False

                    masks = postprocess_masks(pred, (1024, 1024), (1024, 1024))
                    if not return_logits:
                        masks = masks > mask_threshold

                    masks_np = masks[0].detach().cpu().numpy()
                    # true_point = random_click(masks_np[0], inout=True)

                    image = torch.squeeze(imgs, dim=0).permute(1, 2, 0)
                    image = image.detach().cpu().numpy()
                    show_img_mask(image, masks_np,os.path.join(args.path_helper['test_sample_path'],
                                                                      namecat + '_mask_gt' + '.jpg'))

                    temp = eval_seg(pred, mask_old, threshold)
                    
                    print(temp)
                    
                    metrics.append(temp)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    print(metrics)

    return tot / n_val, tuple([a / n_val for a in mix_res])