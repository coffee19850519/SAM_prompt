#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Medical-SAM-Adapter-main 
@File    ：precess_log.py
@Author  ：yang
@Date    ：2023/6/6 10:42 
'''
from matplotlib import pyplot as plt

log_file = r'log_1_dice.txt'
loss_list = []

def draw_loss(Loss_list,epoch,str):
    plt.cla()
    x1 = range(1, epoch+1)
    print(x1)
    y1 = Loss_list
    print(y1)
    plt.title(str+' loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel(str+'loss', fontsize=20)
    plt.grid()
    plt.savefig(str+".png")
    plt.show()

with open(log_file,'r') as f:
    file_contents = f.readlines()
    for line in file_contents:
        if '- Train loss: ' in line:
            info = line.split("||")
            loss = float(info[0][info[0].find('- Train loss: ')+len('- Train loss: '):])
            epoch = int(info[1].split('epoch ')[1].split('.\n')[0])+1
            loss_list.append(loss)
            print(epoch,loss)
        if 'Total score: ' in line:
            info = line.split("||")
            if info[0].find(" - ") != -1:
                info[0] = info[0][info[0].find(" - ")+len(" - "):]
            total_score = float(info[0].split(",")[0].replace("Total score: ", ''))
            iou = float(info[0].split(",")[1].replace(" IOU: ", ''))
            dice = float(info[0].split(",")[2].replace(" DICE: ", ''))
            epoch = int(info[1].split('epoch ')[1].split('.\n')[0])+1
            print(epoch,total_score,iou,dice)

draw_loss(loss_list, epoch, "train_2")
