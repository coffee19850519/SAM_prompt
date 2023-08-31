#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Medical-SAM-Adapter_cryoppp 
@File    ：process_data.py
@Author  ：yang
@Date    ：2023/6/20 9:16 
'''
import os
import numpy as np
import copy
import shutil


def mkdir_(remain_num_list, save_training_data_new_path):
    for remain_num in remain_num_list:
        savePath = f'{save_training_data_new_path}CryoPPP_data_{remain_num}/'
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        savePath = f'{save_training_data_new_path}CryoPPP_data_{remain_num}/train/'
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        savePath_image = f'{save_training_data_new_path}CryoPPP_data_{remain_num}/train/images/'
        if not os.path.exists(savePath_image):
            os.mkdir(savePath_image)

        savePath_lable = f'{save_training_data_new_path}CryoPPP_data_{remain_num}/train/labels/'
        if not os.path.exists(savePath_lable):
            os.mkdir(savePath_lable)


def getRemianName(need_del_numIndex: list, file_nameList: list):
    counterList = []

    for index in need_del_numIndex:
        counterList.append(file_nameList[index])
    sorted(counterList)
    return list(filter(lambda name: name not in counterList, file_nameList))


def getDelNumIndex(end, num):
    return list(np.random.choice(end, num, replace=False))


def save(remain_name, save_training_data_new_path, ori_data_path):
    for name in remain_name:
        savePath_image = f'{save_training_data_new_path}CryoPPP_data_{remain_num}/train/images/'
        savePath_label = f'{save_training_data_new_path}CryoPPP_data_{remain_num}/train/labels/'

        shutil.copyfile(os.path.join(f'{ori_data_path}/images/', name), os.path.join(savePath_image, name))
        shutil.copyfile(os.path.join(f'{ori_data_path}/labels/', name), os.path.join(savePath_label, name))


if __name__ == '__main__':
    save_training_data_new_path = '/mnt/Data1/yzy/data/SAM_profile/Cyro_ppp/'
    training_data_path = '/mnt/Data1/gmy/segment-anything-main/CryoPPP_data_472/train/'
    image_data_path = os.path.join(training_data_path, 'images')
    label_data_path = os.path.join(training_data_path, 'labels')

    isProportion = False

    all_data_name = sorted(os.listdir(image_data_path))

    file_name_10028 = list(filter(lambda x: '10028' in x, all_data_name))
    file_name_Foil = list(filter(lambda x: 'Foil' in x, all_data_name))

    remain_num_list = [300, 200, 100, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    # remain_num_list = [50]
    mkdir_(remain_num_list, save_training_data_new_path)

    for remain_num in remain_num_list:
        file_name_10028_copy = copy.deepcopy(file_name_10028)
        file_name_Foil_copy = copy.deepcopy(file_name_Foil)

        need_del_num = len(all_data_name) - remain_num

        if isProportion:
            need_del_num_for_10028 = int(need_del_num * 0.5)
            need_del_num_for_Foil = int(need_del_num * 0.5)
        else:
            need_del_num_for_10028 = int(need_del_num * len(file_name_10028_copy) / len(all_data_name))
            need_del_num_for_Foil = int(need_del_num * len(file_name_Foil_copy) / len(all_data_name))
            if need_del_num_for_10028 + need_del_num_for_Foil + remain_num != len(all_data_name):
                need_del_num_for_10028 += (
                            len(all_data_name) - need_del_num_for_10028 - need_del_num_for_Foil - remain_num)

        need_del_numIndex_for_10028 = getDelNumIndex(len(file_name_10028), need_del_num_for_10028)
        need_del_numIndex_for_Foil = getDelNumIndex(len(file_name_Foil), need_del_num_for_Foil)

        remain_name_for_10028 = getRemianName(need_del_numIndex_for_10028, file_name_10028_copy)
        remain_name_for_Foil = getRemianName(need_del_numIndex_for_Foil, file_name_Foil_copy)

        # save
        save(remain_name_for_10028, save_training_data_new_path, training_data_path)
        save(remain_name_for_Foil, save_training_data_new_path, training_data_path)

        # print(list(filter(lambda x: x[0].lower() in 'aeiou', creature_names)))
