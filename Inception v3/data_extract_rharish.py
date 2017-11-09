#!/bin/python2
import numpy as np
from scipy.misc import imread, imresize
import random
import os

imagenet_folder = '/home/rharish/Data/ILSVRC2012/raw-data'
img_dict = {}
folder_count = 0
for folder in os.listdir(imagenet_folder):
    if folder == 'val':
        continue
    elif os.path.isdir(imagenet_folder + '/' + folder):
        for img_name in os.listdir(imagenet_folder + '/' + folder):
            if img_name[-5:] == '.JPEG':
                img_dict[img_name] = (folder, folder_count)
        folder_count += 1

def one_hot(label):
    return np.identity(1000)[label]

def gen_data(batch_size=32):
    img_list = img_dict.keys()
    random.shuffle(img_list)
    while True:
        for i in range(0, len(img_list), batch_size):
            img_batch = []
            labels_batch = []
            for img_name in img_list[i:(i+batch_size)]:
                img_batch.append(imresize(imread(imagenet_folder + '/' +
                                                 img_dict[img_name][0] + '/' +
                                                 img_name), (299, 299, 3)))
                labels_batch.append(one_hot(img_dict[img_name][1]))
            yield (np.array(img_batch), np.array(labels_batch))

