import numpy as np
import time
import random
import torch
import glob
import os
import imageio 
import skimage


def dummy_collate_fn(data_list):
    return data_list[0]


def simple_collate_fn(data_list):
    ks = data_list[0].keys()
    outputs = {k: [] for k in ks}
    for k in ks:
        for data in data_list:
            outputs[k].append(data[k])
        outputs[k] = torch.stack(outputs[k], 0)
    return outputs


def set_seed(index, is_train):
    if is_train:
        np.random.seed((index + int(time.time())) % (2 ** 16))
        random.seed((index + int(time.time())) % (2 ** 16) + 1)
        torch.random.manual_seed((index + int(time.time())) % (2 ** 16) + 1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path, ext))) # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    return imgs

def load_rgb(path, normalize_rgb = False):  # load rgb image
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # if normalize_rgb: # [-1,1] --> [0,1]?这里应该是[0,1]-->[-1,1]吧
    #    img -= 0.5
    #    img *= 2.
    #img = img.transpose(2, 0, 1) # 改变图像数据的维度顺序，将原来的(H,W,C)变为(C,H,W)
    return img