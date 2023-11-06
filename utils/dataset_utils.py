import numpy as np
import time
import random
import torch
import glob
import os
import imageio 
import skimage
import json
# import pyexr

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

def glob_imgs(path): # 抓取所有的图片
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

def load_cams_from_sfmscene(path):

    # load json file
    with open(path) as f:
        sfm_scene = json.load(f)

    # camera parameters
    intrinsics = dict()
    extrinsics = dict()
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera 
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1
            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            intrinsics[index] = intrinsic
            extrinsics[index] = extrinsic

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)

    # compute scale_mat for coordinate normalization
    scale_mat = bbox_transform.copy()
    scale_mat[[0,1,2],[0,1,2]] = scale_mat[[0,1,2],[0,1,2]].max() / 2
    
    # meta info
    image_list = sfm_scene['image_path']['file_paths']
    image_indexes = [str(k) for k in sorted([int(k) for k in image_list])] # index of images
    resolution = camera_info_list[image_indexes[0]]['size'][::-1] # 逆序

    return intrinsics, extrinsics, scale_mat, image_list, image_indexes, resolution



def load_rgb_image(path):
    ''' 
    from NeILFPP
    Load RGB image (both uint8 and float32) into image in range [0, 1] '''
    # ext = os.path.splitext(path)[1] # 获取文件后缀
    # if ext == '.exr':
    #     # NOTE imageio read exr has artifact https://github.com/imageio/imageio/issues/517
    #     image = pyexr.read(path)
        
        
    # else:
    #     image = imageio.imread(path) # imageio读图像
    # if image.shape[-1] > 3: # 如果是rgba图像,取前三个通道
    #     image = image[..., :3]                          # [H, W, 4] -> [H, W ,3]
    # image = skimage.img_as_float32(image)
    # save_iamge = image.copy()
    # pyexr.write('tmp2.exr', save_iamge)
    # return image
    pass


def load_rgb_image_with_prefix(prefix):
    '''
    from NeILFPP
    Load image using prefix to support different data type '''
    # exts = ['.png', '.jpg', '.tiff', '.exr']
    # for ext in exts:
    #     path = prefix + ext
    #     if os.path.exists(path):
    #         return load_rgb_image(path)
    # print ('Does not exists any image file with prefix: ' + prefix)
    # return None
    pass