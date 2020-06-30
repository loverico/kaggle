import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd
import numpy as np
import gc
import os
import cv2

import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet
%matplotlib inline
class AdvancedHairAugmentation:
    def __init__(self, hairs: int = 4, hairs_folder: str = "/home/jupyter/kaggle/melanoma_repo/data/01_raw/hair_imgs"):
        self.hairs = hairs
        self.hairs_folder = hairs_folder
        self.hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
    def __call__(self, img):
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        img.flags.writeable = True
#         print(img.flags.writeable)
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(self.hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img
from PIL import Image,ImageDraw,ImageFilter
import os
CROP_WIDTH =CROP_HEIGHT = 512
BACKGROUND_COLOR = (0,0,0)
BACKGROUND = Image.new("RGB", (CROP_WIDTH,CROP_HEIGHT), (0,0,0))
BILINEAR = Image.BILINEAR

def resize(img,size=(CROP_WIDTH,CROP_HEIGHT),resize_filter=Image.NEAREST):
    return img.resize(size,resize_filter)

def crop_center(pil_img, crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def circuler_crop(img,crop_widht=CROP_WIDTH,crop_height=CROP_HEIGHT,background=BACKGROUND, blur_radius=1,):
#     img = Image.open(img_path)
    img_width,img_height = img.size
    offset = blur_radius * 2
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, img.size[0] - offset, img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return Image.composite(img, background, mask)
#     return img.crop(((img_width - crop_width) // 2,
#                          (img_height - crop_height) // 2,
#                          (img_width + crop_width) // 2,
#                          (img_height + crop_height) // 2))
def convert_crop(img_path,output_dir="/home/jupyter/kaggle/melanoma_repo/data/01_raw/cropped/"):
    
    img = Image.open(img_path)
    img = resize(img)
    img_np=np.asarray(img).copy()
#     print(img_np.flags)
#     img_np.flags.writeable = True
    img_np = draw_hair.__call__(img_np)
    
    img = crop_center(Image.fromarray(img_np))
    img = circuler_crop(img)
    img_name =os.path.basename(img_path)
    img.save(output_dir + img_name)
# def bulk_convert_crop(imgs):
%%time
import glob
from multiprocessing import Pool
import multiprocessing as multi


train_img_path = "/home/jupyter/kaggle/melanoma_repo/data/01_raw/jpeg/train"
train_imgs=glob.glob(train_img_path+"/*")
p= Pool(multi.cpu_count())
p.map(convert_crop,train_imgs)
p.close()
from PIL import Image, ImageOps,ImageEnhance
import random
CHOICE_BRIGHTEN = [2,3,5,6,7]
CHOICE_SHARPEN = [2,3,5,6,7]

def flipen(img):
    return ImageOps.flip(img)
def mirroren(img):
    return ImageOps.mirror(img)

def brighten(img):
    con9=ImageEnhance.Brightness(img)
    
    return con9.enhance(random.choice(CHOICE_BRIGHTEN)/4.0)
#     return con9.enhance(7/4.0)


def sharpen(img):
    con11=ImageEnhance.Sharpness(img)
    return con11.enhance(random.choice(CHOICE_SHARPEN)/4.0)


augs = [flipen, brighten,mirroren]
