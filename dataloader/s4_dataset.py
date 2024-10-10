import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
import dataloader.transforms as T


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""

    def __init__(self, split='train', img_size=512, cfg=None):
        super(S4Dataset, self).__init__()
        self.split = split
        self.cfg = cfg
        self.mask_num = 1 if self.split == 'train' else 5
        anno_csv = os.path.join(cfg.data_dir, 's4_meta_data.csv')
        df_all = pd.read_csv(anno_csv, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform = transforms.Compose([         # 进一步提高特征图分辨率为640 x 640
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
        ])

        scales = [288, 320, 352, 392, 416, 448, 480, 512]
        max_size = 640

        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(224, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
        ])

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        
        img_base_path = os.path.join(self.cfg.data_dir, 's4_data/visual_frames', self.split, category, video_name)
        audio_lm_path = os.path.join(self.cfg.data_dir, 's4_data/audio_log_mel', self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(self.cfg.data_dir, 's4_data/gt_masks', self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)

        imgs, masks = [], []
        for img_id in range(1, 6):
            # for data in the training set, using the data augmentation and then resize and normalize the image. 
            if self.split == 'train': 
                img_path = os.path.join(img_base_path, "%s_%d.png" % (video_name, img_id))
                img_PIL = Image.open(img_path).convert('RGB')
                mask_path = os.path.join(mask_base_path, "%s_%d.png" % (video_name, img_id))
                if os.path.exists(mask_path):        # if the mask exists
                    mask_PIL = Image.open(mask_path).convert('P')
                    mask_PIL = transforms.ToTensor()(mask_PIL)
                    img, mask = self.train_transforms(img_PIL, mask_PIL)     # only the first frame has the mask in the training set
                    mask = self.mask_transform(mask)
                    masks.append(mask)
                else:
                    img, _ = self.train_transforms(img_PIL, None)
                img = self.img_transform(img)
                imgs.append(img)
               
            # for data in the validation/testing set, we just simply resize and normalize them. 
            else:         
                img_path = os.path.join(img_base_path, "%s_%d.png" % (video_name, img_id))
                img_PIL = Image.open(img_path).convert('RGB')
                mask_path = os.path.join(mask_base_path, "%s_%d.png" % (video_name, img_id))
                mask_PIL = Image.open(mask_path).convert('P')
                mask_PIL = transforms.ToTensor()(mask_PIL)
                imgs.append(self.img_transform(img_PIL))         # all the images in the validation set have the masks. 
                masks.append(self.mask_transform(mask_PIL))

        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        if self.split == 'train':
            return imgs_tensor, audio_log_mel, masks_tensor
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, category, video_name

    def __len__(self):
        return len(self.df_split)
