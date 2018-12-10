import tensorflow as tf
from tensorpack import *
import cv2
import glob
import os
from tensorpack.utils import logger
import numpy as np


class CITYSCAPES(DataFlow):
    
    def __init__(self, name, sub_rate, shuffle):
        self.name = name
        self.sub_rate = sub_rate
        self.shuffle = shuffle
        self._setup()
        
    def _setup(self):
        dataset_dir = '/home/smartcar/dataset/cityscapes/cityscapes_data'    
        assert self.name in ['train_fine', 'validation_fine', 'test_fine', 'combine_fine', 
                             'train_coarse','train_patches', 'validation_patches']
        if self.name == 'train_fine':
                image_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                            'leftImg8bit/train/*', '*_leftImg8bit.png')))
                gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                         'gtFine/train/*', '*_gtFine_labelTrainIds.png')))            
        elif self.name == 'validation_fine':
            image_files = sorted(glob.glob(os.path.join(dataset_dir, 
                                                        'leftImg8bit/val/*','*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtFine/val/*', '*_gtFine_labelTrainIds.png')))
        elif self.name == 'test_fine':
            image_files = sorted(glob.glob(os.path.join(dataset_dir, 
                                                        'leftImg8bit/test/*','*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtFine/test/*', '*_gtFine_labelTrainIds.png')))
        elif self.name == 'combine_fine':
            image_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                        'leftImg8bit/train/*', '*_leftImg8bit.png'))+
                                glob.glob(os.path.join(dataset_dir, 
                                                        'leftImg8bit/val/*','*_leftImg8bit.png'))) 

            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtFine/train/*', '*_gtFine_labelTrainIds.png'))+
                            glob.glob(os.path.join(dataset_dir,
                                                     'gtFine/val/*', '*_gtFine_labelTrainIds.png')))
        elif self.name == 'train_coarse':       
            image_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                        'leftImg8bit/train_extra/*', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtCoarse/train_extra/*', '*_gtCoarse_labelTrainIds.png')))
        elif self.name == 'train_patches':
            image_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                        'leftImg8bit_patch880/train', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine_patch880/train', '*_gtFine_labelTrainIds.png')))
        elif self.name == 'validation_patches':
            image_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                        'leftImg8bit_patch880/val', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtFine_patch880/val', '*_gtFine_labelTrainIds.png')))

        assert len(image_files) == len(gt_files), len(image_files)

        self.lst = zip(image_files, gt_files)
    
    def __iter__(self):
        
        idx = np.arange(len(self.lst))
        # If training, we should shuffle this list
        if self.shuffle is True:
            np.random.shuffle(idx)
            
        for i in idx:
            img_addr, gt_addr = self.lst[i]
            try:
                img = cv2.cvtColor(cv2.imread(img_addr), cv2.COLOR_BGR2RGB).astype(np.float32) 
            except:
                print(img_addr)
            #img = cv2.imread(img_addr, cv2.IMREAD_COLOR)
            #b,g,r = cv2.split(img)
            #img = cv2.merge([r,g,b])
            gt = cv2.imread(gt_addr, cv2.IMREAD_GRAYSCALE) # dtype uint8

            #if self.sub_rate == 1:
            #    yield [img, gt]
            #else:
            H, W, _ = img.shape 
            sub_rows = int(H/self.sub_rate)
            sub_cols = int(W/self.sub_rate)

            img = cv2.resize(src=img, dsize=(sub_cols, sub_rows) ,interpolation=cv2.INTER_NEAREST)

            # gt remains unresized
            #gt = cv2.resize(src=gt, dsize=(sub_cols, sub_rows) ,interpolation=INTER_NEAREST)
            yield [img, gt]
    
    def __len__(self):
        return len(self.lst)
    
