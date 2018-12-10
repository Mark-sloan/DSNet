import tensorflow as tf
from tensorpack import *
import cv2
import glob
import os
from tensorpack.utils import logger
import numpy as np


class CITYSCAPES(RNGDataFlow):
    
    def __init__(self, name, sub_rate, shuffle, random_crop):
        self.name = name
        self.sub_rate = sub_rate
        self.shuffle = shuffle
        self.rnd_crop = random_crop
        self._setup()
        
    def _setup(self):
        dataset_dir = '/home/smartcar/dataset/cityscapes/cityscapes_data'    
        assert self.name in ['train_fine', 'validation_fine', 'test_fine', 'combine_fine', 
                             'train_coarse','train_patches', 'validation_patches', 
                             'combine_patches', 'combine_val_patches']
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
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtFine_patch880/train', '*_gtFine_labelTrainIds.png')))
        elif self.name == 'validation_patches':
            image_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                        'leftImg8bit_patch880/val', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,
                                                     'gtFine_patch880/val', '*_gtFine_labelTrainIds.png')))
        elif self.name == 'combine_patches':
            train_files = sorted(glob.glob(os.path.join(dataset_dir, 'leftImg8bit_patch880/train', '*_leftImg8bit.png')))
            val_files = sorted(glob.glob(os.path.join(dataset_dir,'leftImg8bit_patch880/val', '*_leftImg8bit.png')))[:3000]
            image_files = train_files + val_files
            
            train_gt_files = sorted(glob.glob(os.path.join(dataset_dir, 'gtFine_patch880/train', '*_gtFine_labelTrainIds.png')))
            val_gt_files = sorted(glob.glob(os.path.join(dataset_dir, 'gtFine_patch880/val', '*_gtFine_labelTrainIds.png')))[:3000]
            gt_files = train_gt_files + val_gt_files
        elif self.name == 'combine_val_patches':
            image_files = sorted(glob.glob(os.path.join(dataset_dir,'leftImg8bit_patch880/val', '*_leftImg8bit.png')))[3000:]
            gt_files = sorted(glob.glob(os.path.join(dataset_dir,'gtFine_patch880/val', '*_gtFine_labelTrainIds.png')))[3000:]

        assert len(image_files) == len(gt_files)

        self.lst = zip(image_files, gt_files)
        
    
    def __iter__(self):

        idx = np.arange(len(self.lst))
        #FLIP = False
        # If training, we should shuffle this list
        if self.shuffle is True:
            # RNGDataFlow has reset_state, otherwise each fork process will have the same random seed
            # see: https://tensorpack.readthedocs.io/tutorial/extend/dataflow.html?highlight=rng
            self.rng.shuffle(idx)
            #np.random.shuffle(idx)
                       
        for i in idx:
            img_addr, gt_addr = self.lst[i]
            try:
                img = cv2.cvtColor(cv2.imread(img_addr), cv2.COLOR_BGR2RGB)#.astype(np.float32) 
            except:
                print('Problematic image', img_addr)
            try:
                gt = cv2.imread(gt_addr, cv2.IMREAD_GRAYSCALE) # dtype uint8
            except:
                print('Problematic gt', gt_addr)
            

            H, W, _ = img.shape 
            sub_rows = int(H/self.sub_rate)
            sub_cols = int(W/self.sub_rate)
            #logger.info('sub_rows:{}, sub_cols:{}'.format(sub_rows, sub_cols))
            # this is a bug!!
            #img = cv2.resize(src=img, dsize=(sub_cols, sub_rows) ,interpolation=cv2.INTER_LINEAR)           
            img = cv2.resize(src=img, dsize=(sub_cols, sub_rows), interpolation=cv2.INTER_NEAREST)
            
            if self.rnd_crop is True:
                #logger.info('Perform random cropping.')
                offset = self.rng.choice([10,20,30,40,50,60,70])
                img = img[offset:offset+800, offset:offset+800]
                gt = gt[offset:offset+800, offset:offset+800]          
            
            if self.shuffle is True:
                FLIP = self.rng.rand() < 0.5
            else:
                FLIP = False
                
            if FLIP is True:
                img = cv2.flip(img, 1)
                gt = cv2.flip(gt, 1)
            
            yield [img, gt]
    
    def __len__(self):
        return len(self.lst)
    
