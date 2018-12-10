import tensorflow as tf
import os
import glob
import numpy as np

class Dataset:

    def __init__(self, dataset_name, batch_size, dataset_dir, sub_rate, mode, is_shuffle=False, patches_crop=False, augment=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.sub_rate = sub_rate # subsample rate for input images, NOT gts
        self.mode = mode # train, val, or test or combine, or patches
        self.patches_crop = patches_crop
        self.crop_size = 584
        self.augment = augment
        self.is_shuffle = is_shuffle
        self._load_dataset()
    
    def _load_dataset(self):
        '''
        choose which dataset to load
        '''
        if self.dataset_name == 'cityscapes':
            self._cityscapes()
        elif slef.dataset_name == 'kitti':
            pass
                
    
    def _cityscapes(self):
        '''
        Dataset basic info:
        params self.size
        params self.num_classes
        params self.ignore_label
        '''
        
        if self.mode == 'train_fine':
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                        'leftImg8bit/train/*', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine/train/*', '*_gtFine_labelTrainIds.png')))
            
        elif self.mode == 'validation_fine':
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir, 
                                                        'leftImg8bit/val/*','*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine/val/*', '*_gtFine_labelIds.png')))
        elif self.mode == 'test_fine':
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir, 
                                                        'leftImg8bit/test/*','*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine/test/*', '*_gtFine_labelIds.png')))
        elif self.mode == 'combine_fine':
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                        'leftImg8bit/train/*', '*_leftImg8bit.png'))+
                                glob.glob(os.path.join(self.dataset_dir, 
                                                        'leftImg8bit/val/*','*_leftImg8bit.png'))) 
                            
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine/train/*', '*_gtFine_labelTrainIds.png'))+
                            glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine/val/*', '*_gtFine_labelTrainIds.png')))
        elif self.mode == 'train_coarse':
            # TODO: download coarse images
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                        'leftImg8bit/train_extra/*', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtCoarse/train_extra/*', '*_gtCoarse_labelTrainIds.png')))
        elif self.mode == 'train_patches':
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                        'leftImg8bit_patch880/train', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine_patch880/train', '*_gtFine_labelTrainIds.png')))
        elif self.mode == 'validation_patches':
            image_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                        'leftImg8bit_patch880/val', '*_leftImg8bit.png')))
            gt_files = sorted(glob.glob(os.path.join(self.dataset_dir,
                                                     'gtFine_patch880/val', '*_gtFine_labelTrainIds.png')))
            
        #print len(image_files)
        assert len(image_files) == len(gt_files)

        # Cityscapes dataset info
        self.size = len(image_files)
        self.num_classes = 19 # There are 19 classes for training
        self.ignore_label = 19 # We create labelId 19 as ignore label
        #if self.mode == 'patches' or self.mode=='patches_val':
        if 'patches' in self.mode:
            self.H = 880 # Patch size
            self.W = 880
        else:
            self.H = 1024 # Original image height
            self.W = 2048 # Original image width
            
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        
        images = tf.convert_to_tensor(image_files)
        gts = tf.convert_to_tensor(gt_files)
        #self.images_addr = images
        #self.gts_addr = gts
        
        return images, gts

    def _random_crop_patches(self, images, gts):
        
        pool = tf.convert_to_tensor([(0,0), (self.H-self.crop_size-1,0), (0,self.H-self.crop_size-1), 
                                     (self.H-self.crop_size-1, self.H-self.crop_size-1)])
        top_left_index = tf.multinomial(logits=tf.log([[1., 1., 1., 1.]]), num_samples=1,seed=0) 
        #top_left_index = np.random.choice([0, 1, 2, 3])
        top_left = pool[top_left_index[0][0]]
        
        offset_height, offset_width = top_left[0], top_left[1]
        target_height, target_width = self.crop_size, self.crop_size
        
        images = tf.image.crop_to_bounding_box(images, offset_height, offset_width, 
                                              target_height, target_width)
        gts = tf.image.crop_to_bounding_box(gts, offset_height, offset_width, 
                                           target_height, target_width)
        return images, gts
        
    
    def _read_and_process_img(self, images_addr, gts_addr):

        input_queue = tf.train.slice_input_producer([images_addr, gts_addr], shuffle=self.is_shuffle)
        # decode 
        image = tf.image.decode_png(tf.read_file(input_queue[0]), channels=3)
        gt = tf.image.decode_png(tf.read_file(input_queue[1]), channels=1)
        # process images
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # substract img mean 
        # image -= self.IMG_MEAN
        
        ##### Subsample ######
        '''
        We downsample input images according to sub_rate, however, gts will always remain unchange, 
        it will be processed in trainer's loss function.
        '''
        if self.sub_rate != 1:
            sub_shape = [self.H/self.sub_rate, self.W/self.sub_rate]
            image = tf.image.resize_images(image, sub_shape, method=1) 
        
        if self.augment is True:
            # Perform data augumentation            
            image, gt = self._data_aug(image, gt)
        
        if self.patches_crop is True:
            image, gt = self._random_crop_patches(image, gt)
        
        ##### SET SHAPE #####
        if self.patches_crop is True:
            image.set_shape(shape=(self.crop_size, self.crop_size,3))
            gt.set_shape(shape=(self.crop_size, self.crop_size,1))
        else:
            image.set_shape(shape=(self.H/self.sub_rate, self.W/self.sub_rate,3))
            gt.set_shape(shape=(self.H, self.W, 1))
        
        return image, gt
    
    def next_batch(self):
        if self.dataset_name == 'cityscapes':            
            images_addr, gts_addr = self._cityscapes()            
            image, gt = self._read_and_process_img(images_addr, gts_addr)  

            # train batch
            images, gts = tf.train.batch([image, gt], batch_size=self.batch_size, allow_smaller_final_batch=True)
            
            return images, gts
        
    
    def _data_aug(self, img, gt):
        '''
        Random flip or process imgs and gts for data augumentation.
        
        '''
        # Random Flip
        #FLIP_LR = np.random.choice([0, 1]).astype(bool) # This is not correct, cuz it is fixed when finishing thr static graph
        
        # Ref: 1.https://www.tensorflow.org/api_docs/python/tf/multinomial
        #      2.https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
        pool = tf.convert_to_tensor([True, False])
        # TODO: seed save to file
        index = tf.multinomial(logits=tf.log([[1., 1.]]), num_samples=1,seed=0) 
        FLIP_LR =pool[index[0][0]]
        #FLIP_UD = np.random.choice([0, 1]).astype(bool)
        tf.logging.info('Perform data augmentation.')
        if FLIP_LR is True:
            tf.logging.info('Perform left-right flip')
            img, gt = tf.image.flip_left_right(img), tf.image.flip_left_right(gt)
        #if FLIP_UD:
        #    tf.logging.info('Perform up-down flip')
        #    img, gt = tf.image.flip_up_down(img), tf.image.flip_up_down(gt)
        
        # Radom 
        # Ref: https://blog.csdn.net/sinat_21585785/article/details/74180217
        #img = tf.image.random_brightness(img, max_delta=63)
        #img = tf.image.random_contrast(img, lower=0.2,upper=1.8)
        
        return img, gt
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    #from dataset import Dataset
    dataset_dir = '/home/smartcar/dataset/cityscapes/cityscapes_data'
    coord = tf.train.Coordinator()
    test = Dataset(dataset_name='cityscapes', batch_size=4, sub_rate=1, dataset_dir=dataset_dir, mode='validation_patches', patches_crop=False, is_shuffle=True) 
    images, gts= test.next_batch()
    
    with tf.Session() as sess:               
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)        
        images = sess.run(images)
        print(images.shape)


