import sys
sys.path.append('..')
import tensorflow as tf
slim = tf.contrib.slim
from layers.layers_tensorpack import initial_block, shuffle_unit, shuffle_down, shuffle_up
from utils.enc import enc
from tensorpack import *
from tensorpack.tfutils import argscope
from utils.ops import loss_wi_ignore, focal_loss
import numpy as np
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorflow.contrib.layers.python.layers import initializers
from tensorpack.tfutils.scope_utils import under_name_scope, auto_reuse_variable_scope
initializer=initializers.xavier_initializer()

class ShuffleTensorpack(ModelDesc):
    def __init__(self, args, sub_rate, batchsize):
        self.args = args
        self.sub_rate = sub_rate
        self.batchsize = batchsize
        self.scope = 'ShuffleTensorpack'
        self.num_classes = 19 # For cityscapes, 19 trainable classes
        self.weight_decay = args.weight_decay
        self._set_shapes()
    
    def _set_shapes(self):
        if 'patches' in self.args.dataset_mode:
            if self.args.random_crop is True:               
                self.H = 800
                self.W = 800
            else:
                self.H = 880
                self.W = 880
        elif 'speed_test' in self.args.dataset_mode:
            self.H = 360
            self.W = 480
        else:
            self.H = 1024
            self.W = 2048
        
    
    def inputs(self):
        
        images = tf.placeholder(tf.float32, (None, int(self.H/self.sub_rate), 
                                             int(self.W/self.sub_rate), 3), 'images')
        gts = tf.placeholder(tf.uint8, (None, self.H, self.W), 'gts')
        
        return [images, gts]
    
    @auto_reuse_variable_scope
    def get_logits(self, images):

        # The two bachnorm params in tensorpack is different with slim, we stick to slim.
        with argscope(BatchNorm, scale=False, epsilon=0.001), \
            argscope([Conv2D, Conv2DTranspose], kernel_initializer=initializer):
            ###### Initial Block ######
            net = initial_block("initial_block", images)
            ###### STAGE ONE ##########
            stage = 1
            net, pooling_indices_1 = shuffle_down(name='Stage'+str(stage)+'_0_regu',
                                                  inputs=net, output_depth=64, 
                                                  num_groups=self.args.num_groups) 
            for i in range(4):
                net = shuffle_unit(name='Stage'+str(stage)+'_%d'%(i+1)+'_regu', 
                                   inputs=net, num_split=2, output_depth=64, 
                                   num_groups=self.args.num_groups)

            ###### STAGE TWO #######
            stage = stage+1
            net, pooling_indices_2 = shuffle_down(name='Stage'+str(stage)+'_0_regu', 
                                                  inputs=net, output_depth=128, 
                                                  num_groups=self.args.num_groups)                
            ## 1,2,5,9,1,2,5,9
            net = shuffle_unit(name='Stage'+str(stage)+'_1_regu', 
                                inputs=net, num_split=2, output_depth=128, 
                                num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_2_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_3_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_4_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=9, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_5_regu', 
                                inputs=net, num_split=2, output_depth=128, 
                                num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_6_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_7_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_8_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=9, num_groups=self.args.num_groups)
            self.net_stage_2 = net
            ######### Context Encoding Module ########
            #if self.args.add_se_loss is True:

            #    self.enc_2 = enc(inputs=net, K=32, name='Stage'+str(stage)+'encoding') # E: BxD(C)
            #    self.fc_2 = FullyConnected('encoding/fc2', self.enc_2, self.num_classes)
                #self.fc_2 = slim.fully_connected(inputs=self.enc_2, 
                #                                num_outputs=self.num_classes, 
                #                                activation_fn=None,
                #                                scope='encoding/fc2')
            ###### STAGE THREE #######
            stage = stage+1

            ## 2,5,9,17,2,5,9,17
            net = shuffle_unit(name='Stage'+str(stage)+'_1_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_2_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_3_dila', 
                               inputs=net, num_split=2, output_depth=128, 
                               dilation_rate=9, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_4_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=17, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_5_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_6_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_7_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=9, num_groups=self.args.num_groups)
            net = shuffle_unit(name='Stage'+str(stage)+'_8_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=17, num_groups=self.args.num_groups)
            self.net_stage_3 = net
            ######### Context Encoding Module ########              
            if self.args.add_se_loss is True:
                self.enc_3 = enc(inputs=net, K=32, name='Stage'+str(stage)+'encoding') # E: BxD(C)
                self.fc_3 = FullyConnected('encoding/fc3', self.enc_3, self.num_classes)
                #self.fc_3 = slim.fully_connected(inputs=self.enc_3, 
                #                                num_outputs=self.num_classes, 
                #                                activation_fn=None, 
                #                                scope='encoding/fc3') # B x num_classes

            #### DECODER ###############
            #### STAGE FOUR ############
            stage = stage+1
            net = shuffle_up(name='Stage'+str(stage)+'_0', 
                            inputs=net, output_depth=64, 
                            pooling_indices=pooling_indices_2)
            net = shuffle_unit(name='Stage'+str(stage)+'_1_regu', 
                               inputs=net, num_split=2, output_depth=64,
                               num_groups=self.args.num_groups)
            #net = shuffle_unit(name='Stage'+str(stage)+'_2_regu', 
            #                   inputs=net, num_split=2, output_depth=64,
            #                   num_groups=self.args.num_groups)

            #### STAGE FIVE ############
            stage=stage+1
            net = shuffle_up(name='Stage'+str(stage)+'_0', 
                            inputs=net, output_depth=16, 
                            pooling_indices=pooling_indices_1)
            net = shuffle_unit(name='Stage'+str(stage)+'_1_regu', 
                               inputs=net, num_split=2, output_depth=16,
                               num_groups=self.args.num_groups)
            #net = shuffle_unit(name='Stage'+str(stage)+'_2_regu', 
            #                   inputs=net, num_split=2, output_depth=16,
            #                   num_groups=self.args.num_groups)

            #### FINAL CONV ############

            logits = Conv2DTranspose('fullconv', net, filters=19, 
                          kernel_size=2, strides=(2,2), use_bias=False)       
        
        return logits

    
    def build_graph(self, images, gts):

        logits = self.get_logits(images)
        total_loss = self.compute_loss(logits, gts)

        return total_loss
    
    def optimizer(self):
        
        lr = tf.get_variable('learning_rate', 
                             initializer=self.args.initial_learning_rate, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        
        if self.args.optimizer == 'Adam':
            # by default we use Adam
            logger.info('Using ADAM optimizer')
            optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
        elif self.args.optimizer == 'SGD':
            logger.info('Using SGD optimizer')
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            
        return optimizer
    
    def compute_loss(self, logits, gts):
        ##### Set shape ################
        #_, H_logits, W_logits, _ = self.inputs.get_shape().as_list()
           
        if self.args.loss_type == 'small2big':
            #assert H_logits == self.H/self.args.sub_rate
            # resize 'small' logits to 'big' to match 'big' gts
            logits = tf.image.resize_images(logits, [self.H,self.W], method=0)        
        elif self.args.loss_type == 'small2small':
            #assert H_logits == self.H/self.args.sub_rate
            gts = tf.image.resize_images(gts, 
                                        [self.H/self.sub_rate,self.W/self.sub_rate], method=1)
        self.gts = gts
        
        #tf.summary.image(name='seg_ground_truth', tensor=self.gts, max_outputs=1)
        ### Compute loss ###############
        if self.args.loss_mode == 'focal':
            tf.logging.info('Using weighted focal loss.')
            #shape = logits.get_shape().as_list()
            #gts = tf.reshape(self.gts, shape=(shape[0], shape[1], shape[2]))
            valid_pixels = tf.reduce_sum(tf.cast(tf.less_equal(self.gts, 18), tf.float32))
            loss = tf.divide(focal_loss(labels=gts, logits=logits, num_classes=19, 
                                     ignore_label=19, scope='focal_loss', scale=1), valid_pixels)
            tf.summary.scalar('focal_loss', loss)
                            
        else:            
            logger.info('Using weighted cross entrophy loss.')
            loss = loss_wi_ignore(labels=gts, logits=logits, num_classes=19, 
                                 ignore_label=19)
            #loss = tf.reduce_mean(loss, name='weighted_cross_entrophy_loss')
            tf.summary.scalar('weighted_cross_entrophy_loss', loss)
            
        if self.args.add_se_loss is True:
            '''
            Semantic encoding loss
            The output of semantic encoding layer is used to classify
            num_classes
            '''
            logger.info('Adding context encoding loss.')
            # generate label vector
            vector = tf.get_variable(name='labels_vector', shape=(self.batchsize, 19), 
                                     initializer=tf.zeros_initializer(), dtype=tf.int32) # B x num_classes
            gts = tf.cast(gts, tf.int32)
            gts_shape = gts.get_shape().as_list()
            for i in range(self.batchsize):
                hist = tf.histogram_fixed_width(values=gts[i], value_range=[0, 19-1], 
                                                nbins=19, dtype=tf.int32)
                
                tf.assign(vector[i], hist)        
            
            vector = tf.cast(vector, tf.float32)/(gts_shape[1]*gts_shape[2])
            #vector = tf.cast((vector/self.H*self.W), tf.float32)
            
            #se2_logits = self.fc_2 # B x num_classes
            #se_loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=se2_logits, labels=vector)
            se3_logtis = self.fc_3 # B x num_classes
            se_loss_3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=se3_logtis, labels=vector)      
        
            se_weight = 1.0
            se_loss = tf.reduce_mean(se_weight * se_loss_3)         
            #tf.losses.add_loss(se_loss)
            tf.summary.scalar('se_loss', se_loss)
        
        ### ADD Weigt decay l2 loss. ###############
        if self.weight_decay > 0:
            weight_decay_pattern = '.*/W'
            wd_loss = regularize_cost(weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay), 
                                      name='l2_regularize_loss')
            #wd_loss = tf.reduce_mean(wd_loss, name='l2_regularize_loss')
            #tf.losses.add_loss(tf.reduce_mean(wd_loss))
            #tf.losses.add_loss(wd_loss)
            tf.summary.scalar('wd_l2_loss', wd_loss)
        
        # Get prediction
        predictions = self.get_prediction(logits)
        # Compute mIoU
        mIoU = self.compute_mIoU(gts, predictions)
        
        
        #total_loss = tf.add_n([loss, wd_loss])
        total_loss = loss + wd_loss
        if self.args.add_se_loss is True:
            #total_loss = tf.add_n([total_loss, se_loss])
            total_loss = total_loss + se_loss
        #total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', total_loss)
        add_moving_summary(total_loss)
        return tf.reduce_mean(total_loss, name='total_loss')    

    
    def get_prediction(self, logits):
        prob = tf.nn.softmax(logits)
        predictions = tf.argmax(prob, -1, 'prediction')
        # summary training segmentation result !
        predictions_shape = predictions.get_shape().as_list()      
        segmentation_output_train = tf.cast(predictions, dtype=tf.float32)
        segmentation_output_train = tf.reshape(segmentation_output_train, 
                                               shape=[-1, predictions_shape[1], predictions_shape[2], 1])

        tf.summary.image(name='seg_result', tensor=segmentation_output_train, max_outputs=1)   

        return predictions
    
    
    def compute_mIoU(self, gts, predictions):
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(gts, shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, 19))
        labels = tf.where(tf.equal(labels, 19), tf.zeros_like(labels), labels)
        confusion_matrix = tf.confusion_matrix(labels=labels, 
                                               predictions=predictions, num_classes=19, weights=weights)
        
        from tensorflow.python.ops import math_ops
        from tensorflow.python.ops import array_ops
        from tensorflow.python.framework import dtypes
        """Compute the mean intersection-over-union via the confusion matrix."""
        """The code below is from:
    https://github.com/tensorflow/tensorflow/blob/fa8c1eabd06f3043be820bf476e8413818853f17/tensorflow/python/ops/metrics_impl.py#L1087"""
        total_cm = confusion_matrix
        sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
        sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
        cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(
          math_ops.cast(
              math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = array_ops.where(
          math_ops.greater(denominator, 0), denominator,
          array_ops.ones_like(denominator))
        iou = math_ops.div(cm_diag, denominator)

        # If the number of valid entries is 0 (no classes) we return 0.
        mean_IoU = array_ops.where(
          math_ops.greater(num_valid_entries, 0),
          math_ops.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
        
        
        add_moving_summary(mean_IoU)
        tf.summary.scalar('meanIoU', mean_IoU)
        
        return mean_IoU

