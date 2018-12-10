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
from shuffle_tensorpack import ShuffleTensorpack


class ShuffleTensorpackInfer(ShuffleTensorpack):
    def __init__(self, H, W, batchsize):
        self.H = H
        self.W = W
        self.batchsize = batchsize

    def inputs(self):
        
        images = tf.placeholder(tf.float32, (None, self.H, self.W, 3), 'images')
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
                                                  num_groups=4) 
            for i in range(4):
                net = shuffle_unit(name='Stage'+str(stage)+'_%d'%(i+1)+'_regu', 
                                   inputs=net, num_split=2, output_depth=64, 
                                   num_groups=4)

            ###### STAGE TWO #######
            stage = stage+1
            net, pooling_indices_2 = shuffle_down(name='Stage'+str(stage)+'_0_regu', 
                                                  inputs=net, output_depth=128, 
                                                  num_groups=4)                
            ## 1,2,5,9,1,2,5,9
            net = shuffle_unit(name='Stage'+str(stage)+'_1_regu', 
                                inputs=net, num_split=2, output_depth=128, 
                                num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_2_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_3_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_4_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=9, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_5_regu', 
                                inputs=net, num_split=2, output_depth=128, 
                                num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_6_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_7_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_8_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=9, num_groups=4)
            self.net_stage_2 = net

            ###### STAGE THREE #######
            stage = stage+1

            ## 2,5,9,17,2,5,9,17
            net = shuffle_unit(name='Stage'+str(stage)+'_1_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_2_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_3_dila', 
                               inputs=net, num_split=2, output_depth=128, 
                               dilation_rate=9, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_4_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=17, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_5_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=2, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_6_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=5, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_7_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=9, num_groups=4)
            net = shuffle_unit(name='Stage'+str(stage)+'_8_dila', 
                                inputs=net, num_split=2, output_depth=128, 
                                dilation_rate=17, num_groups=4)
            self.net_stage_3 = net


            #### DECODER ###############
            #### STAGE FOUR ############
            stage = stage+1
            net = shuffle_up(name='Stage'+str(stage)+'_0', 
                            inputs=net, output_depth=64, 
                            pooling_indices=pooling_indices_2)
            net = shuffle_unit(name='Stage'+str(stage)+'_1_regu', 
                               inputs=net, num_split=2, output_depth=64,
                               num_groups=4)

            #### STAGE FIVE ############
            stage=stage+1
            net = shuffle_up(name='Stage'+str(stage)+'_0', 
                            inputs=net, output_depth=16, 
                            pooling_indices=pooling_indices_1)
            net = shuffle_unit(name='Stage'+str(stage)+'_1_regu', 
                               inputs=net, num_split=2, output_depth=16,
                               num_groups=4)

            #### FINAL CONV ############

            logits = Conv2DTranspose('fullconv', net, filters=19, 
                          kernel_size=2, strides=(2,2), use_bias=False)       
        
        return logits

    
    def build_graph(self, images, gts):

        logits = self.get_logits(images)
        total_loss = self.compute_loss(logits, gts)

        return total_loss
    
    def optimizer(self):
        
        lr = tf.get_variable('learning_rate', initializer=2e-5, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr) 

        return optimizer
    
    def compute_loss(self, logits, gts):

        loss = loss_wi_ignore(labels=gts, logits=logits, num_classes=19, ignore_label=19)
            #loss = tf.reduce_mean(loss, name='weighted_cross_entrophy_loss')
        
        return loss

