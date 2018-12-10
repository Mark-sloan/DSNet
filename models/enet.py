import sys
sys.path.append('..')
from layers.layers_enet import initial_block, bottleneck, bottleneck_asym, bottleneck_dila, bottleneck_down, bottleneck_up, arg_scope
import tensorflow as tf
slim = tf.contrib.slim
from utils.dataset import Dataset


class ENet:
    def __init__(self, version, inputs, num_classes, args, reuse=False, is_training=True):                
        self.scope = 'ENet'
        self.version = version
        self.model_name = self.scope + '_' + version
        self.input_images = inputs[0]
        self.gts = inputs[1]
        self.num_classes = num_classes
        self.args = args
        self.reuse = reuse
        self.is_training = is_training
        self.build_network()
            
    def build_network(self, **kwargs):
        if self.version == 'v0':
            self._building_v0()   
        elif self.version == 'v1':
            self._building_v1()
        elif self.version == 'v2':
            self._building_v2()
        elif self.version == 'gn':
            # number of groups
            self._building_gn()

    def _building_v1(self):
        
        '''
        More params ENet version model. 
        By adding 4 stages more layers.
        '''
        print('#### Building Version 1 Model ######')
        # GET BATCH
        if self.is_training == True:
            batch_size = self.args.batch_size
        elif self.is_training == False:
            batch_size = self.args.batch_size_val
        # GET SHAPE
        inputs_shape = self.input_images.get_shape().as_list()
        self.input_images.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))
        
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with slim.arg_scope([initial_block, bottleneck, bottleneck_down, bottleneck_asym, bottleneck_up, bottleneck_dila], is_training=self.is_training),\
                slim.arg_scope([slim.batch_norm], fused=True),\
                slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):

                ####Initial Block####
                net = initial_block(self.input_images, scope='initial_block')
                ###STAGE ONE#####                 
                STAGE = 1 

                net, pooling_indices_1, inputs_shape_1 = bottleneck_down(net, output_depth=64, regularizer_prob=0.01, scope='bottleneck'+str(STAGE)+'_0')
                
                for i in range(4):
                    net = bottleneck(net, output_depth=64, regularizer_prob=0.01, scope='bottleneck'+str(STAGE)+'_%d' % (i+1))
                
                
                ####STAGE TWO AND THREE or even more####
                STAGE = STAGE + 1
                with slim.arg_scope([bottleneck, bottleneck_down, bottleneck_asym, bottleneck_dila], regularizer_prob=0.1):

                    net, pooling_indices_2, inputs_shape_2 = bottleneck_down(net, output_depth=128, scope='bottleneck'+str(STAGE)+'_0')

                    # Repeat stage 2 to get stage 3 or more stage
                    for stage in range(STAGE,STAGE+6):

                        net = bottleneck(net, output_depth=128, scope='bottleneck'+str(stage)+'_1')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=2, scope='bottleneck'+str(stage)+'_2')
                        net = bottleneck_asym(net, output_depth=128, scope='bottleneck'+str(stage)+'_3')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=4, scope='bottleneck'+str(stage)+'_4')
                        net = bottleneck(net, output_depth=128, scope='bottleneck'+str(stage)+'_5')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=8, scope='bottleneck'+str(stage)+'_6')
                        net = bottleneck_asym(net, output_depth=128, scope='bottleneck'+str(stage)+'_7')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=16, scope='bottleneck'+str(stage)+'_8')
                    print('We have {} encoding stages'.format(stage))
                    STAGE = stage + 1
                                    
                ###############DECODER#######
                # STAGE FOUR is UPSAMPING
                with slim.arg_scope([bottleneck, bottleneck_up], regularizer_prob=0.1):
                    # Upsample

                    net = bottleneck_up(net, output_depth=64, pooling_indices=pooling_indices_2, scope='bottleneck'+str(STAGE)+'_0')
                    
                    net = bottleneck(net, output_depth=64, scope='bottleneck'+str(STAGE)+'_1')
                    net = bottleneck(net, output_depth=64, scope='bottleneck'+str(STAGE)+'_2')

                    ####STAGE FIVE#####
                    STAGE = STAGE + 1
                    # Upsample
                    net = bottleneck_up(net, output_depth=16, pooling_indices=pooling_indices_1, scope='bottleneck'+str(STAGE)+'_0')                    
                    net = bottleneck(net, output_depth=16, scope='bottleneck'+str(STAGE)+'_1')

                ########FINAL CONVULUTION##########
                self.logits = slim.conv2d_transpose(net, self.num_classes, [2,2], stride=2, scope='fullconv')
                #self.prob = tf.nn.softmax(self.logits, name='logits_to_softmax')
    
    
    def _building_v0(self):
        '''
        Original ENet model.
        '''
        # GET BATCH
        if self.is_training == True:
            batch_size = self.args.batch_size
        elif self.is_training == False:
            batch_size = self.args.batch_size_val
        # GET SHAPE
        inputs_shape = self.input_images.get_shape().as_list()
        self.input_images.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))
        
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with slim.arg_scope([initial_block, bottleneck, bottleneck_down, bottleneck_asym, bottleneck_up, bottleneck_dila], is_training=self.is_training),\
                slim.arg_scope([slim.batch_norm], fused=True),\
                slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):

                ####Initial Block####
                net = initial_block(self.input_images, scope='initial_block')
                ###STAGE ONE#####                 
                STAGE = 1 

                net, pooling_indices_1, inputs_shape_1 = bottleneck_down(net, output_depth=64, regularizer_prob=0.01, scope='bottleneck'+str(STAGE)+'_0')
                
                for i in range(4):
                    net = bottleneck(net, output_depth=64, regularizer_prob=0.01, scope='bottleneck'+str(STAGE)+'_%d' % (i+1))
                

                ####STAGE TWO AND THREE or even more####
                STAGE = STAGE + 1
                with slim.arg_scope([bottleneck, bottleneck_down, bottleneck_asym, bottleneck_dila], regularizer_prob=0.1):

                    net, pooling_indices_2, inputs_shape_2 = bottleneck_down(net, output_depth=128, scope='bottleneck'+str(STAGE)+'_0')

                    # Repeat stage 2 to get stage 3 or more stage
                    for stage in range(STAGE,STAGE+2):
                        net = bottleneck(net, output_depth=128, scope='bottleneck'+str(stage)+'_1')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=2, scope='bottleneck'+str(stage)+'_2')
                        net = bottleneck_asym(net, output_depth=128, scope='bottleneck'+str(stage)+'_3')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=4, scope='bottleneck'+str(stage)+'_4')
                        net = bottleneck(net, output_depth=128, scope='bottleneck'+str(stage)+'_5')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=8, scope='bottleneck'+str(stage)+'_6')
                        net = bottleneck_asym(net, output_depth=128, scope='bottleneck'+str(stage)+'_7')
                        net = bottleneck_dila(net, output_depth=128, dilation_rate=16, scope='bottleneck'+str(stage)+'_8')
                    STAGE = stage + 1
                
                self.encoded = slim.conv2d(net, self.num_classes, [1,1], stride=1, activation_fn=None, scope=self.scope+'_encoded')
                
                #####STAGE FOUR##########DECODER#######
                # STAGE FOUR is UPSAMPING
                with slim.arg_scope([bottleneck, bottleneck_up], regularizer_prob=0.1):
                    # Upsample

                    net = bottleneck_up(net, output_depth=64, pooling_indices=pooling_indices_2, scope='bottleneck'+str(STAGE)+'_0')
                    
                    net = bottleneck(net, output_depth=64, scope='bottleneck'+str(STAGE)+'_1')
                    net = bottleneck(net, output_depth=64, scope='bottleneck'+str(STAGE)+'_2')

                    ####STAGE FIVE#####
                    STAGE = STAGE + 1
                    # Upsample
                    net = bottleneck_up(net, output_depth=16, pooling_indices=pooling_indices_1, scope='bottleneck'+str(STAGE)+'_0')                    
                    net = bottleneck(net, output_depth=16, scope='bottleneck'+str(STAGE)+'_1')

                ########FINAL CONVULUTION##########
                self.logits = slim.conv2d_transpose(net, self.num_classes, [2,2], stride=2, scope='fullconv')


                    


def get_enet(version, inputs, num_classes, args, reuse=False, is_training=True):
    '''
    get different version of enet
    '''
    enet = ENet(version, inputs, num_classes, args, reuse, is_training)

    return enet


if __name__ == '__main__':
    import argparse
    dataset_dir = '/home/smartcar/dataset/cityscapes/cityscapes_data'
    test = Dataset(dataset_name='cityscapes', batch_size=4, sub_rate=1, dataset_dir=dataset_dir, mode='train')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Train batch size')
    args = parser.parse_args()
    images, gts = test.next_batch()
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        tf.train.start_queue_runners(coord=coord, sess=sess)
        enet = get_enet(version='v1', inputs=(images, gts), num_classes=19, reuse=False, is_training=True, args=args)
        print enet.model_name
        print enet.logits
