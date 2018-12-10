import sys
sys.path.append('..')
import tensorflow as tf
from utils.ops import prelu, spatial_dropout, unpool, unpool2
slim = tf.contrib.slim

@slim.add_arg_scope
def initial_block(inputs, is_training=True, scope="initial_block"):
    net = slim.conv2d(inputs, 13, [3,3], stride=2, activation_fn=None, scope=scope+'_conv')
    #net = slim.batch_norm(net, is_training, scope=scope+'_bn')
    #net = prelu(net, scope=scope+'_prelu')

    pool = slim.max_pool2d(inputs, [2,2], stride=2, scope=scope+'_max_pool')

    net_concat = tf.concat([net, pool], axis=3, name=scope+'_concat')
    net_concat = slim.batch_norm(net_concat, is_training, scope=scope+'_bn')
    net_concat = prelu(net_concat, scope=scope+'_prelu')

    return net_concat

@slim.add_arg_scope
def bottleneck_down(inputs, output_depth, regularizer_prob, is_training=True, projection_ratio=4, scope='bottleneck_downsample'):
    '''
    Perform downsample bottleneck
    
    '''
    # Reduced depth is for reducing 1x1 conv dim called projection, and expand in the next 1x1 called expansion
    projection_ratio = projection_ratio
    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)
    ###########MAIN BRANCH###############
    net_main, pooling_indices = tf.nn.max_pool_with_argmax(inputs, ksize=[1,2,2,1],
                                                            strides=[1,2,2,1],
                                                            padding='SAME',
                                                            name=scope+'_main_max_pool')

    inputs_shape = inputs.get_shape().as_list()
    depth_to_pad = abs(inputs_shape[3]-output_depth)
    paddings = tf.convert_to_tensor([[0,0],[0,0],[0,0],[0,depth_to_pad]])
    # TODO: understand how to pad
    net_main = tf.pad(net_main, paddings=paddings, name=scope+'_main_padding')

    ############SUB BRANCH################
    # First projection is replaced with 2x2 kernel and stride 2
    net = slim.conv2d(inputs, reduced_depth, [2,2], stride=2, scope=scope+'_conv1')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn1')
    net = prelu(net, scope=scope+'_prelu1')

    # Second conv
    net = slim.conv2d(net, reduced_depth, [3,3], scope=scope+'_conv2')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn2')
    net = prelu(net, scope=scope+'_prelu2')

    # Final projection with 1x1 kernel
    net = slim.conv2d(net, output_depth, [1,1], stride=1, scope=scope+'_conv3')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn3')
    net = prelu(net, scope=scope+'_prelu3')
    
    # Regularizer
    net = spatial_dropout(net, p=regularizer_prob, seed=0, is_training=is_training, scope=scope+'_spatial_dropout')

    net = tf.add(net, net_main, name=scope+'_add')
    net = prelu(net, scope=scope+'_prelu4')

    return net, pooling_indices, inputs_shape

@slim.add_arg_scope
def bottleneck_dila(inputs, output_depth, regularizer_prob, dilation_rate, projection_ratio=4, is_training=True, scope='bottleneck_dilation'):
    
    projection_ratio = projection_ratio

    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)
    
    if not dilation_rate:
        raise ValueError('Dilation rate is not given.')
    
    net_main = inputs

    net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn1')
    net = prelu(net, scope=scope+'_prelu')
    
    net = slim.conv2d(net, reduced_depth, [3,3], rate=dilation_rate, scope=scope+'_dilated_conv2')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn2')
    net = prelu(net, scope=scope+'_prelu2')

    net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn3')
    net = prelu(net, scope=scope+'_prelu3')

    # Regularizer
    net = spatial_dropout(net, p=regularizer_prob, seed=0, is_training=is_training, scope=scope+'_spatial_dropout')
    
    # Add the main branch
    net = tf.add(net_main, net, name=scope+'_add_dilated')
    net = prelu(net, scope=scope+'_prelu5')

    return net

@slim.add_arg_scope
def bottleneck_asym(inputs, output_depth, regularizer_prob, is_training=True, projection_ratio=4,scope='bottleneck_asym'):
    
    projection_ratio = projection_ratio
    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)
    
    net_main = inputs

    net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn1')
    net = prelu(net, scope=scope+'_prelu1')

    net = slim.conv2d(net, reduced_depth, [5,1], scope=scope+'_asym_conv2a')
    net = slim.conv2d(net, reduced_depth, [1,5], scope=scope+'_asym_conv2b')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn2')
    net = prelu(net, scope=scope+'_prelu2')

    net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn3')
    net = prelu(net, scope=scope+'_prelu3')

    net = spatial_dropout(net, p=regularizer_prob, seed=0, is_training=is_training, scope=scope+'_spatial_dropout')
    
    net = tf.add(net_main, net, name=scope+'_add_asym')
    net = prelu(net, scope=scope+'_prelu4')

    return net

@slim.add_arg_scope
def bottleneck(inputs, output_depth, regularizer_prob, projection_ratio=4, is_training=True, scope='bottleneck_regular'):
    
    projection_ratio = projection_ratio
    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)

    net_main = inputs

    net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn1')
    net = prelu(net, scope=scope+'_prelu1')

    net = slim.conv2d(net, reduced_depth, [3,3], scope=scope+'_conv2')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn2')
    net = prelu(net, scope=scope+'_prelu2')

    net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
    net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn3')
    net = prelu(net, scope=scope+'_prelu3')

    net = spatial_dropout(net, p=regularizer_prob, seed=0, is_training=is_training, scope=scope+'_spatial_dropout')

    net = tf.add(net_main, net, name=scope+'_add_regular')
    net = prelu(net, scope=scope+'_prelu4')

    return net

@slim.add_arg_scope
def bottleneck_up(inputs, output_depth, pooling_indices, regularizer_prob, projection_ratio=4, is_training=True, scope='bottleneck_upsamle'):
    
    projection_ratio = projection_ratio
    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)

    if pooling_indices == None:
        raise ValueError('Pooling indices are not given!')
    

    ######## Main Branch##############
    net_unpool = slim.conv2d(inputs, output_depth, [1,1], scope=scope+'_main_conv1')
    net_unpool = slim.batch_norm(net_unpool, is_training=is_training, scope=scope+'_bn1')
    net_unpool = unpool2(net_unpool, pooling_indices, scope=scope+'_unpool')   

    ########Sub Branch#################
    net_sub = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1_sub')
    net_sub = slim.batch_norm(net_sub, is_training=is_training, scope=scope+'_bn2')
    net_sub = prelu(net_sub, scope=scope+'_prelu1')

    # Transpose conv
    net_sub = slim.conv2d_transpose(net_sub, reduced_depth, [3,3], stride=2, scope=scope+'_conv2_transpose')
    net_sub = slim.batch_norm(net_sub, is_training=is_training, scope=scope+'_bn3')
    net_sub = prelu(net_sub, scope=scope+'_prelu2')

    net_sub = slim.conv2d(net_sub, output_depth, [1,1], scope=scope+'_conv3')
    net_sub = slim.batch_norm(net_sub, is_training=is_training, scope=scope+'_bn4')
    net_sub = prelu(net_sub, scope=scope+'_prelu3')
    net_sub = spatial_dropout(net_sub, p=regularizer_prob, seed=0, is_training=is_training, scope=scope+'_spatial_dropout')
    
    net = tf.add(net_unpool, net_sub, name=scope+'_add_upsample')
    net = prelu(net, scope=scope+'_prelu4')
    return net

def arg_scope(weight_decay=2e-4,
                   batch_norm_decay=0.1,
                   batch_norm_epsilon=0.001):
  '''
  The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
  Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.

  INPUTS:
  - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

  OUTPUTS:
  - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    # Set parameters for batch_norm.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope   