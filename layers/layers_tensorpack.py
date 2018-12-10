import tensorflow as tf
from utils.ops import prelu, unpool2, unpool
slim = tf.contrib.slim
from tensorpack import *
from tensorpack.tfutils import argscope
from tensorpack.tfutils.scope_utils import under_name_scope, auto_reuse_variable_scope


'''
Use tensorpack's BatchNorm to achieve cross-GPU sync bn.

'''
#auto_reuse_variable_scope
def initial_block(name, inputs):
    '''
    This initial block is borrowed from enet.
    '''
    # Conv
    net = Conv2D(name+'_conv', inputs, filters=13, kernel_size=3, strides=(2,2), 
                use_bias=False, dilation_rate=(1, 1))  
    net = BatchNorm(name+'_bn1', net, sync_statistics='nccl')
    net = PReLU(name+'_prelu1', net)
    #net = tf.nn.relu(net, name+'_relu1')
    # Pool
    pool = MaxPooling(name+'_max_pool', inputs, 
                      pool_size=2, strides=(2,2))
    # Concat
    net_concat = tf.concat([net, pool], axis=3, name=name+'_concat')
    net_concat = BatchNorm(name+'_bn2', net_concat, sync_statistics='nccl')
    net_concat = PReLU(name+'_prelu2', net_concat)
    #net_concat = tf.nn.relu(net_concat, name+'_relu2')

    return net_concat

#auto_reuse_variable_scope
def shuffle_unit(name, inputs, num_split, output_depth, num_groups, dilation_rate=1):
    '''
    For a shuffle v2 unit, its input depth is the same as output depth,
    i.e. inputs.shape[3] = output_depth.
    
    name: name of this op
    inputs: input of this op
    num_spilt: if it is a integer, it spilts evenly; if it not, it spilt according to this num_spilt; 
               axis=-1 means the channel
    dilation_rate: dilation rate for dilated conv
    num_groups: for channel shuffle
    '''
    
    
    # Channel spilt  
    main, sub = tf.split(value=inputs, num_or_size_splits=num_split, axis=-1, name=name+'_split')
    
    ## First 1x1 conv, 1x1 conv remains channel unchanged, i.e. depth
    sub = Conv2D(name+'_conv1', sub, filters=output_depth/2, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    sub = BatchNorm(name+'_bn1', sub, sync_statistics="nccl")
    sub = PReLU(name+'_prelu1', sub)
    #sub = tf.nn.relu(sub, name+'_relu1')
    ## Sub conv may be regular conv, dilated conv
    if 'regu' in name:
        sub = Conv2D(name+'_conv2', sub, filters=output_depth/2, 
                     kernel_size=3, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    elif 'dila' in name:
        sub = Conv2D(name+'_dilated_conv2', sub, filters=output_depth/2, 
                     kernel_size=3, strides=(1,1), use_bias=False, 
                     dilation_rate=(dilation_rate, dilation_rate))

    sub = BatchNorm(name+'_bn2', sub, sync_statistics="nccl")
    sub = PReLU(name+'_prelu2', sub)
    #sub = tf.nn.relu(sub, name+'_relu2')
    ## Final 1x1 conv
    sub = Conv2D(name+'_conv3', sub, filters=output_depth/2, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    sub = BatchNorm(name+'_bn3', sub, sync_statistics="nccl")
    sub = PReLU(name+'_prelu3', sub)
    #sub = tf.nn.relu(sub, name+'_relu3')
    
    ## Concat
    net = tf.concat([main, sub], axis=-1, name=name+'_concat')
    
    ## SHUFFLE CHANNEL
    #net = _channel_shuffle(net, num_groups, scope=name+'_channel_shuffle')
    net = _channel_shuffle2(net, num_groups)
    
    return net

#auto_reuse_variable_scope
def shuffle_down(name, inputs, output_depth, num_groups):
    '''
    Downsample bottleneck unit, ref enet and shufflev2, this is new
    
    inputs:
    output_depth:
    '''
    
    # MAIN branch: max pool and 1x1 conv
    main, pooling_indices = tf.nn.max_pool_with_argmax(inputs, ksize=[1,2,2,1],
                                                            strides=[1,2,2,1],
                                                            padding='SAME',
                                                            name=name+'_main_max_pool')

    main = Conv2D(name+'_main_conv1', main, filters=output_depth/2, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    main = BatchNorm(name+'_main_bn1', main, sync_statistics="nccl")
    main = PReLU(name+'_main_prelu1', main)
    #main = tf.nn.relu(main, name+'_main_relu1')
    
    # SUB branch: 1x1 conv, conv(dw or regu conv), 1x1 conv
    ## 1x1 conv
    sub = Conv2D(name+'_sub_conv1', inputs, filters=output_depth/2, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    sub = BatchNorm(name+'_sub_bn1', sub, sync_statistics="nccl")
    sub = PReLU(name+'_sub_prelu1', sub)
    #sub = tf.nn.relu(sub, name+'_sub_relu1')
    ## conv(dw or regu conv), stride=2
    if 'regu' in name:
        sub = Conv2D(name+'_sub_conv2', sub, filters=output_depth/2, 
                     kernel_size=3, strides=(2,2), use_bias=False, dilation_rate=(1, 1))

    sub = BatchNorm(name+'_sub_bn2', sub, sync_statistics="nccl")
    sub = PReLU(name+'_sub_prelu2', sub)
    #sub = tf.nn.relu(sub, name+'_sub_relu2')
    ## 1x1 conv
    sub = Conv2D(name+'_sub_conv3', sub, filters=output_depth/2, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    sub = BatchNorm(name+'_sub_bn3', sub, sync_statistics="nccl")
    sub = PReLU(name+'_sub_prelu3', sub)
    #sub = tf.nn.relu(sub, name+'_sub_relu3')
    # CONCAT
    net = tf.concat([main, sub], axis=-1, name=name+'_concat')
    
    # SHUFFLE CHANNEL
    #net = _channel_shuffle(net, num_groups, scope=name+'_channel_shuffle')
    net = _channel_shuffle2(net, num_groups)
    return net, pooling_indices

#auto_reuse_variable_scope
def shuffle_up(name, inputs, output_depth, pooling_indices):
    '''
    Upsample
    upsample uses relu, not prelu, according to enet paper
    '''
    if pooling_indices == None:
        raise ValueError('Pooling indices are not given!')
    
    # UNPOOL 
    main = Conv2D(name+'_main_conv1', inputs, filters=output_depth, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    main = BatchNorm(name+'_main_bn1', main, sync_statistics="nccl")
    main = tf.nn.relu(main, name=name+'_main_relu1')
    up = unpool2(main, pooling_indices, scope=name+'_unpool')
    
    # TRANSPOSE
    # 1x1 conv
    sub = Conv2D(name+'_sub_conv1', inputs, filters=output_depth, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    sub = BatchNorm(name+'_sub_bn1', sub, sync_statistics="nccl")
    sub = tf.nn.relu(sub, name=name+'_sub_relu1')
    # transpose conv
    sub = Conv2DTranspose(name+'_sub_conv2_transpose', sub, filters=output_depth, 
                          kernel_size=3, strides=(2,2), use_bias=False)
    sub = BatchNorm(name+'_sub_bn2', sub, sync_statistics="nccl")
    sub = tf.nn.relu(sub, name=name+'_sub_relu2')

    # 1x1 conv
    sub = Conv2D(name+'_sub_conv3', sub, filters=output_depth, 
                 kernel_size=1, strides=(1,1), use_bias=False, dilation_rate=(1, 1))
    sub = BatchNorm(name+'_sub_bn3', sub, sync_statistics="nccl")
    sub = tf.nn.relu(sub, name=name+'_sub_relu3') 

    # ADD
    net = tf.add(up, sub, name=name+'_add')
    net = tf.nn.relu(net, name=name+'_relu4')
    
    return net

#auto_reuse_variable_scope
def _channel_shuffle(x, num_groups, scope):
    with tf.variable_scope(scope) as scope:
        n, h, w, c = x.get_shape().as_list() # Get tensor shape
        #n, h, w, c = x.shape.as_list() # Get numpy shape
        x_reshaped = tf.reshape(x, [n, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [n, h, w, c])
        
        return output
@auto_reuse_variable_scope
def _channel_shuffle2(x, num_groups):

    _, h, w, c = x.get_shape().as_list() # Get tensor shape
    #n, h, w, c = x.shape.as_list() # Get numpy shape
    x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])

    return output


