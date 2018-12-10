import tensorflow as tf
import numpy as np
xa_initializer = tf.contrib.layers.xavier_initializer
slim = tf.contrib.slim

def prelu(x, scope, decoder=False):

    '''
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow

    For the decoder portion, prelu becomes just a normal prelu

    INPUTS:
    - x(Tensor): a 4D Tensor that undergoes prelu
    - scope(str): the string to name your prelu operation's alpha variable.
    - decoder(bool): if True, prelu becomes a normal relu.

    OUTPUTS:
    - pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.

    '''
    #If decoder, then perform relu and just return the output
    if decoder:
        return tf.nn.relu(x, name=scope)

    alpha= tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def spatial_dropout(x, p, seed, scope, is_training=True):
    '''
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input feature map.
    Note that p stands for the probability of dropping, but tf.nn.relu uses probability of keeping.

    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------

    INPUTS:
    - x(Tensor): a 4D Tensor of the input feature map.
    - p(float): a float representing the probability of dropping a layer
    - seed(int): an integer for random seeding the random_uniform distribution that runs under tf.nn.relu
    - scope(str): the string name for naming the spatial_dropout
    - is_training(bool): to turn on dropout only when training. Optional.

    OUTPUTS:
    - output(Tensor): a 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    '''
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        #noise_shape = tf.constant(value=[-1, 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x

def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, scope=''):
    '''
    Unpooling function based on the implementation by Panaetius at https://github.com/tensorflow/tensorflow/issues/2169

    INPUTS:
    - inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
    - mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
    - k_size(list): a list of values representing the dimensions of the unpooling filter.
    - output_shape(list): a list of values to indicate what the final output shape should be after unpooling
    - scope(str): the string name to name your scope

    OUTPUTS:
    - ret(Tensor): the returned 4D tensor that has the shape of output_shape.

    '''
    with tf.variable_scope(scope):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret


def compute_IoU(confusion_matrix):
    #confusion_matrix = tf.contrib.metrics.confusion_matrix(labels, predictions)
    mIoU = 0
    num_classes = confusion_matrix.shape[0]
    class_IoU = []
    for i in range(num_classes):
        # IoU = true_positive / (true_positive + false_positive + false_negative)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i]) - TP
        if (TP + FP + FN) == 0:
            num_classes -= 1
            continue
        IoU = TP*1.0 / (TP + FP + FN)
        class_IoU.append(IoU)
        mIoU += IoU
    try:
        mIoU /= num_classes
    except: 
        print(confusion_matrix)

    return mIoU, class_IoU


def unpool2(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]
        
        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret
    
def loss_wi_ignore(logits, labels, num_classes, ignore_label):
    
    '''
    Ref:https://github.com/tensorflow/models/blob/master/research/deeplab/utils/train_utils.py
    
    The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    '''
    #ENet class weights
    class_weights = [3.044630406611936, 12.859237726097096, 4.510618814479376, 38.149414675592844, 35.24041465174455, 31.4804554826761, 45.77796608208582, 39.66961410413346, 6.067443465405894, 32.15635357347918, 17.129877639929358, 31.57929663622227, 47.33978418471209, 11.60256216711002, 44.596951840626446, 45.23276260758076, 45.280616737115544, 48.1480158812628, 41.92477381137463]
    class_weights = np.array(class_weights) / 10 # For finetune after training on coarse, lr=2.5e-4
    #class_weights = np.array(class_weights) # For train from sratch
    labels = tf.reshape(labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(labels, ignore_label))
    one_hot_labels = tf.one_hot(indices=labels, depth=num_classes, on_value=1.0, off_value=0.0)
    
    weights = one_hot_labels * class_weights
    weights = tf.reduce_sum(weights, 1)
    weights = tf.reshape(weights,shape=[-1])
    weights = weights * not_ignore_mask
        
    loss = tf.losses.softmax_cross_entropy(
           one_hot_labels,
           tf.reshape(logits, shape=[-1, num_classes]),
           weights=weights)
    
    return loss

def focal_loss(logits, labels, num_classes, ignore_label, scope, gamma=2, scale=10):
    '''
    focal loss: cross entropy with softmax
    
    logits: model output logits
    labels: gts
    '''
    # Get class weights
    class_weights = [3.044630406611936, 12.859237726097096, 4.510618814479376, 38.149414675592844, 35.24041465174455, 31.4804554826761, 45.77796608208582, 39.66961410413346, 6.067443465405894, 32.15635357347918, 17.129877639929358, 31.57929663622227, 47.33978418471209, 11.60256216711002, 44.596951840626446, 45.23276260758076, 45.280616737115544, 48.1480158812628, 41.92477381137463]
    # class weights make loss 7 times bigger, l2 weight decay is not working, so make class weights smaller
    class_weights = np.array(class_weights) / scale
    
    # reshape logits to [-1,19] and gt get rid of ignore label and reshape to [-1,19]
    logits = tf.reshape(logits, shape=[-1, num_classes]) # shape: [-1,19]
    labels = tf.where(tf.equal(labels, ignore_label), tf.zeros_like(labels), labels) # gt shape: [b, h, w]
    one_hot_labels = tf.one_hot(indices=labels, depth=num_classes, on_value=1.0, off_value=0.0) # shape: [b, h, w, 19]
    
    # prob_gamma
    prob = tf.nn.softmax(logits)
    prob_gamma = tf.pow(1.0-prob, gamma)
    
    # FL
    t = tf.multiply(prob_gamma, tf.reshape(one_hot_labels,[-1,num_classes]))
    fl = - 1.0 * tf.multiply(t, tf.log(prob))
    
    loss = fl * class_weights
    print 'focal loss', tf.reduce_sum(loss)
    return tf.reduce_sum(loss)
    



def matrix_wi_ignore(predictions, labels, ignore_label, num_classes):
    
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(labels, shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, ignore_label))    
    labels = tf.where(tf.equal(labels, ignore_label), tf.zeros_like(labels), labels)
    
    confusion_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)
    #mIOU, class_IOU = compute_IoU(confusion_matrix, num_classes)
    
    return confusion_matrix
    


def group_norm(x, groups=32, eps=1e-5, scope='group_norm'):
    '''
    Dummy implementation of group normalization.    
    '''
    
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        
        G = min(groups, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)

        # Normalize        
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta
        
    return x


def sync_batch_norm(inputs,
                    decay=0.999,
                    center=True,
                    scale=False,
                    epsilon=0.001,
                    activation_fn=None,
                    updates_collections=tf.GraphKeys.UPDATE_OPS,
                    is_training=True,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None,
                    num_dev=1):
  '''
  num_dev is how many gpus you use.
  '''
  

  from tensorflow.contrib.nccl.ops import gen_nccl_ops
  from tensorflow.contrib.framework import add_model_variable

  red_axises = [0, 1, 2]
  num_outputs = inputs.get_shape().as_list()[-1]

  if scope is None:
    scope = 'BatchNorm'

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope,
      'BatchNorm',
      reuse=reuse,
      custom_getter=layer_variable_getter) as sc:

    gamma = tf.get_variable(name='gamma', shape=[num_outputs], dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0), trainable=trainable,
                            collections=variables_collections)

    beta  = tf.get_variable(name='beta', shape=[num_outputs], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0), trainable=trainable,
                            collections=variables_collections)

    moving_mean = tf.get_variable(name='moving_mean', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0), trainable=False,
                                collections=variables_collections)
                                
    moving_var = tf.get_variable(name='moving_variance', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=False,
                                collections=variables_collections)

    if is_training and trainable:
      
      if num_dev == 1:
        mean, var = tf.nn.moments(inputs, red_axises)
      else:
        shared_name = tf.get_variable_scope().name
        batch_mean        = tf.reduce_mean(inputs, axis=red_axises)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axises)
        batch_mean        = gen_nccl_ops.nccl_all_reduce(
          input=batch_mean,
          reduction='sum',
          num_devices=num_dev,
          shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
        batch_mean_square = gen_nccl_ops.nccl_all_reduce(
          input=batch_mean_square,
          reduction='sum',
          num_devices=num_dev,
          shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        mean              = batch_mean
        var               = batch_mean_square - tf.square(batch_mean)
      outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

      if int(outputs.device[-1])== 0:
        update_moving_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        update_moving_var_op  = tf.assign(moving_var,  moving_var  * decay + var  * (1 - decay))
        add_model_variable(moving_mean)
        add_model_variable(moving_var)
        
        if updates_collections is None:
          with tf.control_dependencies([update_moving_mean_op, update_moving_var_op]):
            outputs = tf.identity(outputs)
        else:
          ops.add_to_collections(updates_collections, update_moving_mean_op)
          ops.add_to_collections(updates_collections, update_moving_var_op)
          outputs = tf.identity(outputs)
      else:
        outputs = tf.identity(outputs)

    else:
      outputs,_,_ = nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean, variance=moving_var, epsilon=epsilon, is_training=False)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
