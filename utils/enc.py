import tensorflow as tf
from tensorpack import *
slim = tf.contrib.slim

def enc(inputs, K, name='encoding'):
    '''
    inputs: BxNxD
    codewords: KxD
    scales: Kx1
    r: BxNxKxD
    l2: BxNxK
    A: BxNxK
    E: 
    '''
    _, h, w, c = inputs.get_shape().as_list()
    
    D = c
    std = 1./((K*D)**(1/2))
    # codewords KxD
    codewords = tf.get_variable(name=name+'/codewords', shape=(K,D), 
                                initializer=tf.random_uniform_initializer(minval=-std,maxval=std,seed=0),
                                regularizer=None, trainable=True)

    # scales K
    scales = tf.get_variable(name=name+'/scales', shape=(K,), 
                             initializer=tf.random_uniform_initializer(minval=-1,maxval=0,seed=0),
                             regularizer=None, trainable=True)
    inputs = tf.reshape(inputs, (-1,h*w,c))

    r = []
    for k in range(K):
        r.append(inputs - codewords[k]) 
    r = tf.transpose(tf.convert_to_tensor(r),(1,2,0,3))
    
    l2 = tf.reduce_sum(tf.square(r), axis=3)
    l2scaled = - scales * l2

    #A = tf.reshape(tf.nn.softmax(l2scaled), [b*h*w*K,1])
    #r = tf.reshape(r, [b*h*w*K,D])
    
    A = tf.reshape(tf.nn.softmax(l2scaled), [-1,1])
    r = tf.reshape(r, [-1,D])
    
    e = A * r    
    e = tf.reshape(e,(-1,h*w,K,D)) #BxNxKxD
    e = tf.reduce_sum(tf.reshape(e, (-1,h*w,K,D)), axis=1) #BxKxD      
    e = tf.reduce_sum(e, axis=1) # BxD
    e = BatchNorm(name+'_bn', e)
    e = tf.nn.relu(e, name=name+'_relu')
    #e = slim.batch_norm(inputs=e, decay=0.9, scope=name+'_bn')
       
    return e
    
    
if __name__ == '__main__':
    
    inputs = tf.random_uniform(shape=[4, 64, 64, 128], minval=0, maxval=1, dtype=tf.float32)
    K = 32
    
    e = enc(inputs, K)
    
    with tf.Session() as sess:
        
        e = sess.run(e)
        print e
    