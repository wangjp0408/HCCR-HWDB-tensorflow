# -*- coding=utf-8 -*-
import tensorflow as tf

NUM_LABELS=3755
stddev=0.01
prob=0.5 #dropout

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg

def hccr_cnnnet(input_tensor,train,regularizer,channels):
 
    conv1_deep=96
    conv2_deep=128
    conv3_deep=160
    conv4_deep=256
    conv5_deep=256
    conv6_deep=384
    conv7_deep=384
    fc1_num=1024
    
    with tf.variable_scope('layer0-bn'): 
        bn0 = tf.layers.batch_normalization(input_tensor,training=train,name='bn0')
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [3, 3, channels, conv1_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1_biases = tf.get_variable("bias", [conv1_deep], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(bn0, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv1 = tf.layers.batch_normalization(tf.nn.bias_add(conv1, conv1_biases),training=train,name='bn_conv1')
        prelu1 = parametric_relu(bn_conv1)  
        
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(prelu1, ksize = [1, 3, 3, 1],strides=[1, 2, 2, 1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [3,3,conv1_deep,conv2_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv2_biases = tf.get_variable("bias", [conv2_deep], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv2 = tf.layers.batch_normalization(tf.nn.bias_add(conv2, conv2_biases),training=train,name='bn_conv2')
        prelu2 = parametric_relu(bn_conv2)

    with tf.name_scope("layer4-pool2"): 
        pool2 = tf.nn.max_pool(prelu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3,3,conv2_deep,conv3_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3_biases = tf.get_variable("bias", [conv3_deep], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv3 = tf.layers.batch_normalization(tf.nn.bias_add(conv3, conv3_biases),training=train,name='bn_conv3')
        prelu3 = parametric_relu(bn_conv3)
        
    with tf.name_scope("layer6-pool3"): 
        pool3 = tf.nn.max_pool(prelu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3,3,conv3_deep,conv4_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv4_biases = tf.get_variable("bias", [conv4_deep], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv4 = tf.layers.batch_normalization(tf.nn.bias_add(conv4, conv4_biases),training=train,name='bn_conv4')
        prelu4 = parametric_relu(bn_conv4)
        
    with tf.variable_scope("layer8-conv5"):
        conv5_weights = tf.get_variable("weight", [3,3,conv4_deep,conv5_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv5_biases = tf.get_variable("bias", [conv5_deep], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(prelu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv5 = tf.layers.batch_normalization(tf.nn.bias_add(conv5, conv5_biases),training=train,name='bn_conv5')
        prelu5 = parametric_relu(bn_conv5)
        
    with tf.name_scope("layer9-pool4"): 
        pool4 = tf.nn.max_pool(prelu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    with tf.variable_scope("layer10-conv6"):
        conv6_weights = tf.get_variable("weight", [3,3,conv5_deep,conv6_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv6_biases = tf.get_variable("bias", [conv6_deep], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(pool4, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv6 = tf.layers.batch_normalization(tf.nn.bias_add(conv6, conv6_biases),training=train,name='bn_conv6')
        prelu6 = parametric_relu(bn_conv6)
        
    with tf.variable_scope("layer11-conv7"):
        conv7_weights = tf.get_variable("weight", [3,3,conv6_deep,conv7_deep],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv7_biases = tf.get_variable("bias", [conv7_deep], initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(prelu6, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv7 = tf.layers.batch_normalization(tf.nn.bias_add(conv7, conv7_biases),training=train,name='bn_conv7')
        prelu7 = parametric_relu(bn_conv7)
        
    with tf.name_scope("layer12-pool5"): 
        pool5 = tf.nn.max_pool(prelu7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')    
        
    pool_shape = pool5.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
    reshaped = tf.reshape(pool5, [-1, nodes])

    with tf.variable_scope('layer13-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, fc1_num],initializer=tf.truncated_normal_initializer(stddev=stddev))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights)) 
        fc1_biases = tf.get_variable("bias", [fc1_num], initializer=tf.constant_initializer(0.1))
        bn_fc1=tf.layers.batch_normalization(tf.matmul(reshaped, fc1_weights) + fc1_biases,training=train,name='bn_fc1')
        fc1 = parametric_relu(bn_fc1)
        if train:
            fc1 = tf.nn.dropout(fc1, prob)

    with tf.variable_scope('layer14-output'):
        fc2_weights = tf.get_variable("weight", [fc1_num, NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=stddev))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    
    return logit
