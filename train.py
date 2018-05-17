#-*- coding=utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from signal import SIGINT, SIGTERM

import lbtoolbox as lb
from models import hccr_cnnnet

gpunum='0'
lr_base=0.1
lr_decay=0.1
momentum=0.9
lr_steps=7000
save_steps=7000
print_steps=100
train_nums=30000
buffer_size=100000
regular_rate=0.0005

batch_size = 128
img_size=[96,96]
channels=1

save_path='/.../.../checkpoint' #模型保存路径
train_dir='/.../.../train'      #训练图片路径
log_dir = '/.../.../log'        #日志保存路径

aug=False #是否进行图像增强？
resume=False #是否继续训练模型？

file_and_label=[]
files=[]
labels=[]
'''
losslist = []
accuracy = []
'''
os.environ['CUDA_VISIBLE_DEVICES']=gpunum


def data_augmentation(images):
    images = tf.image.random_brightness(images, max_delta=0.3)
    images = tf.image.random_contrast(images, 0.8, 1.2)
    return images

def _parse_function(filename, label):
  image_decoded = tf.image.decode_jpeg(tf.read_file(filename),channels=channels)
  image_decoded = tf.image.resize_images(image_decoded, img_size)
  image_decoded = tf.cast(image_decoded , tf.float32)
  if aug:
      image_decoded = data_augmentation(image_decoded)
  label = tf.cast(label,tf.int32)
  return image_decoded, label

for label_name in os.listdir(train_dir):
    for file_name in os.listdir(train_dir+'/'+label_name):
        file_and_label.append([label_name,train_dir + '/'+label_name+'/'+file_name])
        
file_and_label=np.array(file_and_label)
np.random.shuffle(file_and_label)
labels=list(map(int,file_and_label[:,0]))
files=list(file_and_label[:,1])

files=tf.constant(files)
labels=tf.constant(labels)

dataset = tf.contrib.data.Dataset.from_tensor_slices((files, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size).repeat()

image_batch,label_batch = dataset.make_one_shot_iterator().get_next()

regularizer=tf.contrib.layers.l2_regularizer(regular_rate)

logits=hccr_cnnnet(image_batch,train=True,regularizer=regularizer,channels=channels)

global_step=tf.Variable(0,trainable=False)

prob_batch = tf.nn.softmax(logits)
accuracy_top1_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 1), tf.float32))
accuracy_top5_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 5), tf.float32))
accuracy_top10_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 10), tf.float32))

update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#variable_ave = tf.train.ExponentialMovingAverage(0.99,global_step)
#ave_op = variable_ave.apply(tf.trainable_variables())

cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batch))
if regularizer==None:
    loss=cross_entropy_mean
else:
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

lr=tf.train.exponential_decay(lr_base,global_step,lr_steps,lr_decay,staircase=True)
train_step = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum)

with tf.control_dependencies(update_op):
    grads = train_step.compute_gradients(loss)
    train_op = train_step.apply_gradients(grads, global_step=global_step)
    
var_list = tf.trainable_variables()
if global_step is not None:
    var_list.append(global_step)
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    if resume:
            last_checkpoint = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, last_checkpoint)
            start_step = sess.run(global_step)
            print('Resume training ... Start from step %d / %d .'%(start_step,train_nums))
            resume=False
    else:
            start_step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
      for i in range(start_step,train_nums):

        _,loss_value,step=sess.run([train_op,loss,global_step])

        if i % print_steps == 0:
            top1,top5,top10=sess.run([accuracy_top1_batch,accuracy_top5_batch,accuracy_top10_batch])
            print("After %d training step(s),loss on training batch is %g.The batch test accuracy = %g , %g ，%g."%(i,loss_value,top1,top5,top10))
            '''
            losslist.append([step,loss_value])
            accuracy.append([step,top1])
            '''
        if (i!=0 and i % save_steps == 0):
                    model_name="trainnum_%d_"%train_nums
                    saver.save(sess, os.path.join(save_path, model_name), global_step=global_step)

        if u.interrupted:
                    print("Interrupted on request...")
                    break
    '''              
    file1=open(log_dir+'/loss.txt','a')
    for loss in losslist:
          loss = str(loss).strip('[').strip(']').replace(',','')
          file1.write(loss+'\n')
    file1.close()
            
    file2=open(log_dir+'/accu.txt','a')
    for acc in accuracy:
          acc = str(acc).strip('[').strip(']').replace(',','')
          file2.write(acc+'\n')
    file2.close()
    '''
    model_name="trainnum_%d_"%train_nums
    saver.save(sess,os.path.join(save_path,model_name),global_step=global_step)
    print('Train finished...')        
         
    coord.request_stop()
    coord.join(threads)
