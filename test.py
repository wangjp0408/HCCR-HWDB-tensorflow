#-*- coding=utf-8 -*-
import os
import tensorflow as tf

from models import hccr_cnnnet

gpunum='0'

batch_size = 64
img_size=[96,96]
channels=1

save_path='/.../.../checkpoint' #模型保存路径
test_dir='/.../.../test'        #测试图片路径

files=[]
labels=[]

os.environ['CUDA_VISIBLE_DEVICES']=gpunum

def _parse_function(filename, label):
  image_decoded = tf.image.decode_jpeg(tf.read_file(filename),channels=channels)
  image_decoded = tf.image.resize_images(image_decoded, img_size)
  image_decoded = tf.cast(image_decoded , tf.float32)
  label = tf.cast(label,tf.int32)
  return image_decoded, label

with tf.Graph().as_default() as g:
    
    for label_name in os.listdir(test_dir):
        for file_name in os.listdir(test_dir+'/'+label_name):
            files.append(test_dir + '/'+label_name+'/'+file_name)
            labels.append(int(label_name))

    files=tf.constant(files)
    labels=tf.constant(labels)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(_parse_function)#,num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    
    image_batch,label_batch= dataset.make_one_shot_iterator().get_next()
    
    logits=hccr_cnnnet(image_batch,train=False,regularizer=None,channels=channels)
    
    prob_batch = tf.nn.softmax(logits)
    accuracy_top1_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 1), tf.float32))
    accuracy_top5_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 5), tf.float32))
    accuracy_top10_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 10), tf.float32))
    '''
    variable_ave = tf.train.ExponentialMovingAverage(0.99)
    variables_to_restore = variable_ave.variables_to_restore()
    '''
    saver=tf.train.Saver()
    
    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            iternum=0
            top1sum=0
            top5sum=0
            top10sum=0

            while True:
                try:
                    top1,top5,top10 = sess.run([accuracy_top1_batch,accuracy_top5_batch,accuracy_top10_batch])
                    iternum=iternum+1
                    top1sum=top1sum+top1
                    top5sum=top5sum+top5
                    top10sum=top10sum+top10
                    if iternum%500==0:
                        print("The current test accuracy (in %d pics) = top1: %g , top5: %g ，top10: %g." % (iternum*batch_size,top1sum/iternum,top5sum/iternum,top10sum/iternum))
                except tf.errors.OutOfRangeError:
                    print("The final test accuracy (in %d pics) = top1: %g , top5: %g ，top10: %g." % (iternum*batch_size,top1sum/iternum,top5sum/iternum,top10sum/iternum))
                    print('Test finished...')
                    break
        else:
            print('No checkpoint file found !')
