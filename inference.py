#-*- coding=utf-8 -*-
import tensorflow as tf

from models import hccr_cnnnet

model_path='/.../.../checkpoint' #模型保存路径
inf_pic='/.../.../input.jpg'     #推理图片路径
 
def inference(model_path,inf_pic):
  files=[]
  channels=1
  img_size=[96,96]

  def _parse_function(filename):
    image_decoded = tf.image.decode_jpeg(tf.read_file(filename),channels=channels)
    image_decoded = tf.image.resize_images(image_decoded, img_size)
    return image_decoded

  with tf.Graph().as_default() as g:
    
    image_batch = tf.expand_dims(_parse_function(inf_pic),0)
    logits = hccr_cnnnet(image_batch,train=False,regularizer=None,channels=channels)
    label_pre = tf.argmax(logits, 1)
    saver=tf.train.Saver()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            label = sess.run(label_pre)
        else:
            print('No checkpoint file found !')
  return label

result = inference(model_path=model_path,inf_dir=inf_dir)
print(result)
