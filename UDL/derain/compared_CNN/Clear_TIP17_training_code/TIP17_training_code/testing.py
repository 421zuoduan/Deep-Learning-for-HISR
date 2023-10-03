# -*- coding: utf-8 -*-
#!/usr/bin/env python2



# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, X. Ding, Y. Liao and J. Paisley. “Clearing the Skies: A deep network architecture for single-image rain removal”, 
# IEEE Transactions on Image Processing, vol. 26, no. 6, pp. 2944-2956, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import skimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import training

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

model_path = './model/'
pre_trained_model_path = './model/trained/model'


img_path = './TestData/input/' # the path of testing images
results_path = './TestData/results/' # the path of de-rained images


def _parse_function(filename):   
  image_string = tf.read_file(filename)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0 
  return rainy 


if __name__ == '__main__':
   imgName = os.listdir(img_path)
   num_img = len(imgName)
   
   whole_path = []
   for i in range(num_img):
      whole_path.append(img_path + imgName[i])
      
    
   filename_tensor = tf.convert_to_tensor(whole_path, dtype=tf.string)     
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=10)
   dataset = dataset.batch(batch_size=1).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   rain = iterator.get_next()  
   rain_pad = tf.pad(rain,[[0,0],[10,10],[10,10],[0,0]],"SYMMETRIC")  
   
   detail, base = training.inference(rain_pad)
   
   detail = detail[:,6:tf.shape(detail)[1]-6, 6:tf.shape(detail)[2]-6, :] 
   base = base[:,10:tf.shape(base)[1]-10, 10:tf.shape(base)[2]-10, :] 
   
   output = tf.clip_by_value(base + detail, 0., 1.)
   output = output[0,:,:,:]

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True   
   saver = tf.train.Saver()
   
   with tf.Session(config=config) as sess:
      with tf.device('/gpu:0'): 
          if tf.train.get_checkpoint_state(model_path):  
              ckpt = tf.train.latest_checkpoint(model_path)  # try your own model 
              saver.restore(sess, ckpt)
              print ("Loading model")
          else:
             saver.restore(sess, pre_trained_model_path) # try a pre-trained model 
             print ("Loading pre-trained model")

          for i in range(num_img):     
             derained, ori = sess.run([output, rain])              
             derained = np.uint8(derained * 255.)
             index = imgName[i].rfind('.')
             name = imgName[i][:index]
             skimage.io.imsave(results_path + name +'.png', derained)         
             print('%d / %d images processed' % (i+1,num_img))
              
      print('All done')
   sess.close()   
   
   plt.subplot(1,2,1)     
   plt.imshow(ori[0,:,:,:])          
   plt.title('rainy')
   plt.subplot(1,2,2)    
   plt.imshow(derained)
   plt.title('derained')
   plt.show()       