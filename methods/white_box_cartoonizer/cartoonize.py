"""
Internal code snippets were obtained from https://github.com/SystemErrorWang/White-box-Cartoonization/

For it to work tensorflow version 2.x changes were obtained from https://github.com/steubk/White-box-Cartoonization 
"""
import os
import uuid
import time
import subprocess
import sys

import cv2
import numpy as np

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from methods.white_box_cartoonizer.components.guided_filter import gf
from methods.white_box_cartoonizer.components.network import nk

weights_dir = f'{os.getcwd()}/methods/white_box_cartoonizer/saved_models'
gpu = len(sys.argv) < 2 or sys.argv[1] != '--cpu'

class WB_Cartoonize:
    def __init__(self):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError("Weights Directory not found, check path")             
    
    def resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720*h/w), 720
            else:
                h, w = 720, int(720*w/h)
        image = cv2.resize(image, (w, h),
                            interpolation=cv2.INTER_AREA)
        h, w = (h//8)*8, (w//8)*8
        image = image[:h, :w, :]
        return image

    def load_model(self, weights_dir, gpu):
        try:
            tf.disable_eager_execution()
        except:
            None
        
        tf.reset_default_graph()

        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        network_out = nk.unet_generator(self.input_photo)
        self.final_out = gf.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)
        
        if gpu:
            gpu_options = tf.GPUOptions(allow_growth=True)
            device_count = {'GPU':1}
        else:
            gpu_options = None
            device_count = {'GPU':0}
        
        config = tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)
        
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(weights_dir))

    def infer(self, image):
        self.input_photo = image
        self.load_model(weights_dir, gpu)
        image = self.resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output