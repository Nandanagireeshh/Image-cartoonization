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

import network
import guided_filter

class WB_Cartoonize:
    def __init__(self, weights_dir, gpu):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError("Weights Directory not found, check path")
        self.load_model(weights_dir, gpu)
        print("Weights successfully loaded")
    
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
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

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

