import os
import uuid
import sys

import cv2
import numpy as np
try:
    import tensorflow.compat.v1 as tf # type: ignore
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

    def infer(self, image, return_steps=False):
        """
        Process the image through the cartoonization pipeline.
        If return_steps is True, return a dictionary containing intermediate steps.
        """
        # Resize and crop
        image = self.resize_crop(image)
        steps = {}
        steps['resized'] = image.copy()
        
        # Additional intermediate processing:
        # Grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        steps['grayscale'] = gray_bgr
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 100, 200)
        # Convert single channel edges to 3-channel image for display
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        steps['edges'] = edges_color
        
        # Smoothing using Gaussian Blur
        smooth = cv2.GaussianBlur(image, (7,7), 0)
        steps['smoothing'] = smooth
        
        # Stylization using OpenCV's stylization function
        stylized = cv2.stylization(image, sigma_s=60, sigma_r=0.07)
        steps['stylization'] = stylized
        
        # Prepare image for cartoonization
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        
        # Session Run for final cartoonization
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        cartoon_output = np.clip(output, 0, 255).astype(np.uint8)
        steps['cartoon'] = cartoon_output
        
        if return_steps:
            return steps
        else:
            return cartoon_output

if __name__ == '__main__':
    gpu = len(sys.argv) < 2 or sys.argv[1] != '--cpu'
    wbc = WB_Cartoonize(os.path.abspath('white_box_cartoonizer/saved_models'), gpu)
    img = cv2.imread('white_box_cartoonizer/test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # To view intermediate steps, set return_steps=True
    steps = wbc.infer(img, return_steps=True)
    import matplotlib.pyplot as plt # type: ignore
    # Display all intermediate steps in a subplot grid
    titles = ['Resized', 'Grayscale', 'Edges', 'Smoothing', 'Stylization', 'Cartoon']
    images = [steps['resized'], steps['grayscale'], steps['edges'], steps['smoothing'], steps['stylization'], steps['cartoon']]
    plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.show()
