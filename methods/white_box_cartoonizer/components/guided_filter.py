"""
Code copyrights are with: https://github.com/SystemErrorWang/White-box-Cartoonization/

To adapt the code with tensorflow v2 changes obtained from: https://github.com/steubk/White-box-Cartoonization 
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import numpy as np

class gf:
    def tf_box_filter(x, r):
        k_size = int(2*r+1)
        ch = x.get_shape().as_list()[-1]
        weight = 1/(k_size**2)
        box_kernel = weight*np.ones((k_size, k_size, ch, 1))
        box_kernel = np.array(box_kernel).astype(np.float32)
        output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
        return output



    def guided_filter(x, y, r, eps=1e-2):
        
        x_shape = tf.shape(x)
        #y_shape = tf.shape(y)

        N = gf.tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

        mean_x = gf.tf_box_filter(x, r) / N
        mean_y = gf.tf_box_filter(y, r) / N
        cov_xy = gf.tf_box_filter(x * y, r) / N - mean_x * mean_y
        var_x  = gf.tf_box_filter(x * x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = gf.tf_box_filter(A, r) / N
        mean_b = gf.tf_box_filter(b, r) / N

        output = tf.add(mean_A * x, mean_b, name='final_add')

        return output



    def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
        
        #assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4
    
        lr_x_shape = tf.shape(lr_x)
        #lr_y_shape = tf.shape(lr_y)
        hr_x_shape = tf.shape(hr_x)
        
        N = gf.tf_box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

        mean_x = gf.tf_box_filter(lr_x, r) / N
        mean_y = gf.tf_box_filter(lr_y, r) / N
        cov_xy = gf.tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
        var_x  = gf.tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = tf.image.resize_images(A, hr_x_shape[1: 3])
        mean_b = tf.image.resize_images(b, hr_x_shape[1: 3])

        output = mean_A * hr_x + mean_b
        
        return output

