from pyexpat import model
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, ReLU, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization
import tensorflow as tf

class ProcessFeatures():
    def __init__(self):
        super(ProcessFeatures, self).__init__()
   

    def tf_shape(self, x, rank):
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(input=x), rank)
        return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]

    # TENSORFLOW
    def corr(self, sat_matrix, grd_matrix):
        s_h, s_w, s_c = sat_matrix.get_shape().as_list()[1:]
        g_h, g_w, g_c = grd_matrix.get_shape().as_list()[1:]
        assert s_h == g_h, s_c == g_c #devono avere la stessa altezza e lo stesso numero di canali
        def warp_pad_columns(x, n):
            out = tf.concat([x, x[:, :, :n, :]], axis=2)
            return out
        n = g_w - 1
        
        x = warp_pad_columns(sat_matrix, n)

        #Correlation between Fs and Fg (use convolution but transposing the filter = correlation)
        f = tf.transpose(a=grd_matrix, perm=[1, 2, 3, 0])
        out = tf.nn.conv2d(input=x, filters=f,  strides=[1, 1, 1, 1], padding='VALID')
        h, w = out.get_shape().as_list()[1:-1]
        
        assert h==1, w==s_w
        
        out = tf.squeeze(out,axis=1)  # shape = [batch_sat, w, batch_grd]
        #Get the area with the maximum correlation
        orien = tf.argmax(input=out, axis=1)  # shape = [batch_sat, batch_grd]
        
        return out, tf.cast(orien, tf.int32)


    def crop_sat(self, sat_matrix, orien, grd_width):
        batch_sat, batch_grd = self.tf_shape(orien, 2) #ritorna le dimensioni dei due batch
        h, w, channel = sat_matrix.get_shape().as_list()[1:]
        sat_matrix = tf.expand_dims(sat_matrix, 1) # shape=[batch_sat, 1, h, w, channel]
        sat_matrix = tf.tile(sat_matrix, [1, batch_grd, 1, 1, 1])
        sat_matrix = tf.transpose(a=sat_matrix, perm=[0, 1, 3, 2, 4])  # shape = [batch_sat, batch_grd, w, h, channel]
        orien = tf.expand_dims(orien, -1) # shape = [batch_sat, batch_grd, 1]

        i = tf.range(batch_sat)
        j = tf.range(batch_grd)
        k = tf.range(w)

        x, y, z = tf.meshgrid(i, j, k, indexing='ij')
        z_index = tf.math.floormod(z + orien, w)
        x1 = tf.reshape(x, [-1])
        y1 = tf.reshape(y, [-1])
        z1 = tf.reshape(z_index, [-1])
        index = tf.stack([x1, y1, z1], axis=1)
        sat = tf.reshape(tf.gather_nd(sat_matrix, index), [batch_sat, batch_grd, w, h, channel])
        index1 = tf.range(grd_width)
        sat_crop_matrix = tf.transpose(a=tf.gather(tf.transpose(a=sat, perm=[2, 0, 1, 3, 4]), index1), perm=[1, 2, 3, 0, 4])
        # shape = [batch_sat, batch_grd, h, grd_width, channel]
        assert sat_crop_matrix.get_shape().as_list()[3] == grd_width

        return sat_crop_matrix


    def corr_crop_distance(self, sat_vgg, grd_vgg):
        corr_out, corr_orien = self.corr(sat_vgg, grd_vgg)
        sat_cropped = self.crop_sat(sat_vgg, corr_orien, grd_vgg.get_shape().as_list()[2])
        # shape = [batch_sat, batch_grd, h, grd_width, channel]

        sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])
        distance = 2 - 2 * tf.transpose(a=tf.reduce_sum(input_tensor=sat_matrix * tf.expand_dims(grd_vgg, axis=0), axis=[2, 3, 4]))
        # shape = [batch_grd, batch_sat]
        return sat_matrix, distance, corr_orien


    def conv2(self, x, scope_name, dimension=16):
        with tf.compat.v1.variable_scope(scope_name):
            layer15_output = self.conv_layer2(x, 3, [1, 2, 1, 1], 512, 256, self.trainable, True, 'conv_dim_reduct_1')

            layer16_output = self.conv_layer2(layer15_output, 3, [1, 2, 1, 1], 256, 64, self.trainable, True, 'conv5_dim_reduct_2')

            layer17_output = self.conv_layer(layer16_output, 3, 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [7, 39]

        return layer17_output

    def VGG_13_conv_v2_cir(self,x_sat, x_grd):

        sat_matrix, distance, pred_orien = self.corr_crop_distance(x_sat, x_grd)
        return x_sat, x_grd, distance, pred_orien
    
