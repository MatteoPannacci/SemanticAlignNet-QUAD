import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cir_net_FOV_mb import *
from polar_input_data_orien_FOV_3_Segmap_Concatenation import InputData
from VGG_no_session import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import argparse
from PIL import Image
import scipy.io as scio
from numpy import fft

#tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()
parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=100)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=360)

args = parser.parse_args()

# Save parameters --------------------------------------- #
start_epoch = args.start_epoch
train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise
train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV
number_of_epoch = args.number_of_epoch
loss_type = 'l1'
batch_size = 8
is_training = True
loss_weight = 10.0
learning_rate_val = 1e-5
keep_prob_val = 0.8
keep_prob = 0.8
dimension = 4

print("SETTED PARAMETERS: ")
print("Train ground FOV: {}".format(train_grd_FOV))
print("Train ground noise: {}".format(train_grd_noise))
print("Test ground FOV: {}".format(test_grd_FOV))
print("Test ground noise: {}".format(test_grd_noise))
print("Number of epochs: {}".format(number_of_epoch))
print("Learning rate: {}".format(learning_rate_val))
# -------------------------------------------------------- #

def validate(dist_array, topK):
    accuracy = 0.0
    data_amount = 0.0

    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[i, :] < gt_dist)
        if prediction < topK:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(dist_array):

    pos_dist = tf.linalg.tensor_diag_part(dist_array)
    pair_n = batch_size * (batch_size - 1.0)

    # satellite to ground
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = tf.reduce_sum(input_tensor=tf.math.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

    # ground to satellite
    triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
    loss_s2g = tf.reduce_sum(input_tensor=tf.math.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

    loss = (loss_g2s + loss_s2g) / 2.0
    return loss

def train(start_epoch=14):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 0.
    '''
    # import data
    input_data = InputData()
    width = int(test_grd_FOV / 360 * 512)

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_val)

    groundNet = VGGModel(tf.keras.Input(shape=(None, None, 3)))
    satNet = VGGModelCir(tf.keras.Input(shape=(None, None, 3)),'_sat')
    segMapNet = VGGModelCir(tf.keras.Input(shape=(None, None, 3)),'_seg')
    
    processor = ProcessFeatures()

    model = Model(inputs=[groundNet.model.input, satNet.model.input, segMapNet.model.input], outputs=[groundNet.model.output, satNet.model.output, segMapNet.model.output])    
    print("Model created")


    grd_x = np.float32(np.zeros([2, 128, width, 3]))
    sat_x = np.float32(np.zeros([2, 256, 512, 3]))
    polar_sat_x = np.float32(np.zeros([2, 128, 512, 3]))
    segmap_x = np.float32(np.zeros([2, 128, 512, 3]))

    grd_features, sat_features, segmap_features = model([grd_x, polar_sat_x, segmap_x])
    sat_features = tf.concat([sat_features, segmap_features], axis=-1)
    
    # build model
    sat_matrix, grd_matrix, distance, pred_orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)#model.call([grd_x,polar_sat_x])

    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])
    pred_orientation = np.zeros([input_data.get_test_dataset_size()])

    # load Model
    model_path = "./saved_models/path/epoch/"
    model = keras.models.load_model(model_path)
    print("Model checkpoint uploaded")

    print("Validation...")
    val_i = 0
    count = 0
    while True:
        # print('      progress %d' % val_i)
        
        batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = input_data.next_batch_scan(8, grd_noise=test_grd_noise,
                                                                        FOV=test_grd_FOV)
        if batch_sat is None:
            break
        grd_features, sat_features, segmap_features = model([batch_grd, batch_sat_polar, batch_segmap])
        sat_features = tf.concat([sat_features, segmap_features], axis=-1)
        
        grd_features = tf.nn.l2_normalize(grd_features, axis=[1, 2, 3])
        sat_matrix, grd_matrix, distance, orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)

        sat_global_matrix[val_i: val_i + sat_matrix.shape[0], :] = sat_matrix
        grd_global_matrix[val_i: val_i + grd_matrix.shape[0], :] = grd_matrix
        orientation_gth[val_i: val_i + grd_matrix.shape[0]] = batch_orien

        val_i += sat_matrix.shape[0]
        count += 1

    file = './saved_models/path/filename.mat'
    scio.savemat(file, {'orientation_gth': orientation_gth, 'grd_descriptor': grd_global_matrix, 'sat_descriptor': sat_global_matrix})
    grd_descriptor = grd_global_matrix
    sat_descriptor = sat_global_matrix
    
    

    data_amount = grd_descriptor.shape[0]
    print('      data_amount %d' % data_amount)
    top1_percent = int(data_amount * 0.01) + 1
    print('      top1_percent %d' % top1_percent)

    if test_grd_noise == 0:  
        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel])

        dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
        #dist_array = 2 - 2 * np.matmul(grd_descriptor, sat_descriptor.transpose())
 
        val_accuracy = validate(dist_array, 1)
        print('accuracy = %.1f%%' % (val_accuracy * 100.0))

        gt_dist = dist_array.diagonal()
        prediction = np.sum(dist_array < gt_dist.reshape(-1, 1), axis=-1)
        loc_acc = np.sum(prediction.reshape(-1, 1) < np.arange(top1_percent), axis=0) / data_amount
        scio.savemat(file, {'loc_acc': loc_acc, 'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor}) 
   
if __name__ == '__main__':
    train(start_epoch)
