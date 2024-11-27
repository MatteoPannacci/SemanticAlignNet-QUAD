import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from cir_net_FOV_mb import *
from polar_input_data_quad import InputData
from VGG_no_session import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import argparse
from PIL import Image
import scipy.io as scio
from numpy import fft
import os


tf.compat.v1.enable_eager_execution()
parser = argparse.ArgumentParser(description='TensorFlow implementation.')


# Parser
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--input_path', type=str, default='./saved_models/unnamed/1')
parser.add_argument('--output_path', type=str, default='./saved_models/unnamed/')
args = parser.parse_args()


# Data Parameters
test_grd_noise = args.test_grd_noise
test_grd_FOV = args.test_grd_FOV

# Model Parameters
input_path = args.input_path
output_path = args.output_path
combination_type = 'sum' # concat, sum
grd_c = 16
grdseg_c = 8
sat_c = 16
satseg_c = 8

# Test Parameters
batch_size = args.batch_size



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



def test():

    width = int(test_grd_FOV / 360 * 512)
    
    # import data
    input_data = InputData()
    processor = ProcessFeatures()                
    
    # Siamese-like network branches
    grdNet = VGGModel(tf.keras.Input(shape=(None, None, 3)),'_grd', out_channels=grd_c, freeze=True)
    grdSegNet = VGGModel(tf.keras.Input(shape=(None, None, 3)),'_grdseg', out_channels=grdseg_c, freeze=True)
    satNet = VGGModelCir(tf.keras.Input(shape=(None, None, 3)),'_sat', out_channels=sat_c, freeze=True)
    satSegNet = VGGModelCir(tf.keras.Input(shape=(None, None, 3)),'satseg', out_channels=satseg_c, freeze=True)
 
    # Full Model
    model = Model(
         inputs=[grdNet.model.input, grdSegNet.model.input, satNet.model.input, satSegNet.model.input], 
         outputs=[grdNet.model.output, grdSegNet.model.output, satNet.model.output, satSegNet.model.output]
    )
    
    print("Model created")

    # The two halves of the model should have the same number of channels in total
    # otherwise the output feature maps will be of different sizes
    if combination_type == 'concat':
        assert grdNet.out_channels + grdSegNet.out_channels == satNet.out_channels + satSegNet.out_channels
    elif combination_type == 'sum':
        assert grdNet.out_channels == satNet.out_channels
    else:
        raise Exception("Combination method not implemented!")  

    # (empty) input tensors
    grd_x = np.float32(np.zeros([2, 128, width, 3]))
    grdseg_x = np.float32(np.zeros([2, 128, width, 3]))
    sat_x = np.float32(np.zeros([2, 256, 512, 3])) # not used
    polar_sat_x = np.float32(np.zeros([2, 128, 512, 3]))
    satseg_x = np.float32(np.zeros([2, 128, 512, 3])) # (it's polar)

    # feature extraction and concatenation
    grd_features, grdseg_features, sat_features, satseg_features = model([grd_x, grdseg_x, polar_sat_x, satseg_x])
    # feature extraction and concatenation
    if combination_type == 'concat':
        grd_features = tf.concat([grd_features, grdseg_features], axis=-1)                        
        sat_features = tf.concat([sat_features, satseg_features], axis=-1)                        
    elif combination_type == 'sum':
        grd_features = tf.concat([tf.add(grd_features[:, :, :, :grdseg_c], grdseg_features), grd_features[:, :, :, grdseg_c:]], -1)
        sat_features = tf.concat([tf.add(sat_features[:, :, :, :satseg_c], satseg_features), sat_features[:, :, :, satseg_c:]], -1)                        
    else:
        raise Exception("Combination method not implemented!")

    # computing the distance and the matching
    sat_matrix, grd_matrix, distance, pred_orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)

    # computing shapes
    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])

    # load a Model
    model_path = input_path
    model = keras.models.load_model(model_path)
    print("Model checkpoint uploaded")


    print("Test...")
    val_i = 0
    count = 0
    while True:
        # print('      progress %d' % val_i)
        
        # take next batch
        batch_sat_polar, batch_sat, batch_grd, batch_satseg, batch_grdseg, batch_orien  = input_data.next_batch_scan(
            batch_size, 
            grd_noise=test_grd_noise,
            FOV=test_grd_FOV
        )
        
        if batch_sat is None:
            break
        
        # Forward pass
        grd_features, grdseg_features, sat_features, satseg_features = model([batch_grd, batch_grdseg, batch_sat_polar, batch_satseg])
        
        # feature extraction and concatenation
        if combination_type == 'concat':
            grd_features = tf.concat([grd_features, grdseg_features], axis=-1)                        
            sat_features = tf.concat([sat_features, satseg_features], axis=-1)                        
        elif combination_type == 'sum':
            grd_features = tf.concat([tf.add(grd_features[:, :, :, :grdseg_c], grdseg_features), grd_features[:, :, :, grdseg_c:]], -1)
            sat_features = tf.concat([tf.add(sat_features[:, :, :, :satseg_c], satseg_features), sat_features[:, :, :, satseg_c:]], -1)                        
        else:
            raise Exception("Combination method not implemented!")

        grd_features = tf.nn.l2_normalize(grd_features, axis=[1, 2, 3])
        # sat_features is normalized after cropping
        
        # Compute correlation and distance matrix
        sat_matrix, grd_matrix, distance, orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)

        # accumulate the feature maps
        sat_global_matrix[val_i: val_i + sat_matrix.shape[0], :] = sat_matrix
        grd_global_matrix[val_i: val_i + grd_matrix.shape[0], :] = grd_matrix
        orientation_gth[val_i: val_i + grd_matrix.shape[0]] = batch_orien

        val_i += sat_matrix.shape[0]
        count += 1

    os.makedirs(output_path, exist_ok=True)
    
    # file = output_path + '/descriptors.mat'
    # scio.savemat(file, {'orientation_gth': orientation_gth, 'grd_descriptor': grd_global_matrix, 'sat_descriptor': sat_global_matrix})
    
    grd_descriptor = grd_global_matrix
    sat_descriptor = sat_global_matrix

    data_amount = grd_descriptor.shape[0]
    print('      data_amount %d' % data_amount)
    top1_percent = int(data_amount * 0.01) + 1
    print('      top1_percent %d' % top1_percent)

    if test_grd_noise == 0:  

        # at the end of accumulation reshape the feature maps into vectors (and normalize sat)
        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel]) 
        # the grd_descriptor is already normalized

        # compute distances
        dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
 
        # compute metrics
        val_top1 = validate(dist_array, 1)
        print('top1 = %.4f%%' % (val_top1 * 100.0))
        val_top5 = validate(dist_array, 5)
        print('top5 = %.4f%%' % (val_top5 * 100.0))
        val_top10 = validate(dist_array, 10)
        print('top10 = %.4f%%' % (val_top10 * 100.0))
        val_top1perc = validate(dist_array, top1_percent)
        print('top1perc = %.4f%%' % (val_top1perc * 100.0))

        # save model
        with open(output_path + '/test.txt', 'a') as file:
                file.write('Model Path ' + input_path +
                           ', Test FOV ' + str(test_grd_FOV) + 
                           ', top1 ' + format(val_top1, '.4f') +
                           ', top5 ' + format(val_top5, '.4f') +
                           ', top10 ' + format(val_top10, '.4f') +
                           ', top1perc ' + format(val_top1perc, '.4f') + 
                           '\n')
        
        # gt_dist = dist_array.diagonal()
        # prediction = np.sum(dist_array < gt_dist.reshape(-1, 1), axis=-1)
        # loc_acc = np.sum(prediction.reshape(-1, 1) < np.arange(top1_percent), axis=0) / data_amount
        # scio.savemat(file, {'loc_acc': loc_acc, 'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor})            



if __name__ == '__main__':
    test()
