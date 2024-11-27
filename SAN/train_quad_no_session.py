import os
from pickletools import optimize
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cir_net_FOV_mb import *
# Import its own InputData
from polar_input_data_quad import InputData
from VGG_no_session import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import argparse
from PIL import Image
print(keras.__version__)
tf.compat.v1.enable_eager_execution()
parser = argparse.ArgumentParser(description='TensorFlow implementation.')


# Parser
parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=30)
parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)
parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--name', type=str, default='unnamed')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--acc_size', type=int, default=4)
parser.add_argument('--all_rgb', type=bool, default=False)
args = parser.parse_args()


# Data Parameters
train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise
train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV
all_rgb_images = args.all_rgb

# Model Parameters
model_save_name = args.name
combination_type = 'concat' # concat, sum
grd_c = 12
grdseg_c = 4
sat_c = 12
satseg_c = 4

# Training Parameters
start_epoch = args.start_epoch
number_of_epoch = args.number_of_epoch
batch_size = args.batch_size
accumulation_size = args.acc_size
loss_type  = 'triplet' # (unused)
loss_weight = 10.0
optimizer_type = 'adam' # adam, adamw
learning_rate_val = 1e-4
weight_decay = 0.004



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



def compute_loss_triplet(dist_array):

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



def compute_loss_InfoNCE(sat_descriptor, grd_descriptor, logit_scale = 10.0):
    
    logits_sat_descriptor = logit_scale*sat_descriptor@grd_descriptor.T
    labels = tf.range(tf.shape(logits_sat_descriptor)[0], dtype=tf.int64) #check the [0], could be wrong
    loss_g2s = keras.ops.categorical_crossentropy(labels, logits_sat_descriptor, from_logits = True)

    logits_grd_descriptor = logits_sat_descriptor.T
    loss_s2g = keras.ops.categorical_crossentropy(labels, logits_grd_descriptor, from_logits = True)

    loss = (loss_g2s + loss_s2g) / 2.0
    return loss



def train(start_epoch=0):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 0.
    '''

    width = int(train_grd_FOV / 360 * 512)
    
    # import data
    input_data = InputData(all_rgb = all_rgb_images)
    processor = ProcessFeatures()

    # choose loss function
    if loss_type == 'triplet':
        compute_loss = compute_loss_triplet
    else:
        raise Exception("Loss not implemented: %s" % loss_type)    

    # Define the optimizer
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_val
        )
    elif optimizer_type == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate_val,
            weight_decay=weight_decay
        )
    else:
        raise Exception("Optimizer not implemented: %s" % optimizer_type)

    
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
        raise Exception("Combination method not implemented: %s" % combination_type)  
    

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
    if start_epoch != 0:
        model_path = "./saved_models/QUAD-12-4_post_correction/18"
        model = keras.models.load_model(model_path)
        print("Model checkpoint uploaded")


    ### TRAINING - Iterate over the desired number of epochs ###
    for epoch in range(start_epoch, start_epoch + number_of_epoch):
        print(f"Epoch {epoch+1}/{start_epoch + number_of_epoch}")
        
        iter = 0
        end = False
        finalEpochLoss = 0
        while True:
            total_loss = 0
            
            # gradient accumulation (batch=8, 4 iterations => total batch=32)
            for i in range(accumulation_size):

                # take next batch
                batch_sat_polar, batch_sat, batch_grd, batch_satseg, batch_grdseg, batch_orien = input_data.next_pair_batch(
                    batch_size, 
                    grd_noise=train_grd_noise, 
                    FOV=train_grd_FOV
                )

                if batch_sat is None:
                    end = True
                    break

                with tf.GradientTape() as tape:

                    # Forward pass through the model
                    grd_features, grdseg_features, sat_features, satseg_features = model([batch_grd, batch_grdseg, batch_sat_polar, batch_satseg])
                    
                    # feature extraction and concatenation
                    if combination_type == 'concat':
                        grd_features = tf.concat([grd_features, grdseg_features], axis=-1)                        
                        sat_features = tf.concat([sat_features, satseg_features], axis=-1)                        
                    elif combination_type == 'sum':
                        grd_features = tf.concat([tf.add(grd_features[:, :, :, :grdseg_c], grdseg_features), grd_features[:, :, :, grdseg_c:]], -1)
                        sat_features = tf.concat([tf.add(sat_features[:, :, :, :satseg_c], satseg_features), sat_features[:, :, :, satseg_c:]], -1)                        
                    
                    grd_features = tf.nn.l2_normalize(grd_features, axis=[1, 2, 3])
                    # sat_features is normalized after cropping
                    
                    # Compute correlation and distance matrix
                    sat_matrix, grd_matrix, distance, orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)

                    # Compute the loss
                    loss_value = compute_loss(distance)
                    total_loss += loss_value 
                    
                # Compute the gradients
                gradients = tape.gradient(loss_value, model.trainable_variables)
                if i == 0:
                        accumulated_gradients = gradients
                else:
                        accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]

            # at the end of the accumulation normalize the gradient and update the model's weights
            gradients = [acum_grad / tf.cast(accumulation_size, tf.float32) for acum_grad in accumulated_gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if iter % 25 == 0:
                print("ITERATION: {}, LOSS VALUE: {}, TOTAL LOSS: {}".format(iter, loss_value.numpy(), total_loss/accumulation_size))

            iter+=1

            if end:
                 break
    
        # Save the model
        model_path = "./saved_models/" + model_save_name + "/"+str(epoch)+"/"
        model.save(model_path)

        print("Validation...")
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

        # at the end of accumulation reshape the feature maps into vectors (and normalize sat)
        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel]) 
        # the grd_descriptor is already normalized

        data_amount = grd_descriptor.shape[0]
        print('      data_amount %d' % data_amount)
        top1_percent = int(data_amount * 0.01) + 1
        print('      top1_percent %d' % top1_percent)

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
        val_loss = compute_loss(dist_array) / data_amount
        print('loss (validation) = ' + format(val_loss.numpy(), '.8f'))

        # save model
        with open('./saved_models/' + model_save_name + '/train.txt', 'a') as file:
                file.write('Epoch ' + str(epoch) + 
                           ', top1 ' + format(val_top1, '.4f') +
                           ', top5 ' + format(val_top5, '.4f') +
                           ', top10 ' + format(val_top10, '.4f') +
                           ', top1perc ' + format(val_top1perc, '.4f') +
                           ', Loss (last train batch) ' + format(loss_value.numpy(), '.8f') + 
                           ', Loss (validation) ' + format(val_loss.numpy(), '.8f') + 
                           '\n')
                            


if __name__ == '__main__':
    train(start_epoch)
