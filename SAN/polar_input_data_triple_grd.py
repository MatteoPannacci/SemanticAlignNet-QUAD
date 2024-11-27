
from calendar import c
import cv2
import random
import numpy as np
#from cir_net_FOV import *
from PIL import Image


class InputData:

    img_root = '../Data/CVUSA_subset/' # USE AUGMENTED DATASET

    # Mean and standard deviation of the used subset
    ground_mean = np.array([[0.46,0.48,0.47]])
    ground_std = np.array([[0.24,0.20,0.21]])
    grdseg_mean = np.array([0.30, 0.70, 0.36])
    grdseg_std = np.array([0.28, 0.33, 0.43])
    sat_polar_mean = np.array([[0.36,0.41,0.40]])
    sat_polar_std = np.array([[0.15,0.14,0.15]])
    flipped = 0


    def __init__(self, data_type='CVUSA_subset', all_rgb=False):

        self.data_type = data_type
        self.all_rgb = all_rgb
        self.img_root = '../Data/' + self.data_type + '/'

        ### substituted with augmented list
        self.train_list = self.img_root + 'train_updated.csv'
        self.test_list = self.img_root + 'val_updated.csv'
        ###

        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            next(file) # remove header
            idx = 0            
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([
                    data[3], # satellite polar
                    data[0], # satellite
                    data[1], # ground
                    pano_id
                ])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)


        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            next(file) # remove header
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([
                    data[3], # satellite polar
                    data[0], # satellite
                    data[1], # ground
                    pano_id
                ])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)


    def next_batch_scan(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None, None, None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        grd_width = int(FOV/360*512)

        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype = np.float32)
        batch_grdseg = np.zeros([batch_size, 128, grd_width, 3], dtype = np.float32)
        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)      
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        grd_shift = np.zeros([batch_size], dtype=np.int_)

        for i in range(batch_size):

            img_idx = self.__cur_test_id + i

            # Satellite RGB Polar
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = img.astype(np.float32)
            img = img/255
            img = (img - self.sat_polar_mean) / self.sat_polar_std
            batch_sat_polar[i, :, :, :] = img
            ###

            # Satellite RGB (not used)
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img = img/255
            batch_sat[i, :, :, :] = img
            ###

            # Ground RGB
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][2])
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img = img/255
            img = (img - self.ground_mean) / self.ground_std
            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            grd_shift[i] = random_shift
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]
            batch_grd[i, :, :, :] = img_dup
            ###

            # Ground SEG
            if self.all_rgb:
                img = cv2.imread(self.img_root + self.id_test_list[img_idx][2])
            else:
                img = cv2.imread(self.img_root + self.id_test_list[img_idx][2].replace("streetview","streetview_segmentation"))
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img = img/255
            img = (img - self.grdseg_mean) / self.grdseg_std
            j = np.arange(0, 512)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :] # USE SAME SHIFT
            batch_grdseg[i, :, :, :] = img_dup
            ###

        self.__cur_test_id += batch_size

        batch_orientation = (np.around(((512-grd_shift)/512*64)%64)).astype(np.int_)

        return batch_sat_polar, batch_sat, batch_grd, batch_grdseg, batch_orientation


    def next_pair_batch(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None, None, None, None

        grd_width = int(FOV/360*512)

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        batch_grdseg = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        grd_shift = np.zeros([batch_size,], dtype=np.int_)

        i = 0
        batch_idx = 0
        count = 0
        while True:
            
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # SATELLITE POLAR TRANSFORMED
            img = cv2.imread(self.img_root + self.id_list[img_idx][0])
            img = img.astype(np.float32)
            img = img/255
            img = (img - self.sat_polar_mean) / self.sat_polar_std
            batch_sat_polar[batch_idx, :, :, :] = img
            #######################################

            # SATELLITE IMAGE (unused)
            img = cv2.imread(self.img_root + self.id_list[img_idx][1])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img = img/255
            batch_sat[batch_idx, :, :, :] = img
            #######################################

            # GROUND IMAGE
            img = cv2.imread(self.img_root + self.id_list[img_idx][2])
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img = img/255
            img = (img - self.ground_mean) / self.ground_std
            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            grd_shift[batch_idx] = random_shift
            img_dup = img[:, ((j-random_shift)%512)[:grd_width], :]
            batch_grd[batch_idx, :, :, :] = img_dup
            #######################################

            ### GROUND SEGMENTATION
            if self.all_rgb:
                img = cv2.imread(self.img_root + self.id_list[img_idx][2])
            else:
                img = cv2.imread(self.img_root + self.id_list[img_idx][2].replace("streetview","streetview_segmentation"))
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img = img/255
            img = (img - self.grdseg_mean) / self.grdseg_std
            j = np.arange(0, 512)
            img_dup = img[:, ((j-random_shift)%512)[:grd_width], :] # use same shift
            batch_grdseg[batch_idx, :, :, :] = img_dup
            #######################################

            # check size
            #if img is None or img.shape[0] != 512 or img.shape[1] != 128:
            #    print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i),
            #          img.shape)
            #    continue

            batch_idx += 1
            count+=1

        self.__cur_id += i

        batch_orientation = (np.around(((512-grd_shift)/512*64)%64)).astype(np.int_)

        return batch_sat_polar, batch_sat, batch_grd, batch_grdseg, batch_orientation


    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0
