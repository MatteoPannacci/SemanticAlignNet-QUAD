from email.mime import base
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from PIL import Image


# Network for Satellite images
class VGGModelCir:

    def __init__(self, input_shape, name = '', out_channels = 8, freeze = True):
        self.input_shape = input_shape
        self.model = None
        self.name = name
        self.out_channels = out_channels
        self.freeze = freeze
        self.build_model(input_shape)


    def warp_pad_columns(self, x, n=1):
        out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)
        return tf.pad(tensor=out, paddings=[[0, 0], [n, n], [0, 0], [0, 0]])


    def build_model(self, input_shape):
        # Load the VGG16 model without the top (dense) layers
        base_model = VGG16(weights='imagenet', include_top=False)

        # Get the output of each layer and process it before passing it to the next layer
        x = input_shape
        for i,layer in enumerate(base_model.layers):  
            layer._name = layer.name + self.name
            
            if i <= 9 and self.freeze:
                layer.trainable = False
            if i == 0:
                continue

            x = layer(x)
            if x.name == "block4_conv1_sat/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv2_sat/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv3_sat/Relu:0":
                x = Dropout(0.2)(x)
            if i >= len(base_model.layers) - 6:  # Skip the last three convolutional layers
                break      
        
        # Add three additional convolutional layers
        # Circular convolution only on last three layers
        x = self.warp_pad_columns(x,1)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid',strides=(2,1))(x)
        x = self.warp_pad_columns(x,1)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid',strides=(2,1))(x)
        x = self.warp_pad_columns(x,1)
        x = Conv2D(self.out_channels, (3, 3), activation='relu', padding='valid',strides=(1,1))(x)

        # Create the modified VGG16 model
        self.model = Model(inputs=input_shape, outputs=x)


    def call(self, input):
        return self.model(input)
    

    def summary(self):
        self.model.summary()
        

# Network for the Ground images
class VGGModel:

    def __init__(self, input_shape, name = '', out_channels = 16, freeze = True):
        self.input_shape = input_shape
        self.model = None
        self.name = name
        self.out_channels = out_channels
        self.freeze = freeze
        self.build_model(input_shape)


    def warp_pad_columns(self, x, n=1):
        out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)
        return tf.pad(tensor=out, paddings=[[0, 0], [n, n], [0, 0], [0, 0]])


    def build_model(self, input_shape):
        # Load the VGG16 model without the top (dense) layers
        base_model = VGG16(weights='imagenet', include_top=False)

        # Get the output of each layer and process it before passing it to the next layer
        x = input_shape
        for i,layer in enumerate(base_model.layers):
            layer._name = layer.name + self.name
            
            if i <= 9 and self.freeze:
                layer.trainable = False
            if i == 0:
                continue
            
            x = layer(x)
            if x.name == "block4_conv1/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv2/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv3/Relu:0":
                x = Dropout(0.2)(x)
            if i >= len(base_model.layers) - 6:  # Skip the last three convolutional layers
                break    
        
        # Add three additional convolutional layers
        x = Conv2D(256, (3, 3), activation='relu', padding='same',strides=(2,1))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same',strides=(2,1))(x)
        x = Conv2D(self.out_channels, (3, 3), activation='relu', padding='same',strides=(1,1))(x)

        # Create the modified VGG16 model
        self.model = Model(inputs=input_shape, outputs=x)


    def call(self, input):
        return self.model(input)


    def summary(self):
        self.model.summary()
