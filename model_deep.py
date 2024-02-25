# -*- coding: utf-8 -*-
"""
@author: Devatraj
"""

from keras import layers

def my_model(input_tensor, input_channels):
    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', kernel_initializer = 'glorot_normal', input_shape=(None, input_channels))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    y = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same',kernel_initializer = 'glorot_normal')(x)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    int_norm1 = layers.Add()([x,y])
    #     x = layers.MaxPooling1D(2, 2)(tf.nn.relu(x))
    #     x = layers.MaxPooling1D(2, 2)(tf.nn.relu(x))
    #     x = layers.Flatten()(x)
    #     x = layers.Dense(units=256, activation='relu')(x)
    #     x = layers.Dense(units=2)(x)
    kernels = range(2, 32, 4)
    nets = []
    for i in range(len(kernels)): 
        conv3 = layers.Conv1D(filters = 64, kernel_size = 8, padding = 'same', 
                              name='conv3_' + str(i), kernel_initializer = 'glorot_normal')(int_norm1)
        norm3 = layers.BatchNormalization(name='norm3_' + str(i))(conv3)
        norm3 = layers.LeakyReLU(alpha=0.2)(norm3)

        int_norm2 = layers.Add()([int_norm1, norm3])
 
        conv4 = layers.Conv1D(filters = 128, kernel_size = kernels[i], padding = 'valid', 
                              name='conv4_' + str(i), kernel_initializer = 'glorot_normal')(int_norm2)
        norm4 = layers.BatchNormalization(name='norm4_' + str(i))(conv4)
        norm4 = layers.LeakyReLU(alpha=0.2)(norm4)
 
        pool = layers.MaxPooling1D(2,2)(norm4)
        flat = layers.Flatten(name='flat_' + str(i))(pool)
        nets.append(flat)
    net = layers.Concatenate(axis=1)(nets)
    for i in range(0):
        net = layers.Dense(2, activation='relu', name='dense_' + str(i))(net)
 
    net = layers.Dropout(0.4)(net)
    net = layers.Dense(2, activation='sigmoid', name='dense_out_6')(net)
 
    return net

#input_tensor = tf.keras.Input(shape=(41,13))
#output_tensor = my_model(input_tensor,13)
#model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

#tmp = tf.random.normal((10, 41, 13))
#out = model(tmp)
#print('MyModel:', out.shape)