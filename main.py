# -*- coding: utf-8 -*-
"""
@author: Devatraj
"""

import tensorflow as tf
import random
import numpy as np
from keras.initializers import glorot_uniform
from data_processing import load_dataset
from model_deep import my_model

def initialize(layer):
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        layer.kernel_initializer = glorot_uniform()
        if layer.bias is not None:
            layer.bias_initializer = tf.keras.initializers.Zeros()

def obtain_random(length):
    list_info = []
    while True:
        info = random.randint(0,length-1)
        if info not in list_info:
            list_info.append(info)
        if len(list_info) ==length:
            break

    return list_info

def cat_batch(input_list, info_list, idx, batch_size, batch_num):
    if idx == (batch_num - 1):
        batch = info_list[idx * batch_size:]
    else:
        batch = info_list[idx * batch_size: (idx + 1) * batch_size]

    # Initialize an empty tensor with the same shape as the tensors in input_list
    # catbatch = input_list[batch[1]]
    for idx, x in enumerate(batch):
        if idx==0:
          catbatch = tf.expand_dims(input_list[x], axis=0)
        else:
          x_tensor = tf.expand_dims(input_list[x], axis=0)
          catbatch = tf.concat([catbatch, x_tensor], axis=0)

    return catbatch

train_data_path = r"C:\Users\Devatraj\MLproj\ipynb\DNA-Sequence-Cancer-Diagnosis\Dataset-tsv\train.tsv"
train_sequence_list, train_label_list, train_One_hot_matrix_input, train_NCP_matrix_input, train_DPCP_matrix_input, train_all_matrix_input = load_dataset(train_data_path)
train_list_info = obtain_random(len(train_sequence_list))

val_data_path = r"C:\Users\Devatraj\MLproj\ipynb\DNA-Sequence-Cancer-Diagnosis\Dataset-tsv\test.tsv"
val_sequence_list, val_label_list,  val_One_hot_matrix_input, val_NCP_matrix_input, val_DPCP_matrix_input, val_all_matrix_input = load_dataset(val_data_path)
val_list_info = obtain_random(len(val_sequence_list))

input_tensor = tf.keras.Input(shape=(41, 13))
output_tensor = my_model(input_tensor,13)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

initialize(model)

# if len(physical_devices) > 0:
#     model = tf.keras.utils.multi_gpu_model(model, gpus=len(physical_devices))

class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_factor, decay_steps):
        self.initial_learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps

    def __call__(self, step):
        val = tf.dtypes.cast(step // self.decay_steps, tf.float32)
        return self.initial_learning_rate * tf.pow(tf.constant(self.decay_factor, dtype=tf.float32), val)

lr1=0.01
lr_schedule = CustomLearningRateSchedule(initial_learning_rate=lr1, decay_factor=0.7, decay_steps=4)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_schedule,
    momentum=0.9,
    nesterov=False,
    weight_decay=0.01,
    name='SGD')


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 16
train_patience = 10
best_acc, patience = None, 0

for epoch in range(100):
    if patience == train_patience:
        print(f"val_loss did not improve after {train_patience} Epochs, thus Earlystopping is calling")
        break

    cnt, loss_sum = 0, 0
    val = epoch // 4
    lr = 1e-2 * pow(0.7, val)
    if lr < 1e-5:
        lr = 1e-5

    print(lr)

    batch_num = np.ceil(len(train_list_info) / batch_size).astype(int)
    for idx in range(batch_num):
        with tf.GradientTape() as tape:
            x = cat_batch(train_all_matrix_input, train_list_info, idx, batch_size, batch_num)
            label = cat_batch(train_label_list, train_list_info, idx, batch_size, batch_num)
            label = tf.one_hot(label, depth=2)
            logits = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label, logits))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_sum += loss.numpy()
        cnt += 1

    final_loss = loss_sum / cnt
    print(f"Epoch: {epoch + 1}, Train_Loss: {final_loss}")

    if epoch % 1 == 0:
        cnt, total_correct = 0, 0
        batch_num = np.ceil(len(val_list_info) / batch_size).astype(int)
        for idx in range(batch_num):
            x = cat_batch(val_all_matrix_input, val_list_info, idx, batch_size, batch_num)
            label = cat_batch(val_label_list, val_list_info, idx, batch_size, batch_num)

            logits = model(x)
            pred = tf.argmax(logits, axis=1)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, label), dtype=tf.int32))
            total_correct += correct.numpy()
            cnt += x.shape[0]

        acc = total_correct / cnt

        if best_acc is None or acc > best_acc:
            best_acc, patience = acc, 0
            model.save_weights("1.tf")
        else:
            patience += 1

        print(f"Epoch: {epoch + 1}, Valid_acc: {acc}")
