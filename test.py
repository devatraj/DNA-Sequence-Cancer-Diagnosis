# -*- coding: utf-8 -*-
"""
@author: Devatraj
"""
import os
import csv
import numpy as np
import tensorflow as tf
from model_deep import my_model
from data_processing import load_dataset
from main import obtain_random,cat_batch


def eff(labels, preds):

    TP, FN, FP, TN = 0, 0, 0, 0

    for idx,label in enumerate(labels):

        if label == 1:
            if label == preds[idx]:
                TP += 1
            else: FN += 1
        elif label == preds[idx]:
            TN += 1
        else: FP += 1

    return TP, FN, FP, TN

def Judeff(TP, FN, FP, TN):

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    # MCC = (TP * TN - FP * FN) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))

    return SN, SP, ACC

def Calauc(labels, preds):

    labels = tf.convert_to_tensor(labels)
    preds = tf.convert_to_tensor(preds)
    labels_np = labels.numpy()
    preds_np = preds.numpy()

    f = list(zip(preds, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    pos_cnt = np.sum(labels == 1)
    neg_cnt = np.sum(labels == 0)
    AUC = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)

    return AUC

folder_path = r"C:\Users\Devatraj\MLproj\ipynb\DNA-Sequence-Cancer-Diagnosis\Dataset-tsv"

dataset_files = [f for f in os.listdir(folder_path) if f.endswith(".tsv")]

for dataset_file in dataset_files:
    dataset_path = os.path.join(folder_path, dataset_file)
    test_sequence_list, test_label_list, test_One_hot_matrix_input, test_NCP_matrix_input, test_DPCP_matrix_input, test_all_matrix_input = load_dataset(dataset_path)
    test_list_info = obtain_random(len(test_label_list))
    batch_size = len(test_sequence_list)
    input_tensor = tf.keras.Input(shape=(41, 13))
    output_tensor = my_model(input_tensor,13)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights("1.tf")
    model.compile()
    TP, FN, FP, TN = 0, 0, 0, 0
    AUC = 0

    batch_num = int(np.ceil(len(test_list_info) / batch_size))

    for idx in range(batch_num):
        x = cat_batch(test_all_matrix_input, test_list_info, idx, batch_size, batch_num)
        label = cat_batch(test_label_list, test_list_info, idx, batch_size, batch_num)

        logits = model(x)
        pred = tf.argmax(logits, axis=1)

        A, B, C, D = eff(label, pred)
        TP += A
        FN += B
        FP += C
        TN += D
        AUC += Calauc(label, pred)

    SN, SP, ACC = Judeff(TP, FN, FP, TN)
    print(f"Results for dataset: {dataset_file}")
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("SN: {}, SP: {}, ACC: {}, AUC: {}".format(SN, SP, ACC, AUC / batch_num))
    modelname = 'medcnn'
    modelname = f'{dataset_file.split(".")[0]}_{modelname}'
    date = [modelname, TP, FN, FP, TN, SN, SP, ACC, AUC / batch_num]

    with open('rundate.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'TP', 'FN', 'FP', 'TN', 'SN', 'SP', 'ACC', 'AUC'])
        writer.writerow(date)