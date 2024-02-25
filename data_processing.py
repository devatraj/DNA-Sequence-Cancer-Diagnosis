# -*- coding: utf-8 -*-
"""
@author: Devatraj
"""

import tensorflow as tf
import numpy as np

def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            items = line.split('\t')
            sequences.append(items[2])
            labels.append(int(items[1]))

    return sequences, labels

def load_dataset(data_path):
    One_hot = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'C': [0, 0, 0, 1]}
    NCP = {'A': [1, 1, 1],
           'T': [0, 1, 0],
           'G': [1, 0, 0],
           'C': [0, 0, 1]}
    DPCP = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0,
                   0.76847598772376],
            'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811,
                   0.32686549630327577],
            'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332],
            'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172,
                   0.8400043540595654],
            'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868,
                   0.32686549630327577],
            'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332]}

    sequences, label_list = load_tsv_format_data(data_path)

    One_hot_matrix_list, NCP_matrix_list, DPCP_matrix_list = [], [], []

    for sequence_cur in sequences:
        if len(sequence_cur) < 41:
            print('The input sequence ''%s'' does not meet the minimum length requirement of 40.' % sequence_cur)
        else:
            One_hot_matrix_middle = np.zeros([4, 38])
            NCP_matrix_middle = np.zeros([3, 38])
            DPCP_matrix_middle = np.zeros([6, 38])
            for pos in range(38):
                One_hot_matrix_middle[0:4, pos] += np.asarray(np.float32(One_hot[sequence_cur[pos + 1]]))
                NCP_matrix_middle[0:3, pos] += np.asarray(np.float32(NCP[sequence_cur[pos + 1]]))
                DPCP_matrix_middle[0:6, pos] += np.asarray(np.float32(DPCP[sequence_cur[pos:pos + 2]]))

            for left in range(len(sequence_cur) - 40):
                right = left + 40

                One_hot_matrix_left = np.asarray(np.float32(One_hot[sequence_cur[left]]))
                One_hot_matrix_right1 = np.asarray(np.float32(One_hot[sequence_cur[right - 1]]))
                One_hot_matrix_right2 = np.asarray(np.float32(One_hot[sequence_cur[right]]))
                One_hot_matrix_right = np.concatenate((One_hot_matrix_right1[:, np.newaxis], One_hot_matrix_right2[:, np.newaxis]), axis=1)
                One_hot_matrix_cur = np.concatenate((One_hot_matrix_left[:, np.newaxis], One_hot_matrix_middle, One_hot_matrix_right), axis=1)

                NCP_matrix_left = np.asarray(np.float32(NCP[sequence_cur[left]]))
                NCP_matrix_right1 = np.asarray(np.float32(NCP[sequence_cur[right - 1]]))
                NCP_matrix_right2 = np.asarray(np.float32(NCP[sequence_cur[right]]))
                NCP_matrix_right = np.concatenate((NCP_matrix_right1[:, np.newaxis], NCP_matrix_right2[:, np.newaxis]), axis=1)
                NCP_matrix_cur = np.concatenate((NCP_matrix_left[:, np.newaxis], NCP_matrix_middle, NCP_matrix_right), axis=1)

                DPCP_matrix_left = np.asarray(np.float32(DPCP[sequence_cur[left:left + 2]]))
                DPCP_matrix_right1 =  np.asarray(np.float32(DPCP[sequence_cur[right - 2:right]]))
                DPCP_matrix_right2 = np.asarray(np.float32(DPCP[sequence_cur[right - 1:right + 1]]))
                DPCP_matrix_right = np.concatenate((DPCP_matrix_right1[:, np.newaxis], DPCP_matrix_right2[:, np.newaxis]), axis=1)
                DPCP_matrix_cur = np.concatenate((DPCP_matrix_left[:, np.newaxis], DPCP_matrix_middle, DPCP_matrix_right), axis=1)

                One_hot_matrix_list.append(One_hot_matrix_cur)
                NCP_matrix_list.append(NCP_matrix_cur)
                DPCP_matrix_list.append(DPCP_matrix_cur)

    One_hot_matrix_input = np.asarray([i for i in One_hot_matrix_list], dtype=np.float32)
    NCP_matrix_input = np.asarray([i for i in NCP_matrix_list], dtype=np.float32)
    DPCP_matrix_input = np.asarray([i for i in DPCP_matrix_list], dtype=np.float32)
    label_list_input = np.asarray([i for i in label_list], dtype=np.int64)
    all_matrix_input = np.concatenate((One_hot_matrix_input, NCP_matrix_input, DPCP_matrix_input), axis=1)
    all_matrix_input= tf.transpose(all_matrix_input,perm=[0, 2, 1])
    return sequences, label_list_input, One_hot_matrix_input, NCP_matrix_input, DPCP_matrix_input, all_matrix_input


#path= r"C:\Users\Devatraj\MLproj\ipynb\DNA-Sequence-Cancer-Diagnosis\dataset-tsv\train.tsv"
#sequences, labels = load_tsv_format_data(path)
#sequence_list, label_list, One_hot_matrix_input, NCP_matrix_input, DPCP_matrix_input, all_list_input = load_dataset(path)
#print(label_list.shape)
#print(One_hot_matrix_input.shape)
#print(NCP_matrix_input.shape)
#print(DPCP_matrix_input.shape)
#print(all_list_input.shape)