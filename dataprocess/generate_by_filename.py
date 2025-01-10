######author: Hongwei Ren#######
######This script is used to generate the h5 file for the UCF101-DVS and HMDB51-DVS dataset, which is splited by filename######
import scipy.io as sio
import numpy as np
import sys
import os
import numpy as np
import uti
import h5py
import ast
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def read_file(file_list):
    data= []
    label = []
    mark = []
    sample_count = 0
    for file in tqdm(file_list, desc="Processing Files"):
        file_path = file[0]
        index = file[1]
        data_ucf = sio.loadmat(file_path)
        T = data_ucf['ts'].reshape(-1)
        X = data_ucf['x'].reshape(-1)
        Y = data_ucf['y'].reshape(-1)
        P = data_ucf['pol'].reshape(-1)
        class_events = np.zeros(shape=(int(len(X)),3),dtype=np.int64)
        class_events[:,0] = T
        class_events[:,1] = X
        class_events[:,2] = Y
        if len(T) > 0:
            win_start_index,win_end_index = uti.get_window_index(T,T[0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
            NUM_WINDOWS = len(win_start_index)
            if NUM_WINDOWS < 40:
                for n in range(NUM_WINDOWS):
                    window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
                    if window_events.shape[0] > 10000:
                        extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                        extracted_events[:,0] = extracted_events[:,0]-extracted_events[:,0].min(axis=0)
                        events_normed = extracted_events / extracted_events.max(axis=0)
                        events_normed[:,1] = extracted_events[:,1] / 320
                        events_normed[:,2] = extracted_events[:,2] / 240
                        data.append(events_normed)
                        label.append(index)
                        mark.append(sample_count)
                    else:
                        continue
        sample_count += 1
    return data,label,mark

NUM_CLASSES = 101
DATA_PATH = ""
EXPORT_PATH = ""
WINDOW_SIZE = 1
STEP_SIZE = 0.5
SEQ_LEN = 1
NUM_POINTS = 8192
SAVE_PATH = BASE_DIR
print('Data will save to', EXPORT_PATH)

file_train =[]
with open("./train.txt", "r") as txt_file:
    for line in txt_file:
        file_train.append(ast.literal_eval(line.strip()))
file_test =[]
with open("./test.txt", "r") as txt_file:
    for line in txt_file:
        file_test.append(ast.literal_eval(line.strip()))

data_test,label_test,mark_test = read_file(file_test)
data_train,label_train,mark_train = read_file(file_train)

data_test = np.array(data_test)
label_test = np.array(label_test)
mark_test = np.array(mark_test)
data_train = np.array(data_train)
label_train = np.array(label_train)
mark_train = np.array(mark_train)

print(data_train.shape)
print(label_train.shape)
print(mark_train.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data_train.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label_train.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark_train.shape, maxshape = (None), chunks=True, dtype='int32')
    hf['data'][:] = data_train
    hf['label'][:] = label_train
    hf['mark'][:] = mark_train

print(data_test.shape)
print(label_test.shape)
print(mark_test.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data_test.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label_test.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark_test.shape, maxshape = (None), chunks=True, dtype='int32')
    hf['data'][:] = data_test
    hf['label'][:] = label_test
    hf['mark'][:] = mark_test
