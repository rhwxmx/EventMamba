######author: Hongwei Ren#######
######This script is used to generate the h5 file for the UCF101-DVS and HMDB51-DVS dataset, which is splited by sliding window######
import scipy.io as sio
import numpy as np
import sys
import os
import numpy as np
import uti as uti
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

NUM_CLASSES = 101
DATA_PATH = ""
WINDOW_SIZE = 1
STEP_SIZE = 0.5
SEQ_LEN = 1
NUM_POINTS = 2048
SAVE_PATH = BASE_DIR

class_index=[]
for root,dirs,files in os.walk(DATA_PATH):
    for dirs_son in dirs:
        class_index.append(dirs_son)

print(class_index)

EXPORT_PATH = uti.get_export_path(SAVE_PATH, NUM_CLASSES, WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS)

print('Data will save to', EXPORT_PATH)

data= []
label = []

for index,class_label in enumerate(class_index):
    for root,dirs,files in os.walk(DATA_PATH+"\\"+class_label):
        for file in files:
            file_path = os.path.join(root,file)
            print(file_path)
            data_ucf = sio.loadmat(file_path)
            T = data_ucf['ts'].reshape(-1)
            X = data_ucf['x'].reshape(-1)
            Y = data_ucf['y'].reshape(-1)
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
                        else:
                            continue

data = np.array(data)
label = np.array(label)
idx_out = np.arange(data.shape[0])
np.random.shuffle(idx_out)
data = data[idx_out]
label = label[idx_out]
print(data.shape)
print(label.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

data = X_train
label = y_train
print(data.shape)
print(label.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:

    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label

data = X_test
label = y_test
print(data.shape)
print(label.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:

    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label