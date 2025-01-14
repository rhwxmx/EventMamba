######author: Hongwei Ren#######
######This script is used to generate the h5 file for the thu-challenge dataset######
import scipy.io as sio
import numpy as np
import sys
import os
import numpy as np
import  uti
import h5py
import open3d as o3d
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def load_npy_file(file_path):
    data = np.load(file_path)
    return data

def read_file(file_list):
    data= []
    label = []
    mark = []
    sample_count = 0
    for file in tqdm(file_list, desc="Processing Files"):
        file_path = file[0]
        category = file[1]
        data_npy = load_npy_file(file_path)
        T = data_npy[:,2].reshape(-1)
        X = data_npy[:,0].reshape(-1)
        Y = data_npy[:,1].reshape(-1)
        class_events = np.zeros(shape=(int(len(X)),3),dtype=np.int64)
        class_events[:,0] = T
        class_events[:,1] = X
        class_events[:,2] = Y
        if len(T) > 0:
            win_start_index,win_end_index = uti.get_window_index(T,T[0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
            NUM_WINDOWS = len(win_start_index)
            for n in range(NUM_WINDOWS):
                window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
                ###### Liqun's normalization to reduce the size of the event cloud ######
                
                # window_events = window_events.astype(float)
                # window_events[:,1]/=2
                # window_events[:,2]/=2
                # window_events = window_events.astype(int)
                # _,unique_x_y_combinations = np.unique(window_events, axis=0,return_index=True) 
                # window_events = window_events[unique_x_y_combinations]
                # window_events[:,1]*=2
                # window_events[:,2]*=2
                # window_events = window_events.astype(float)

                window_events_liqun = window_events.copy()
                window_events_liqun[:,0] = window_events_liqun[:,0]-window_events_liqun[:,0].min(axis=0)
                liqun_normed = window_events_liqun / window_events_liqun.max(axis=0)
                liqun_normed[:,1] = window_events_liqun[:,1] / 346
                liqun_normed[:,2] = window_events_liqun[:,2] / 260
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(liqun_normed)
                radius = 0.01
                min_neighbors = 5
                pcd, ind =pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
                window_events = window_events[ind]
                
                if window_events.shape[0] >= 100:
                    extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                    extracted_events[:,0] = extracted_events[:,0]-extracted_events[:,0].min(axis=0)
                    events_normed = extracted_events / extracted_events.max(axis=0)
                    events_normed[:,1] = extracted_events[:,1] / 346
                    events_normed[:,2] = extracted_events[:,2] / 260
                    data.append(events_normed)
                    label.append(category)
                    mark.append(sample_count)
                else:
                    continue
        sample_count += 1
    return data,label,mark

NUM_CLASSES = 50
DATA_PATH = "E:\\dataset\\THU_AC\\THU-EACT-50-CHL\\THU-EACT-50-CHL\\"
WINDOW_SIZE = 1
STEP_SIZE = 0.5
SEQ_LEN = 1
NUM_POINTS = 2048
SAVE_PATH = BASE_DIR
EXPORT_PATH = "E:\\dataset\\THU_AC\\THU-EACT-50-CHL\\"
print('Data will save to', EXPORT_PATH)

train_list = []
test_list = []
with open(DATA_PATH+'train.txt', 'r') as file:
    for line in file:
        file_path, category = line.strip().split(' ')
        file_path = file_path.replace('../DVS-action-data-npy/',DATA_PATH)
        file_path = file_path.replace('/','\\')
        train_list.append((file_path, int(category)))

with open(DATA_PATH+'test.txt', 'r') as file:
    for line in file:
        file_path, category = line.strip().split(' ')
        file_path = file_path.replace('../DVS-action-data-npy/',DATA_PATH)
        file_path = file_path.replace('/','\\')
        test_list.append((file_path, int(category)))

data_test,label_test,mark_test = read_file(test_list)
print(len(data_test))
data_train,label_train,mark_train = read_file(train_list)

data_train = np.array(data_train)
label_train = np.array(label_train)
data_test = np.array(data_test)
label_test = np.array(label_test)
mark_train = np.array(mark_train)
mark_test = np.array(mark_test)

data = data_train
label = label_train
mark = mark_train
print(data.shape)
print(label.shape)
print(mark.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label
    hf['mark'][:] = mark

data = data_test
label = label_test
mark = mark_test
print(data.shape)
print(label.shape)
print(mark.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label
    hf['mark'][:] = mark