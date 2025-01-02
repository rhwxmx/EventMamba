######author: Hongwei Ren#######
######This script is used to generate the h5 file for the IJRR dataset######  
import sys
import os
import h5py
import numpy as np
import csv
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

NUM_POINTS =1024
EXPORT_PATH = BASE_DIR+"/shape_tran_1024"
event_path = r'D:/dataset/evo/shapes_translation/events.txt'
gt_path = r'D:/dataset/evo/shapes_translation/groundtruth.txt'
frame_path = r'D:/dataset/evo/shapes_translation/images.txt'

if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)
    print(f"Path {EXPORT_PATH} created.")
else:
    print(f"Path {EXPORT_PATH} already exists.")

##### Read the data from the txt file
event = list(open(event_path, 'r'))
gt = list(open(gt_path, 'r'))
frame = list(open(frame_path, 'r'))

##### Read the GROUND TRUTH #####
gt_qua = []
for gt_son in  tqdm(gt, desc="Reading Ground Truth"):
    gt_son = gt_son.rstrip('\n')
    gt_son = gt_son.split(' ')
    gt_son = [float(i) for i in gt_son]
    gt_qua.append(gt_son)

##### convert the quaternion to euler angle #####
gt_euler = []
for i in tqdm(range(len(gt_qua)-1), desc="Interpolating Ground Truth"):
    gt_euler.append([])
    for j in range(0,4):
        gt_euler[-1].append(gt_qua[i][j])
    rotation = Rotation.from_quat(gt_qua[i][4:8])
    angle = rotation.as_euler('xyz', degrees=False)
    for j in range(len(angle)):
        gt_euler[-1].append(angle[j])

    ##### Interpolation: assuming that the camera is moving at a constant speed between two ground truth#####
    clip = 10
    rotation_previous = Rotation.from_quat(gt_qua[i][4:8])
    angle_previous = rotation.as_euler('xyz', degrees=False)
    rotation_pos = Rotation.from_quat(gt_qua[i+1][4:8])
    angle_pos = rotation_pos.as_euler('xyz', degrees=False)

    for n in range(1,clip):
        gt_euler.append([])
        ##### Interpolation for the position #####
        for j in range(0,4):
            gt_euler[-1].append((gt_qua[i+1][j]-gt_qua[i][j])*n/clip + gt_qua[i][j])
        ##### Interpolation for the rotation #####
        for j in range(len(angle)):
            gt_euler[-1].append((angle_pos[j]-angle_previous[j])*n/clip + angle_previous[j])

##### Read the event data #####
list_in_event =[]
for son in tqdm(event, desc="Reading Events"):
    son = son.rstrip('\n')
    son = son.split(' ')
    son = [float(i) for i in son]
    list_in_event.append(son)


##### Find a sufficient number of events (1024) between interpolated ground truth #####
list_gap_gt=[]
list_gap_event=[[]]
index = 0
for i in tqdm(range(len(gt_euler)-1), desc="Finding Events"):
    interval=[]
    if len(list_gap_event[-1]) > NUM_POINTS:
        list_gap_event.append([])
    flag = 0
    ##### Find the events between two ground truth #####
    for m in range(index,len(list_in_event)):
        ##### if the timestamp of the event is between two ground truth, then add it to list_gap_event#####
        if (list_in_event[m][0] > gt_euler[i][0]) and (list_in_event[m][0] < gt_euler[i+1][0]):
                list_gap_event[-1].append(list_in_event[m][:])
        ##### if the timestamp of the event is larger than the timestamp of the next ground truth, then break and preserve the index#####
        if list_in_event[m][0] > gt_euler[i+1][0]:
            index = m
            break
    ##### if the number of events between two ground truth is larger than 2048, then add the label to the list_gap_gt#####
    if len(list_gap_event[-1]) > NUM_POINTS:        
        for j in range(1,7):
            interval.append(gt_euler[i+1][j])
        list_gap_gt.append(interval)

##### Randomly select NUMPOINTs events from the list_gap_event#####
list_gap_event_sample = []
list_gap_label=[]
list_num_event = []
for i in tqdm(range(len(list_gap_event)), desc="Randomly sample Events"):
    size = len(list_gap_event[i])
    data = np.array(list_gap_event[i])
    idx = np.arange(size)
    np.random.shuffle(idx)
    idx = idx[0:NUM_POINTS]
    ##### Sort the events according to the timestamp#####
    idx.sort()
    sample_data = data[idx,...]
    list_num_event.append(len(data))
    if len(data) >= NUM_POINTS:
        list_gap_event_sample.append(data[idx,...])
        list_gap_label.append(list_gap_gt[i])

##### Normalize the data#####
list_gap_event_sample = np.array(list_gap_event_sample)
list_gap_label = np.array(list_gap_label)
list_gap_event_sample[:,:,0] = (list_gap_event_sample[:,:,0] -  np.expand_dims(list_gap_event_sample[:,:,0].min(axis=1),axis=1))
list_gap_event_sample[:,:,0] = list_gap_event_sample[:,:,0]/np.expand_dims(list_gap_event_sample[:,:,0].max(axis=1),axis=1)
list_gap_event_sample[:,:,1] = list_gap_event_sample[:,:,1]/240
list_gap_event_sample[:,:,2] = list_gap_event_sample[:,:,2]/180

X_train, X_test, y_train, y_test = train_test_split(list_gap_event_sample, list_gap_label, test_size=0.3,shuffle=True)

data = X_train
label = y_train
print(data.shape)
print(label.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None,6), chunks=True, dtype='float32')
    hf['data'][:] = data
    hf['label'][:] = label

data = X_test
label = y_test
print(data.shape)
print(label.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None,6), chunks=True, dtype='float32')
    hf['data'][:] = data
    hf['label'][:] = label