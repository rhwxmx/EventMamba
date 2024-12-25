######author: Hongwei Ren#######
######This script is used to generate the h5 file for the IBM gesture dataset######
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import uti 
import numpy as np
import h5py
import os
from tqdm import tqdm

#############read the data from the dataset#############
# NUM_POINTS is the number of events after downsample
# STEP_SIZE is the step size of the sliding window
# WINDOW_SIZE is the size of the sliding window
# train is the flag to indicate whether the data is used for training or testing
def read_file(dataset,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=True):
    data= []
    labels = []
    # marks is used to record the index of the data in the dataset, when we validate the model, we can use the index to find the corresponding data to calculate the accuracy_seq
    marks = []
    for i in tqdm(range(len(dataset))):
        events,label= dataset[i]
        # this is used to remove the label 10
        if label == 10: continue
        class_events = np.zeros(shape=(int(len(events['x'])),3),dtype=np.int64)
        class_events[:,0] = events['t']
        class_events[:,1] = events['x']
        class_events[:,2] = events['y']
        win_start_index,win_end_index = uti.get_window_index(events['t'],events['t'][0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
        NUM_WINDOWS = len(win_start_index)
        for n in range(NUM_WINDOWS):
            window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
            if window_events.shape[0] > 100:
                extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    # normalize the events, and False means we do not process the polarity
                    events_normed = uti.normaliztion(extracted_events,128,128,False)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
                if train:
                    # data augmentation
                    crop_events = uti.random_crop(window_events,128,128)
                    extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                    

    data = np.array(data)
    labels = np.array(labels)
    marks = np.array(marks)
    return data,labels,marks
# the path of the dataset
ROOT_DIR = 'D:\\dataset\\ibm-gesture\\dataset\\'
# the path to save the h5 file
EXPORT_PATH = 'D:\\dataset\\ibm-gesture\\'
NUM_POINTS = 1024
STEP_SIZE = 0.25
WINDOW_SIZE = 0.5
train_set = DVS128Gesture(ROOT_DIR, train=True)
test_set = DVS128Gesture(ROOT_DIR, train=False)

data_train,label_train,mark_train = read_file(train_set,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=True)
data_test,label_test,mark_test = read_file(test_set,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=False)

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