
import numpy as np
import torch
import random

def shuffle_downsample(data,num=None):
    ''' data is a numpy array '''
    if num == None:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
    elif num > data.shape[0]:
        idx = np.random.choice(np.arange(data.shape[0]), size=num, replace=True)
        idx.sort()
    else:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        idx = idx[0:num]
        idx.sort()
    return data[idx,...]

def normaliztion(orinal_events,w,h,process_p = False):
    """
    Normalize events.
    """
    events = orinal_events.copy()
    events = events.astype('float32')
    events[:, 0] = (events[:, 0] - events[:, 0].min(axis=0)) / (events[:, 0].max(axis=0) - events[:, 0].min(axis=0)+1e-6)
    events[:, 1] = events[:, 1] / w
    events[:, 2] = events[:, 2] / h
    if process_p:
        events[:, 3] = events[:, 3]*2-1
    return events

def random_crop(events,w=32,h=32):
    """
    Randomly crop events in space and time.
    """
    spatial_crop_range = [0.7, (w-1) / w]
    time_crop_range=[0.6, 1.0]
    min_x, max_x = 0, w
    min_y, max_y = 0, h
    min_t, max_t = int(events[0, 0]),  int(events[-1, 0])
    # print(min_t,max_t)
    events = torch.from_numpy(events)
    if random.random() > 0.5:
        # Spatial cropping
        scale = torch.rand(2) * (spatial_crop_range[1] - spatial_crop_range[0]) + spatial_crop_range[0]
        crop_size_x = int(scale[0] * (max_x - min_x))
        crop_size_y = int(scale[1] * (max_y - min_y))
        start_x = int(torch.randint(0, max_x - crop_size_x, (1,)))
        start_y = int(torch.randint(0, max_y - crop_size_y, (1,)))
        mask_x = torch.logical_and(events[:, 1] >= start_x, events[:, 1] <= start_x + crop_size_x)
        mask_y = torch.logical_and(events[:, 2] >= start_y, events[:, 2] <= start_y + crop_size_y)
        crop_mask = torch.logical_and(mask_x, mask_y)
        cropped_events = events[crop_mask]
        # Adaptive shift based on crop size
        x_shift = torch.randint(-start_x, w - start_x - crop_size_x + 1, size=(1,))
        y_shift = torch.randint(-start_y, h - start_y - crop_size_y + 1, size=(1,))
        cropped_events[:, 1] += x_shift
        cropped_events[:, 2] += y_shift    

    else:
        # Time cropping
        time_crop_range[1] = (max_t - min_t - 1) / (max_t - min_t)
        scale = torch.rand(1) * (time_crop_range[1] - time_crop_range[0]) + time_crop_range[0]
        crop_size_t = int(scale * (max_t - min_t))
        start_t = int(torch.randint(min_t, max_t - crop_size_t, (1,)))
        crop_mask = torch.logical_and(events[:, 0] >= start_t, events[:, 0] <= start_t + crop_size_t)
        cropped_events = events[crop_mask]
    
    return cropped_events.numpy()


def get_window_index(events,start,stepsize,windowsize):
    """
    Extract each class from original video
    """
    win_start_index = []
    win_end_index = []
    win_end_index_ = []
    idx = 0
   
    while idx < len(events):
        if (events[idx] >= start)&(start+windowsize<events[-1]):
            win_start_index.append(idx)
            start = start + stepsize
        else:
            idx = idx + 1
   
    idx = len(events)-1
    end = start - stepsize + windowsize
    while idx >= 0:
        if events[idx] <= end:
            win_end_index_.append(idx)
            end = end - stepsize
        else:
            idx = idx - 1
    
    
    for j in range(len(win_start_index)):
        win_end_index.append(win_end_index_[j])
    
    win_end_index=win_end_index[::-1]
   
    return win_start_index,win_end_index