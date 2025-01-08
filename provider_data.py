import os
import sys
import numpy as np
import h5py

def load_h5_mark(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    mark = f['mark'][:]
    return (data, label,mark)

def load_h5(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_and_resample(h5_filename, sample_size=1024):
    print(f"Loading {h5_filename}")
    with h5py.File(h5_filename, 'r') as f:
        data = []
        labels = []
        for group_name in f:
            group = f[group_name]
            x = group['x'][:]
            y = group['y'][:]
            t = group['t'][:]
            current_sample_size = len(x)
            if current_sample_size < sample_size:
                repeat_factor = np.ceil(sample_size / current_sample_size).astype(int)
                x = np.tile(x, repeat_factor)[:sample_size]
                y = np.tile(y, repeat_factor)[:sample_size]
                t = np.tile(t, repeat_factor)[:sample_size]
            else:
                indices = np.random.choice(current_sample_size, sample_size, replace=False)
                indices = np.argsort(indices)
                x = x[indices]
                y = y[indices]
                t = t[indices]

            if current_sample_size >= 1024:
                sample_data = np.stack((t,x,y), axis=-1)
                data.append(sample_data)
                sample_label = group.attrs['label']
                labels.append(sample_label)

    return (data, labels)