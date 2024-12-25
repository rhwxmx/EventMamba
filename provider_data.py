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