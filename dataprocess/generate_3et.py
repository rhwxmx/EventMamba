######author: Xiaopeng Lin#######
######This script is used to generate the h5 file for the 3et eye tracking dataset######
import os
import h5py
import numpy as np
import tqdm

#### Process the labels by divide the resolution of event camera####
def process_labels(line):
    return [float(line[0]) / 640, float(line[1]) / 480]

def process_h5_and_labels(file_path, label_path):
    with h5py.File(file_path, 'r') as f:
        x = f['events'][:,1]
        y = f['events'][:,2]
        t = f['events'][:,0]
        p = f['events'][:,3]
        frame_ts = f['frame_ts'][:]
    with open(label_path, 'r') as file:
        labels = np.array([process_labels(line.split()) for line in file], dtype=float)

    samples = []
    sample_labels = []
    segments = []
    segment_index = 0

    for i in range(len(frame_ts)-1):
        segment_start = frame_ts[i]
        segment_end = frame_ts[i+1]
        mask = (t >= segment_start) & (t < segment_end)
        if mask.any():
            segments.append((x[mask], y[mask], t[mask],p[mask]))

    size = []
    for i, (segment_x, segment_y, segment_t, segment_p) in enumerate(segments):
        expand = 0
        ##### Frequency adpative sampling.if the number of events between two ground truth is larger than 6144, then add the label to the list_gap_gt#####
        if i < 5:
            expand_segments = segments[:5]
        elif i >= len(segments) - 5:
            expand_segments = segments[-5:]
        else:
            expand_segments = segments[i-expand:i+1+expand]
            number = [len(ex[0])for ex in expand_segments]
            while sum(number) < 6144 and expand < 5:
                expand += 1
                expand_segments = segments[i-expand:i+1+expand]
                number = [len(ex[0])for ex in expand_segments]

        ##### normalize the data#####
        expand_x = np.concatenate([seg[0] for seg in expand_segments])/240
        expand_y = np.concatenate([seg[1] for seg in expand_segments])/180
        expand_t = np.concatenate([seg[2] for seg in expand_segments]) 
        expand_p = np.concatenate([seg[3] for seg in expand_segments])  
        
        ##### Sort the events according to the timestamp#####
        sort_indices = np.argsort(expand_t)
        sorted_x = expand_x[sort_indices]
        sorted_y = expand_y[sort_indices]
        sorted_t = expand_t[sort_indices]
        sorted_t = (expand_t[sort_indices] - expand_t[sort_indices][0]) / (expand_t[sort_indices][-1] - expand_t[sort_indices][0]+1e-5)
        sorted_p = expand_p[sort_indices]
        if segment_index < len(labels):
            samples.append((sorted_x, sorted_y, sorted_t,sorted_p))
        print(samples)
        if i < len(labels):
            sample_labels.append(labels[i])
        segment_index += 1
    return samples, sample_labels

def save_samples_to_hdf5(samples, labels, filename):
    with h5py.File(filename, 'w') as f:
        for i, ((x, y, t,p), label) in enumerate(zip(samples, labels)):
            grp = f.create_group(f'sample_{i}')
            grp.create_dataset('x', data=np.array(x), compression="gzip")
            grp.create_dataset('y', data=np.array(y), compression="gzip")
            grp.create_dataset('t', data=np.array(t), compression="gzip")
            grp.create_dataset('p', data=np.array(p), compression="gzip")
            grp.attrs['label'] = label

def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
def get_data(data_dir):
    sorted_data_file_paths = sorted(data_dir)
    print(sorted_data_file_paths)
    all_samples = []
    all_labels = []
    for file_path in tqdm.tqdm(sorted_data_file_paths):
        label_path = file_path.replace("data", "label").replace(".h5", ".txt")
        samples, labels = process_h5_and_labels(file_path, label_path)
        all_samples += samples
        all_labels += labels
    return all_samples,all_labels

root_dir = 'D://eye_tracking//'
trainfile = "D://eye_tracking//train.txt"
testfile = "D://eye_tracking//test.txt"

train_filenames = load_filenames(trainfile)
val_filenames = load_filenames(testfile)
data_train = [os.path.join(root_dir+'data//', f + '.h5') for f in train_filenames]
data_val = [os.path.join(root_dir+'data//', f + '.h5') for f in val_filenames]
data_train,label_train = get_data(data_train)
data_test, label_test = get_data(data_val)

print('len(all_samples)=', len(data_train))
print('len(all_labels)=', len(data_test))

X_train = data_train
y_train = label_train
X_test = data_test
y_test = label_test

train_h5_path = './3et/train.h5'
test_h5_path = './3et/test.h5'
if not os.path.exists('./3et/'):
    os.makedirs('./3et/')
##### Save the data to h5 file#####
save_samples_to_hdf5(X_train, y_train, train_h5_path)
save_samples_to_hdf5(X_test, y_test, test_h5_path)