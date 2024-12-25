######author: Hongwei Ren#######
######This script is used to generate the h5 file for the DVS Action dataset######
import sys
import os
import struct
import cv2
import numpy as np
import uti
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

##### utilize the train-free SNN model to denoise the events #####
class SNN():
    """Spiking Neural Network.
    ts: timestamp list of the event stream.
    x: x-coordinate list of the event stream.
    y: y-coordinate list of the event stream.
    pol: polarity list of the event stream.  
    threshold: threshold of neuron firing.
    decay: decay of MP with time.
    margin: margin for lateral inhibition.
    spikeVal: MP increment for each event.
    network: MP of each neuron.
    timenet: firing timestamp for each neuron.
    firing: firing numbers for each neuron.
    """                    
    def __init__(self): 
        self.ts = []
        self.x = []
        self.y = []
        self.pol = []
        self.threshold = 1.2                                          
        self.decay     = 0.02                                          
        self.margin    = 3                                             
        self.spikeVal  = 1
        self.network   = np.zeros((260, 346), dtype = np.float64)
        self.timenet   = np.zeros((260, 346), dtype = np.int64)    
        self.firing = np.zeros((260, 346), dtype = np.int64)
        self.image = np.zeros((260, 346), dtype = np.int64)
        self.X = []
        self.Y = []
        self.T = []

    def init_timenet(self, t):
        """initialize the timenet with timestamp of the first event"""
        self.timenet[:] = t

    def spiking(self, data):
        """"main process"""
        count = 0
        img_count = 0   
        startindex = 0

        for line in data:
            self.ts.insert(count, int(line[0]))
            self.x.insert(count, int(line[1]))
            self.y.insert(count, int(line[2]))
            self.pol.insert(count, int(line[3]))

            if count == 0:
                self.init_timenet(self.ts[0])
                starttime = self.ts[0]
               
            self.neuron_update(count, self.spikeVal)
            
            if self.ts[count] - starttime > 50000:
                self.show_image(img_count,count)
                img_count += 1
                starttime = self.ts[count]
                self.image *= 0
                self.firing *= 0

            count += 1

        print('done')
        
    def clear_neuron(self, position):
        """reset MP value of the fired neuron"""             
        for i in range((-1)*self.margin, self.margin):
            for j in range((-1)*self.margin, self.margin):
                if position[0]+i<0 or position[0]+i>=180 or position[1]+j<0 or position[1]+j>=180:
                    continue
                else:
                    self.network[ position[0]+i ][ position[1]+j ] = 0.0

    def neuron_update(self, i, spike_value):
        """update the MP values in the network"""
        x = self.x[i]
        y = self.y[i]
        escape_time = (self.ts[i]-self.timenet[y][x])/1000.0
        residual = max(self.network[y][x]-self.decay*escape_time, 0)
        self.network[y][x] = residual + spike_value
        self.timenet[y][x] = self.ts[i]
        if self.network[y][x] > self.threshold:
            self.firing[y][x] += 1      # countor + 1
            self.clear_neuron([x,y])

    def show_image(self, img_count,ts):
        """convert to and save grayscale images"""
        self.image = np.flip(255*2*(1/(1+np.exp(-self.firing))-0.5),0)
        self.image = self.image[50:200,100:250]
        kernel = np.ones((2, 2), dtype=np.uint8)
        self.image = cv2.dilate(self.image, kernel, 1)
        kernel = np.ones((3, 3), dtype=np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=1)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.image[i, j] > 0:
                    self.X.append(j)
                    self.Y.append(i)
                    self.T.append(self.ts[ts])

def getDVSeventsDavis(file, numEvents=1e10, startTime=0):
    print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
    print('Reading in at most', str(numEvents))
    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []
    pol = []
    numeventsread = 0
    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)
    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue
    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if tm >= startTime:
            if (ad & typemask) == typedvs:
                xo = sizeX - 1 - float((ad & xmask) >> xshift)
                yo = float((ad & ymask) >> yshift)
                polo = 1 - float((ad & polmask) >> polshift)
                if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                    x.append(xo)
                    y.append(yo)
                    pol.append(polo)
                    ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1
    print('Total number of events read =', numeventsread)
    print('Total number of DVS events returned =', len(ts))
    return ts, x, y, pol

NUM_CLASSES = 10
DATA_PATH = "D:\\dataset\\action\\dataset\\Action-Recognition\\"
EXPORT_PATH = "D:\\dataset\\action\\dataset\\"
print('Data will save to', EXPORT_PATH)
class_index = ["arm-crossing","get-up","kicking","picking-up","jumping","sit-down","throwing","turning-around","walking","waving"]
WINDOW_SIZE = 0.5
STEP_SIZE = 0.25
SEQ_LEN = 1
NUM_POINTS = 2048

##### Read train and test txt file #####
train_txt =[]
with open("./train_action.txt", "r") as txt_file:
    for line in txt_file:
        train_txt.append(line.strip())
test_txt =[]
with open("./test_action.txt", "r") as txt_file:
    for line in txt_file:
        test_txt.append(line.strip())

data_train= []
label_train = []
marks_train =[]
data_test= []
label_test = []
marks_test = []
sample_count = 0
sequence = []
for index,class_label in enumerate(class_index):
    for root,dirs,files in os.walk(DATA_PATH+class_label):
        for file in files:
            print(os.path.join(root,file))
            T, X, Y, Pol = getDVSeventsDavis(os.path.join(root,file))
            len_event = len(T)
            EVENT_BEGIN = int(1/2 * len_event)
            EVENT_END = EVENT_BEGIN+int(2*len(T)/5)
            T = T[EVENT_BEGIN:EVENT_END]
            X = X[EVENT_BEGIN:EVENT_END]
            Y = Y[EVENT_BEGIN:EVENT_END]
            Pol = Pol[EVENT_BEGIN:EVENT_END]
            T = np.array(T).reshape((-1, 1))
            X = np.array(X).reshape((-1, 1))
            Y = np.array(Y).reshape((-1, 1))
            Pol = np.array(Pol).reshape((-1, 1))
            data_snn = np.hstack((T, X, Y, Pol))
            print(np.shape(data_snn))
            dvs_snn = SNN()
            dvs_snn.spiking(data_snn)
            class_events = np.zeros(shape=(int(len(dvs_snn.X)),3),dtype=np.int64)
            print(len(dvs_snn.X))
            class_events[:,0] = dvs_snn.T
            class_events[:,1] = dvs_snn.X
            class_events[:,2] = dvs_snn.Y
            win_start_index,win_end_index = uti.get_window_index(dvs_snn.T,dvs_snn.T[0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
            NUM_WINDOWS = len(win_start_index)
            count_numwindows = 0
            for n in range(NUM_WINDOWS):
                window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
                if window_events.shape[0] > 5000:
                    extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                    extracted_events[:,0] = extracted_events[:,0]-extracted_events[:,0].min(axis=0)
                    events_normed = extracted_events / extracted_events.max(axis=0)
                    events_normed[:,1] = extracted_events[:,1] / 150
                    events_normed[:,2] = extracted_events[:,2] / 150
                    count_numwindows +=1
                    if file in train_txt:
                        data_train.append(events_normed)
                        label_train.append(index)
                        marks_train.append(sample_count)
                    else:
                        data_test.append(events_normed)
                        label_test.append(index)
                        marks_test.append(sample_count)
                else:
                    continue
            sample_count += 1


data_train = np.array(data_train)
label_train = np.array(label_train)
marks_train = np.array(marks_train)
data_test = np.array(data_test)
label_test = np.array(label_test)
marks_test = np.array(marks_test)

data = data_train
label = label_train
mark = marks_train
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
mark = marks_test
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

