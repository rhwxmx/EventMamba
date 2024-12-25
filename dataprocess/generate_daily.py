######author: Hongwei Ren#######
######This script is used to generate the h5 file for the Daily DVS dataset######
import numpy as np
import sys
import os
import struct
import numpy as np
import uti
import h5py
import random
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class aedatUtils:
    def loadaerdat(datafile='path.aedat', length=0, version="aedat", debug=1, camera='DVS128'):
        # constants
        aeLen = 8  # 1 AE event takes 8 bytes
        readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
        td = 0.000001  # timestep is 1us
        if(camera == 'DVS128'):
            xmask = 0x00fe  # Bin -> 0000 0000 1111 1110 || Dec -> 254
            xshift = 1
            ymask = 0x7f00  # Bin -> 0111 1111 0000 0000 || Dec -> 32512
            yshift = 8
            pmask = 0x1     # Bin -> 0000 0000 0000 0001 || Dec -> 1
            pshift = 0
        else:
            raise ValueError("Unsupported camera: %s" % (camera))

        aerdatafh = open(datafile, 'rb')
        k = 0  # line number
        p = 0  # pointer, position on bytes
        statinfo = os.stat(datafile)
        if length == 0:
            length = statinfo.st_size # Define 'length' = Tamanho do arquivo

        print("file size", length)
        
        # Verifica a versão do Python. 
        if sys.version[0] == '3':
            value = 35 # Se for >= 3 le o cabeçalho em binário.
        else:
            value = '#' # Se for < 3 le o cabeçalho como string.

        # header
        lt = aerdatafh.readline()
        while lt and lt[0] == value:
            p += len(lt)
            k += 1
            lt = aerdatafh.readline() 
            if debug >= 2:
                print(str(lt))
            continue
        
        # variables to parse
        timestamps = []
        xaddr = []
        yaddr = []
        pol = []
        
        # read data-part of file
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen
        
        print(xmask, xshift, ymask, yshift, pmask, pshift)    
        while p < length:
            addr, ts = struct.unpack(readMode, s)
            # parse event type
            if(camera == 'DVS128'):     
                x_addr = (addr & xmask) >> xshift # Endereço x -> bits de 1-7
                y_addr = (addr & ymask) >> yshift # Endereço y -> bits de 8-14
                a_pol = (addr & pmask) >> pshift  # Endereço polaridade -> bit 0            
                if debug >= 3: 
                    print("ts->", ts) 
                    print("x-> ", x_addr)
                    print("y-> ", y_addr)
                    print("pol->", a_pol)

                timestamps.append(ts)
                xaddr.append(x_addr)
                yaddr.append(y_addr)
                pol.append(a_pol)
                    
            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen        

        if debug > 0:
            try:
                print("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
                n = 5
                # print("showing first %i:" % (n))
                # print("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
            except:
                print("failed to print statistics")
        t, x, y, p = np.array(timestamps), np.array(xaddr), np.array(yaddr), np.array(pol)
        return t - t[0], x, y, p



    def matrix_active(x, y, pol,filtered=None):
    
        matrix = np.zeros([128, 128]) # Cria uma matriz de zeros 128x128 onde serão inseridos os eventos
        pol = (pol - 0.5) # Os eventos no array de Polaridade passam a ser -0.5 ou 0.5
        
        if(len(x) == len(y)): # Verifica se o tamanho dos arrays são iguais   
            for i in range(len(x)):
                val = 0
                #se a flag do filtro for true. Os eventos serão somados
                #para que eles sejam normalizados pelo maior valor de um acumulo de eventos
                #e depois retirados por um limiar de ~30%
                if filtered == None or filtered == False:
                    val = pol[i]
                elif filtered == True:
                    val = 1
                matrix[x[i], y[i]] += val # insere os eventos dentro da matriz de zeros
        else:
            print("error x,y missmatch")    

        if filtered:
            maxValue = matrix.max()
            matrix = matrix/maxValue
            #matrix[matrix <= 0.5] = 0
            #matrix[np.logical_and(matrix > 0.1, matrix <= 0.3)] = 0.1
            #matrix[matrix >= 0.5] = 1
            matrix = (matrix * 255) # Normaliza a matriz para 8bits -> 0 - 255
        else:
            idx = 0
            limiar = 0.5
            for i in matrix: # Limita os eventos em dentro do limiar
                for j, v in enumerate(i):
                    if v > limiar:
                        matrix[idx][j] = limiar
                    if v < (limiar-1):
                        matrix[idx][j] = (limiar-1)
                idx += 1
            if limiar != 1:
                matrix = (matrix * 255) + 127.5 # Normaliza a matriz para 8bits -> 0 - 255
            
        return matrix

    def getFrameTimeBased(timeArray, polArray, xPosArray, yPosArray,timeStamp, Ti):
        aux = 0
        t2 = timeArray[(timeArray > Ti) & (timeArray <= Ti + timeStamp)]
        x2 = xPosArray[aux : aux + len(t2)]
        y2 = yPosArray[aux : aux + len(t2)]
        p2 = polArray[aux : aux + len(t2)]
        aux += len(t2)
        img = matrix_active(x2, y2, p2)
        img = rotateMatrix(img)
        return img

    def getFramesTimeBased(timeArray, polArray, xPosArray, yPosArray,timeStamp,filtered=None):
        totalImages = []
        i, aux = 0, 0
        images = []
        
        while (i + timeStamp) < abs(timeArray[-1]):
            t2 = timeArray[(timeArray > i) & (timeArray <= i + timeStamp)]
            x2 = xPosArray[aux : aux + len(t2)]
            y2 = yPosArray[aux : aux + len(t2)]
            p2 = polArray[aux : aux + len(t2)]
            aux += len(t2)
            img = aedatUtils.matrix_active(x2, y2, p2,filtered)
            rotacao = aedatUtils.rotateMatrix(img)
            images.append(img)	
            i += timeStamp
        totalImages.extend(images)
        totalImages = np.array(totalImages)
        return totalImages


    def rotateMatrix(mat): 
        N = len(mat)
        # Consider all squares one by one 
        for x in range(0, int(N/2)): 
            
            # Consider elements in group    
            # of 4 in current square 
            for y in range(x, N-x-1): 
                
                # store current cell in temp variable 
                temp = mat[x][y] 
    
                # move values from right to top 
                mat[x][y] = mat[y][N-1-x] 
    
                # move values from bottom to right 
                mat[y][N-1-x] = mat[N-1-x][N-1-y] 
    
                # move values from left to bottom 
                mat[N-1-x][N-1-y] = mat[N-1-y][x] 
    
                # assign temp to left 
                mat[N-1-y][x] = temp 

        return mat

def read_file(file_list,train=True):
    data= []
    label = []
    marks = []
    for i,file in tqdm(enumerate(file_list)):
        file_path = file[0]
        index = file[1]
        T, X, Y, P = aedatUtils.loadaerdat(file_path)
        class_events = np.zeros(shape=(int(len(X)),3),dtype=np.int64)
        class_events[:,0] = T
        class_events[:,1] = X
        class_events[:,2] = Y
        if len(T) > 0:
            win_start_index,win_end_index = uti.get_window_index(T,T[0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
            NUM_WINDOWS = len(win_start_index)
            for n in range(NUM_WINDOWS):
                window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
                if window_events.shape[0] > 10000:
                    extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        extracted_events[:,0] = extracted_events[:,0]//1000
                        extracted_events[:,0] = extracted_events[:,0]-extracted_events[:,0].min(axis=0)
                        events_normed = extracted_events / extracted_events.max(axis=0)
                        events_normed[:,1] = extracted_events[:,1] / 128
                        events_normed[:,2] = extracted_events[:,2] / 128
                        data.append(events_normed)
                        label.append(index)
                        marks.append(i)
                if train:
                    crop_events = uti.random_crop(window_events,128,128)
                    extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        label.append(index)
                        marks.append(i)
                    reverse_events = uti.reverse_T(window_events)
                    extracted_events = uti.shuffle_downsample(reverse_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        label.append(index)
                        marks.append(i)
                    fliph_events = uti.flip_H(window_events,128)
                    extracted_events = uti.shuffle_downsample(fliph_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        label.append(index)
                        marks.append(i)
                    flipw_events = uti.flip_W(window_events,128)
                    extracted_events = uti.shuffle_downsample(flipw_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        label.append(index)
                        marks.append(i)
                else:
                    continue

    data = np.array(data)
    label = np.array(label)
    marks = np.array(marks)
    print(data.shape)
    print(label.shape)
    return data,label,marks

NUM_CLASSES = 12
DATA_PATH = "D:\\dataset\\dailyaction\\DailyAction-DVS-20230313T111822Z-001\\DailyAction-DVS\\"
EXPORT_PATH ="D:\\dataset\\dailyaction\\DailyAction-DVS-20230313T111822Z-001\\DailyAction-DVS\\"
print('Data will save to', EXPORT_PATH)
WINDOW_SIZE = 1.5
STEP_SIZE = 0.5
SEQ_LEN = 1
NUM_POINTS = 2048
train_test = 0.2

##### divide the dataset to train and test by filenmame #####
class_index=[]
for root,dirs,files in os.walk(DATA_PATH):
    for dirs_son in dirs:
        class_index.append(dirs_son)

file_total=[]
file_train=[]
file_test=[]
for index,class_label in enumerate(class_index):
    for root,dirs,files in os.walk(DATA_PATH+"\\"+class_label):
        for file in files:
            file_total.append([ os.path.join(root,file),index])
num_to_select = int(len(file_total)*train_test)
file_test = random.sample(file_total, num_to_select)
file_train = [x for x in file_total if x not in file_test]


##### read the data and save to h5 file #####
data_train,label_train,mark_train = read_file(file_train,train=True)
data_test,label_test,mark_test = read_file(file_test,train=False)

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