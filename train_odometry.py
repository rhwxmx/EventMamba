##### author: Hongwei Ren#######
##### This script is used to train the model for the pose relocalization tasks#####
import os
import sys
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
import provider_data
import math
import time
import torch.nn.functional as F
from pathlib import Path
import math
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import time
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--num_regression', default=6, type=int,  help='6 dof in eular')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument("--data_path", type=str, default='/home/rhwdmx/github/eventmamba/EventMamba/data/shape_tran_1024/', help="path to dataset")
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training: Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument("--log_path", type=str, default='./tensorboard_log/', help="path to tesnorboard_log")
    parser.add_argument("--log_name", type=str, default='/shape_translation', help="the name of tesnorboard_log")
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def validate(net, testloader, criterion, mean, std, device):
    net.eval()
    test_true = []
    test_pred = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            if label.dim() == 1:
                label = label.unsqueeze(0)
            test_true.append(label.cpu().numpy())
            ##### use the std and mean which is from train dataset to recover the original value#####
            test_pred.append(logits.detach().cpu().numpy()*std+mean)
            progress_bar(batch_idx, len(testloader))

    error =  calculate_error(test_pred, test_true)
    median_result = np.median(error,axis=0)
    average_result = np.average(error, axis=0)
    return {
        "median":median_result,
        "average": average_result
    }

def train(net, trainloader, optimizer, criterion, mean, std, device):
    net.train()
    train_loss = 0
    test_pred = []
    test_true = []
    for batch_idx, (data, label) in enumerate(trainloader):
        data = data.transpose(1,2)
        data, label = data.to(device), label.to(device).squeeze()
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label,net)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        test_true.append(label.cpu().numpy()*std)
        test_pred.append(logits.detach().cpu().numpy()*std)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.6f '
                     % (train_loss / (batch_idx + 1)))
        
    error =  calculate_error(test_pred, test_true)
    median_result = np.median(error,axis=0)
    average_result = np.average(error, axis=0)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "median":median_result,
        "average": average_result,
    }

def calculate_error(pred, gold):
    error = []
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            pose_rotation = gold[i][j][3:6]
            pose_translation = gold[i][j][0:3]
            predicted_rotation = pred[i][j][3:6]
            predicted_translation = pred[i][j][0:3]
            error_rotation = np.linalg.norm(predicted_rotation-pose_rotation)*180/math.pi
            error_translation = np.linalg.norm(predicted_translation-pose_translation)
            error.append([error_translation,error_rotation])
    return error

def cal_loss(pred, gold, net):
    tr = 1
    rr = 1
    pred = pred.float()
    gold = gold.float()
    loss = F.mse_loss(pred[:,0:3], gold[:,0:3],reduction='mean')*tr + F.mse_loss(pred[:,3:6], gold[:,3:6],reduction='mean')*rr
    # l2_reg_coeff = 0.001
    # l2_reg = torch.tensor(0.0).cuda()
    # for param in net.parameters():
    #     l2_reg += torch.norm(param, p=2)
    # loss += l2_reg_coeff * l2_reg 
    return loss

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_model(net, epoch, path, acc, is_best, **kwargs):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'acc': acc
    }
    for key, value in kwargs.items():
        state[key] = value
    filepath = os.path.join(path, "last_checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best_checkpoint.pth'))

def main(args):
    ##### log #####
    def log_string(str):
        logger.info(str)
        print(str)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    writer = SummaryWriter(args.log_path + args.log_name)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eventmamba.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    ##### data loading #####
    log_string('Load dataset ...')
    TRAIN_FILES = [args.data_path+"train.h5"]
    TEST_FILES = [args.data_path+"test.h5"]

    current_data_test, current_label_test = provider_data.load_h5(TEST_FILES[0])
    current_data_test = current_data_test[:,:,:3]
    print("test",len(current_data_test),current_data_test.shape)
    current_label_test = np.squeeze(current_label_test)
    ######################
    ##### if the sequence is box_translation, please use the following code to convert the angle to the range of [-pi,pi]#####
    # last_element = current_label_test[:, 5] 
    # positive_indices = np.where(last_element > 0)
    # current_label_test[positive_indices, 5] -= 2*math.pi
    current_data_test = torch.from_numpy(current_data_test)
    current_label_test = torch.from_numpy(current_label_test)
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle= False, num_workers=0, drop_last=False)
    current_data_train, current_label_train = provider_data.load_h5(TRAIN_FILES[0])
    current_data_train = current_data_train[:,:,:3]
    print("train",len(current_data_train),current_data_train.shape)
    current_label_train = np.squeeze(current_label_train)
    ##################
    ##### if the sequence is box_translation, please use the following code to convert the angle to the range of [-pi,pi]#####
    # last_element = current_label_train[:, 5] 
    # positive_indices = np.where(last_element > 0)
    # current_label_train[positive_indices, 5] -=  2*math.pi
    current_data_train = torch.from_numpy(current_data_train)
    current_label_train = torch.from_numpy(current_label_train)
    ##### standardization and store the mean and etd to recover value in testset#####
    mean_train = torch.mean(current_label_train, dim=0)
    std_train = torch.std(current_label_train, dim=0)
    current_label_train = (current_label_train - mean_train)/std_train
    mean_train = mean_train.numpy()
    std_train = std_train.numpy()   
    print(current_data_train.shape, current_label_train.shape)
    dataset = torch.utils.data.TensorDataset(current_data_train, current_label_train)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
    ##### loading model #####
    from models.eventmamba import EventMamba
    classifier = EventMamba(num_classes=args.num_regression)
    criterion = cal_loss
    classifier.apply(inplace_relu)
    classifier = classifier.cuda()

    ##### try to load the pretrain model #####  
    try:
        checkpoint = torch.load('last_checkpoint.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['net'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    ##### optimizer #####
    if args.optimizer =="SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.optimizer =="AdamW":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.optimizer =="Adam":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    global_epoch = 0


    ##### start training #####
    logger.info('Start training...')
    best_test_t = 180
    best_test_r = 180
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train_out = train(classifier, trainDataLoader, optimizer, criterion,mean_train,std_train, 'cuda')
        log_string('loss : %f' % train_out["loss"])
        log_string('train_median_error m: %f rotation: %f' % (train_out["median"][0],train_out["median"][1]))
        log_string('train_average_error m: %f rotation: %f' % (train_out["average"][0],train_out["average"][1]))
        writer.add_scalar('median_error/train_translation', train_out["median"][0], epoch)
        writer.add_scalar('median_error/train_rotation', train_out["median"][1], epoch)
        writer.add_scalar('average_error/train_translation', train_out["average"][0], epoch)
        writer.add_scalar('average_error/train_rotation', train_out["average"][1], epoch)
        test_out = validate(classifier, testDataLoader, criterion, mean_train,std_train,'cuda')
        scheduler.step()
        if ((test_out["median"][0]+test_out["median"][1]*math.pi/180)<(best_test_t + best_test_r*math.pi/180)):
            is_best = True
        else:
            is_best = False
        best_test_t,best_test_r = (test_out["median"][0],test_out["median"][1]) if ((test_out["median"][0] + test_out["median"][1]*math.pi/180) < (best_test_t + best_test_r*math.pi/180) ) else (best_test_t,best_test_r)
        save_model(classifier, epoch, path=str(checkpoints_dir), acc=test_out["median"],is_best=is_best, optimizer=optimizer.state_dict())
        log_string('Test_median_error m: %f  rotation:%f' % (test_out["median"][0],test_out["median"][1]))
        log_string('Test_average_error m: %f  rotation:%f' % (test_out["average"][0],test_out["average"][1]))
        writer.add_scalar('median_error/test_translation', test_out["median"][0], epoch)
        writer.add_scalar('median_error/test_rotation', test_out["median"][1], epoch)
        writer.add_scalar('average_error/test_translation', test_out["average"][0], epoch)
        writer.add_scalar('average_error/test_rotation', test_out["average"][1], epoch)
        log_string('Best_median_error m: %f  rotation:%f' % (best_test_t,best_test_r))

    logger.info('End of training...')
if __name__ == '__main__':
    args = parse_args()
    main(args)
