#####author: Hongwei Ren#######
##### This script is used to train the model for the Action recognition tasks#####
import os
import sys
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
import provider_data
import time
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import sklearn.metrics as metrics
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default = 16, help='batch size in training')
    parser.add_argument("--data_path", type=str, default='/home/rhwdmx/github/eventmamba/EventMamba/data/dvsaction/', help="path to dataset")
    parser.add_argument('--num_category', default=10, type=int, help='the category of action recogniton')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument("--log_path", type=str, default='./tensorboard_log/', help="path to tesnorboard_log")
    parser.add_argument("--log_name", type=str, default='/dvsaction_2048_512_batch16', help="path to tesnorboard_log")
    parser.add_argument('--epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training: Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def validate(net, testloader, criterion, device, mark, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        label_seq = [[],[]]
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]

            ######calculate the accuracy of all sliding window######
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()

            ######calculate the accuracy of all sequence######
            label_seq[0] = list([j for i in test_pred for j in i])
            label_seq[1] = list([j for i in test_true for j in i])
            from collections import Counter
            count = 0
            correct_seq= 0
            index = 0
            for i in range(len(label_seq[1])-1):
                #### if mark is different, we run this code###
                if (mark[i] != mark[i+1]) or (i == len(label_seq[1])-2):
                    ####statistic the most common label in the sequence####
                    tar = Counter(label_seq[0][index:i+1])
                    tar = tar.most_common(1)[0][0]
                    ####if the most common label is equal to the label of the sequence, we count it####
                    correct_seq += 1 if tar==label_seq[1][i] else 0
                    index = i+1
                    count +=1
            count = -1 if count == 0 else count
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_seq: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,correct_seq/count*100,correct_seq,count))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "time": time_cost,
        "test_acc_seq":correct_seq/count
    }

def train(net, trainloader, optimizer, criterion, device, args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data = data.reshape(args.batch_size,args.num_point,3)
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]
        ######calculate the accuracy of all sliding window######
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "time": time_cost
    }

def cal_loss(pred, gold, smoothing=True):
    ##### Calculate cross entropy loss, apply label smoothing if needed.#####
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
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
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(args.log_path + args.log_name)
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
    current_data_test, current_label_test, current_mark_test = provider_data.load_h5_mark(TEST_FILES[0])
    print("test",len(current_data_test),current_data_test.shape)
    current_label_test = np.squeeze(current_label_test)
    current_data_test = torch.from_numpy(current_data_test)
    current_label_test = torch.from_numpy(current_label_test.astype('int64'))
    current_data_test = current_data_test.reshape(-1,args.num_point,3)
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    current_data_train, current_label_train,current_mark_sum = provider_data.load_h5_mark(TRAIN_FILES[0])
    print("train",len(current_data_train))
    current_label_train = np.squeeze(current_label_train)  
    print(current_data_train.shape,current_label_train.shape)
    current_data_train = torch.from_numpy(current_data_train)
    current_label_train = torch.from_numpy(current_label_train.astype('int64'))
    current_data_train = current_data_train.reshape(-1,args.num_point,3)
    dataset = torch.utils.data.TensorDataset(current_data_train, current_label_train)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    ##### model loading #####
    best_test_acc = 0.  
    best_test_seq = 0.
    best_train_acc = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")

    from models.eventmamba import EventMamba
    classifier = EventMamba(num_classes=args.num_category)
    criterion = cal_loss
    classifier.apply(inplace_relu)
    device = 'cuda'
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
    global_epoch = 0

    ##### start training #####
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        ##### training #####
        train_out = train(classifier, trainDataLoader, optimizer, criterion, device,args)
        scheduler.step()
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        ##### save the training metrice into tensorboard #####
        writer.add_scalar('Loss/train', train_out["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_out["acc"], epoch)
        ##### testing #####
        test_out = validate(classifier, testDataLoader, criterion, device,current_mark_test,args)
        if test_out["test_acc_seq"] > best_test_seq:
            best_test_seq = test_out["test_acc_seq"]
            is_best = True
        else:
            is_best = False
        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc                    
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss            
        ##### save the test metrice into tensorboard #####
        writer.add_scalar('Accuracy/test', test_out["acc"], epoch)
        writer.add_scalar('Accuracy/test_seq', test_out["test_acc_seq"], epoch)
        writer.add_scalar('Loss/test', test_out["loss"], epoch)
        ##### save the model #####
        save_model(classifier, epoch, path=str(checkpoints_dir), acc=test_out["test_acc_seq"], is_best=is_best,
            best_test_acc=best_test_acc, 
            best_train_acc=best_train_acc,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict())            
        print(f"Testing loss:{test_out['loss']} " f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] [best test seq acc: {best_test_seq}%]")
        log_string('Train loss : %f' % train_out["loss"])
        log_string('Train acc : %f' % train_out["acc"])
        log_string('Test Accuracy: %f' % (test_out["acc"]))
        log_string('Test Accuracy seq: %f' % (test_out["test_acc_seq"]))
        log_string('Best Test Accuracy: %f' % (best_test_acc))
        log_string('Best Test Seq Accuracy: %f' % (best_test_seq))

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
