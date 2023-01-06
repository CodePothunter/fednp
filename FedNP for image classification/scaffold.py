import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
from libs import torch as libs
import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn
import copy

from utils import clf_dataset, split_dataset
from models.resnet import ResNet18

local_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='Dylon', type=str)
parser.add_argument('--experiment', default='FedAvg', type=str)
parser.add_argument('--date', default=local_time, type=str)
parser.add_argument('--description', default='non-iid', type=str)
# training data
parser.add_argument('--root', default='path to training set', type=str)
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--data_dist', default='noniid', type=str)
parser.add_argument('--num_workers', default=4, type=int)
# Training Information
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--K', default=10, type=int, help="number of clients")
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--local_epochs', default=5, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--save_freq', default=50, type=int)
parser.add_argument('--beta', default=0.5, type=float, help="parameter of Dirichlet distribution")


args = parser.parse_args()

libs.init_Seed(args.seed)

# logging
log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', f'{args.experiment}{args.date[8:]}')
log_file = log_dir + '.txt'
libs.log_args(log_file)
logging.info('--------------------------------------This is all argsurations----------------------------------')
for arg in vars(args):
    logging.info('{}={}'.format(arg, getattr(args, arg)))
logging.info('----------------------------------------This is a halvingline----------------------------------')
logging.info('{}'.format(args.description))

#checkpoints
checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.date)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# tensorboard
writer = SummaryWriter(f'runs/{args.date}')

# prepare dataset
train_dataset, test_dataset, num_classes = clf_dataset(vars(args))
if args.data_dist != 'noniid':
    train_datasets = split_dataset(vars(args), train_dataset)
else:
    train_datasets, weights = split_dataset(vars(args), train_dataset)
    fig, ax = plt.subplots()
    ax.imshow(weights, cmap='GnBu')
    writer.add_figure('data_distribution', fig)

for i in range(args.K):
    logging.info('Samples for train of {}= {}'.format(i, len(train_datasets[i])))

train_loaders = []
for i in range(args.K):
    train_loaders.append(torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=args.batch_size,
                            drop_last=True, num_workers=args.num_workers, pin_memory=True, shuffle=True))


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256,
                            drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=False)

loss_fn = nn.CrossEntropyLoss()

# prepare model
global_model = ResNet18(num_classes=num_classes).cuda()
models = [ResNet18(num_classes=num_classes).cuda() for i in range(args.K)]

global_variate = copy.deepcopy(global_model.state_dict())
for key in global_variate:
    global_variate[key] = 0.
variates = [copy.deepcopy(global_variate) for i in range(args.K)]


@torch.no_grad()
def evaluate(model):
    losses = libs.AverageMeter()
    top1 = libs.AverageMeter()
    top5 = libs.AverageMeter()
    
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        y_pred, _ = model(X)
        loss = loss_fn(y_pred, y)
        prec1, prec5 = libs.clf_accuracy(y_pred.data, y, topk = (1, 5))
        losses.update(loss.data.item(), X.shape[0])
        top1.update(prec1.data.item(), X.size(0))
        top5.update(prec5.data.item(), X.size(0))
    return losses.avg, top1.avg, top5.avg

def train_one_epoch(client_id, epoch, global_model, model, global_variate, variate, train_loader):
    start_epoch = time.time()

    losses = libs.AverageMeter()
    top1 = libs.AverageMeter()
    top5 = libs.AverageMeter()

    model.load_state_dict(global_model.state_dict())

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model.train()
    global_model.eval()

    lr = args.lr

    for local_epoch in range(args.local_epochs):
        for i, (X, y) in enumerate(train_loader):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            y_pred, _ = model(X)

            loss = loss_fn(y_pred, y)

            prec1, prec5 = libs.clf_accuracy(y_pred.data, y, topk = (1, 5))
            losses.update(loss.data.item(), X.shape[0])
            top1.update(prec1.data.item(), X.size(0))
            top5.update(prec5.data.item(), X.size(0))
            logging.info('Epoch: {}_Iter:{} loss: {:.4f} top1: {:.2f} top5: {:.2f}'.format(epoch, i, loss, top1.avg, top5.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w = model.state_dict()
            for key in w:
                if 'bias' in key or 'weight' in key:
                    w[key] -= lr * (global_variate[key] - variate[key])
            model.load_state_dict(w)
    
    writer.add_scalar(f'loss_{client_id}', losses.avg, epoch)
    writer.add_scalar(f'top1_{client_id}', top1.avg, epoch)
    writer.add_scalar(f'top5_{client_id}', top5.avg, epoch)

    # evaluation
    if client_id == 0:
        loss, top1, top5 = evaluate(model)
        writer.add_scalar(f'eval_loss_{client_id}', loss, epoch)
        writer.add_scalar(f'eval_top1_{client_id}', top1, epoch)
        writer.add_scalar(f'eval_top5_{client_id}', top5, epoch)
        logging.info('Epoch: {} eval_loss: {:.4f} eval_top1: {:.4f} eval_top5: {:.4f}'.format(epoch, loss, top1, top5))
    
    # estimate gradient
    w_backup = copy.deepcopy(model.state_dict())
    for i, (X, y) in enumerate(train_loader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        y_pred, _ = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    k = len(train_loader)
    w_after = model.state_dict()
    for key in variate:
        variate[key] = (w_backup[key] - w_after[key]) / (k * lr)

    end_epoch = time.time()
    epoch_time_minute = (end_epoch-start_epoch)/60
    logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))

    return w_backup, variate

"""
Training
"""
start_time = time.time()
for epoch in range(args.epochs):
    # initialization
    start_epoch = time.time()
    global_w_next = copy.deepcopy(global_model.state_dict())
    for key in global_w_next:
        global_w_next[key] = 0

    global_variate_next = copy.deepcopy(global_variate)
    for key in global_variate_next:
        global_variate_next[key] = 0

    # training
    for i in range(args.K):
        client_w, variates[i] = train_one_epoch(i, epoch, global_model, models[i], global_variate, variates[i], train_loaders[i])
        for key in global_w_next:
            global_w_next[key] += client_w[key] / args.K
        for key in variates[i]:
            global_variate_next[key] += variates[i][key]

    global_model.load_state_dict(global_w_next)
    global_variate = global_variate_next

    # evaluation
    loss, top1, top5 = evaluate(global_model)
    writer.add_scalar('eval_loss', loss, epoch)
    writer.add_scalar('eval_top1', top1, epoch)
    writer.add_scalar('eval_top5', top5, epoch)
    logging.info('=================================================================')
    logging.info('Epoch: {} eval_loss: {:.4f} eval_top1: {:.4f} eval_top5: {:.4f}'.format(epoch, loss, top1, top5))

    end_epoch = time.time()
    epoch_time_minute = (end_epoch-start_epoch)/60
    remaining_time_hour = (args.epochs-epoch-1)*epoch_time_minute/60
    logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
    logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

writer.close()

final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
torch.save({
    'state_dict': global_model.state_dict(),
}, final_name)

end_time = time.time()
total_time = (end_time-start_time)/3600
logging.info('The total training time is {:.2f} hours'.format(total_time))
logging.info('----------------------------------The training process finished!-----------------------------------')
