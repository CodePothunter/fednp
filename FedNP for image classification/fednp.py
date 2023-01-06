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
import math

from utils import clf_dataset, split_dataset

from models.resnet import ResNet18
from models.npn import NPN

local_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='Dylon', type=str)
parser.add_argument('--experiment', default='NP', type=str)
parser.add_argument('--date', default=local_time, type=str)
parser.add_argument('--description', default='non-iid', type=str)
# training data
parser.add_argument('--root', default='path to training set', type=str)
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--data_dist', default='noniid', type=str)
parser.add_argument('--num_workers', default=4, type=int)
# Training Information
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--K', default=10, type=int, help="number of clients")
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--mu', default=0.01, type=float, help="loss factor")
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--local_epochs', default=10, type=int)
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
writer = SummaryWriter(f'runs/{args.experiment}{args.date[8:]}')

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

num_variables = global_model.linear.weight.numel()
npn_models = [NPN(models[i]).cuda() for i in range(args.K)]

global_m = torch.zeros(10).cuda()
global_s = torch.ones(10).cuda() / args.K
cavities = [(torch.zeros(10).cuda(), torch.ones(10).cuda() / (args.K - 1)) for i in range(args.K)]

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


EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)

def calc_template(a, b, c, d):
    return (((a + b) * c) - (b * d))


def calc_statistics(x, y, a, b):
    ksei_square = math.pi / 8
    nu = a * (x + b)
    de = torch.sqrt(torch.relu(1 + ksei_square * a * a * y))
    return torch.sigmoid(nu / (de + EPS))


def get_ep_prior(theta_m, theta_s, fx):
    fx = fx.mean().squeeze()
    fxsq = fx * fx
    cm, cs = fx * theta_m, fxsq * theta_s

    e_1 = calc_statistics(cm, cs, 1, 0)
    e_2 = calc_statistics(cm, cs, 4 - 2 * SQRT2, math.log(SQRT2 + 1))
    e_3 = calc_statistics(cm, cs, 6 * (1 - 1 / POW2_3O2), math.log(POW2_3O2 - 1))

    _p_1 = calc_template(cm, cs, e_1, e_2)
    _p_2 = calc_template(cm, 2 * cs, e_2, e_3) ###
    s_0 = e_1
    s_1 = _p_1 / (s_0 * fx + EPS)
    s_2 = (cs * e_1 + calc_template(cm, cs, _p_1, _p_2)) / (s_0 * fxsq + EPS)

    theta_m, theta_s = s_1, torch.relu(s_2 - s_1 * s_1) + EPS
    theta_m = torch.clamp(theta_m, EPS, 5)
    theta_s = torch.clamp(theta_s, EPS, 5)
    del cm, cs, e_1, e_2, e_3, s_0, s_1, s_2
    return theta_m, theta_s


def merge_ep_prior_and_get_cavity(client_m, client_s):
    lmd = [(1/(s+EPS)) for s in client_s]
    theta_s = 1/(sum(lmd)+EPS)
    theta_m = theta_s * sum([m*l for m,l in zip(client_m, lmd)])
    theta_m = torch.clamp(theta_m, EPS, 5)
    theta_s = torch.clamp(theta_s, EPS, 5)
    cavities = []
    for i in range(len(client_m)):
        rest_lmd = lmd[:i] + lmd[i+1:]
        rest_m = client_m[:i] + client_m[i+1:]
        c_s = (1 / (sum(rest_lmd) + EPS))
        c_m = (c_s * sum([m*l for m,l in zip(rest_m, rest_lmd)]))
        cavities.append((c_m, c_s))
    
    return theta_m, theta_s, cavities


def remove_cavity(tm, ts, cm, cs):
    tb = tm / (ts + EPS)
    td = -0.5 / (ts + EPS)
    cb = cm / (cs + EPS)
    cd = -0.5 / (cs + EPS)
    qb = tb - cb
    qd = torch.relu(td - cd) + EPS
    qs = - 2  / (qd + EPS)
    qm = qb * qs
    qm = torch.clamp(qm, EPS, 5)
    qs = torch.clamp(qs, EPS, 5)
    return qm, qs


def train_one_epoch(client_id, epoch, global_model, model, npn_model, cavity, train_loader):
    start_epoch = time.time()

    losses = libs.AverageMeter()
    top1 = libs.AverageMeter()
    top5 = libs.AverageMeter()

    model.train()
    npn_model.train()
    global_model.eval()

    model.load_state_dict(global_model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_npn = torch.optim.SGD(npn_model.parameters(), lr=args.lr, weight_decay=args.wd)

    cavity_mu, cavity_sigma = cavity

    with torch.no_grad():
        fx_all = []
        for i, (X, y) in enumerate(train_loader):
            X = X.cuda()
            _, fx = model(X)
            fx_all.append(fx)
        
        mu, sigma = get_ep_prior(cavity_mu, cavity_sigma, torch.cat(fx_all))

    for local_epoch in range(args.local_epochs):
        for i, (X, y) in enumerate(train_loader):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            y_pred, fx = model(X)

            kld_loss = torch.mean(-0.5 * ((1 + torch.log(torch.sqrt(sigma + EPS).mean()) - mu.mean() ** 2 - torch.sqrt(sigma + EPS).mean())))
                       
            mu, sigma = npn_model((mu, sigma))

            loss_np = ((model.linear.weight.reshape(-1) - mu) ** 2 / (2 * sigma + EPS)).mean()

            loss = loss_fn(y_pred, y)

            prec1, prec5 = libs.clf_accuracy(y_pred.data, y, topk = (1, 5))
            logging.info('Epoch: {}_Iter:{} loss: {:.4f} top1: {:.2f} top5: {:.2f}'.format(epoch, i, loss, top1.avg, top5.avg))

            losses.update(loss.data.item(), X.shape[0])

            top1.update(prec1.data.item(), X.size(0))
            top5.update(prec5.data.item(), X.size(0))

            optimizer.zero_grad()
            optimizer_npn.zero_grad()
            (loss +  args.mu * (loss_np + kld_loss)).backward()

            optimizer.step()
            optimizer_npn.step()
            mu, sigma = get_ep_prior(cavity_mu, cavity_sigma, fx.detach())

    with torch.no_grad():
        fx_all = []
        for i, (X, y) in enumerate(train_loader):
            X = X.cuda()
            _, fx = model(X)
            fx_all.append(fx)
        mu, sigma = get_ep_prior(cavity_mu, cavity_sigma, torch.cat(fx_all))
        mu, sigma = remove_cavity(mu, sigma, cavity_mu, cavity_sigma)

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

    end_epoch = time.time()
    epoch_time_minute = (end_epoch-start_epoch)/60
    logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))

    return model.state_dict(), mu, sigma

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

    # training
    clients_mu, clients_sigma = [], []
    for i in range(args.K):
        client_w, mu, sigma = train_one_epoch(i, epoch, global_model, models[i], npn_models[i], cavities[i], train_loaders[i])
        clients_mu.append(mu)
        clients_sigma.append(sigma)

        for key in global_w_next:
            global_w_next[key] += client_w[key] / args.K
    
    global_model.load_state_dict(global_w_next)
    _, _, cavities = merge_ep_prior_and_get_cavity(clients_mu, clients_sigma)

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
