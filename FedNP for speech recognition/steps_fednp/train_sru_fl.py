"""
An aggressive version that reduces the influence from GAN, but emphize EP-prior
regularization.

Move the EP-prior to discriminator ...
"""


import json
import math
import os
import random
import sys
import time
#!!! please modify these hyperprameters manually
import warnings
from shutil import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
from npn import *
from sru import SRU

from compute_priors import compute_priors
from dataGenSequences_sru import dataGenSequences
from lib.ops import Dense

EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)

warnings.filterwarnings('ignore')
# this depend on the feature you applied
mfccDim=40
seed = random.randint(0,10000)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) # cpu 
torch.cuda.manual_seed_all(seed) # gpu 
torch.backends.cudnn.deterministic = True


if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 8 and len(sys.argv) != 11:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir init_lr [turns] [epoch_per_turn]')


data_cv = sys.argv[1]
ali_cv  = sys.argv[2]
data_tr = sys.argv[3]
ali_tr  = sys.argv[4]
gmm     = sys.argv[5]
exp     = sys.argv[6]
init_lr = float(sys.argv[7])
try:
    clients = int(sys.argv[8])
    turns = int(sys.argv[9])
    epoch_per_turn = int(sys.argv[10])
except:
    turns = 12
    epoch_per_turn = 1
    clients = 16

##!!! please modify these hyperprameters manually
## Learning parameters
z_size              = 16
batch_size          = 128
labmda              = 1e-3


learning = {'rate' : init_lr,
            'singFeaDim' : mfccDim, 
            'minEpoch' : 30,
            'batchSize' : batch_size,
            'timeSteps' : 20,
            'dilDepth' : 1,
            'minValError' : 0,
            'left' : 0,
            'right': 4,
            'hiddenDim' : 1280,
            'modelOrder' : 1,
            'layerNum': 12,# 12 at first
            'z_size':z_size}

## Copy final model and tree from GMM directory
os.makedirs(exp, exist_ok=True)
copy(gmm + '/final.mdl', exp)
copy(gmm + '/tree', exp)


def return_my_self(batch):
    x=[]
    y=[]
    try:
        for xx,yy in batch:
            x.append(xx)
            y.append(yy)
        return torch.stack(x), torch.stack(y)
    except:
        print("Error")
        print(batch)
        exit(0)

if clients == 16:
    sessions = [["S03"], ["S04"], ["S05"], ["S06"], 
                ["S07"], ["S17"], ["S08"], ["S16"], 
                ["S12"], ["S13"], ["S18"], ["S22"], 
                ["S19"], ["S20"],["S23"], ["S24"]]
elif clients == 8:
    sessions = [["S03", "S04"], ["S05", "S06"], 
                ["S07", "S17"], ["S08", "S16"], 
                ["S12", "S13"], ["S18", "S22"], 
                ["S19", "S20"], ["S23", "S24"]]
elif clients == 4:
    sessions = [["S03", "S04", "S08", "S16"],
                ["S05", "S12", "S19", "S23", "S24"],
                ["S06", "S07", "S17"],
                ["S13", "S18", "S22", "S20"]]
else:
    raise NotImplementedError

## Compute priors
compute_priors(exp, ali_tr, ali_cv)

# The input feature of the neural network has this form:  0-1-4 features
feaDim = (learning['left'] + learning['right'] + 1) * mfccDim # 5 x 40 = 200
# discriminator output as feature extension
disDim = learning['disDim']


# load data from data iterator
trDatasets = [dataGenSequences(data_tr+'/'+str(i), ali_tr, gmm, learning['batchSize'], learning['timeSteps'], 
                    feaDim, learning['left'], learning['right'], 
                    my_sess=sessions[i-1]) for i in range(1, clients+1)]
cvDataset = dataGenSequences(data_cv, ali_cv, gmm, learning['batchSize'], learning['timeSteps'], feaDim,
                             learning['left'], learning['right'])

# Recommend shuffle=False, because this iterator's shuffle can only work on the single split
trGens = [data.DataLoader(trDatasets[i], batch_size=learning['batchSize'], 
                    shuffle=False, num_workers=0, collate_fn=return_my_self) \
                    for i in range(clients)]
client_weights = [trDatasets[i].numFeats for i in range(clients)]

cvGen = data.DataLoader(cvDataset, batch_size=learning['batchSize'], 
                        shuffle=False, num_workers=0, collate_fn=return_my_self)


##load the configurations from the training data
learning['targetDim'] = trDatasets[0].outputFeatDim

with open(exp + '/learning.json', 'w') as json_file:
    json_file.write(json.dumps(learning))

def FedAvg(params, weights=None):
    if not weights:
        weights = [1] * len(params)
    # assert len(params) == len(weights), (len(params), len(weights))
    avg_params = []
    for _, param in enumerate(zip(*params)):
        avg_params.append(weighted_sum(param, weights))
    return avg_params

def weighted_sum(inputs, weights):
    weights_sum = sum(weights)
    # print(weights[0] / weights_sum)
    res = 0
    for i, w in zip(inputs, weights):
        res += i * (w / weights_sum)
    return res


def mean(a):
    assert type(a) == list 
    return sum(a) / len(a)


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


class SRU_FedNP(nn.Module):
    def __init__(self, input_size=feaDim, hidden_size=1024, output_size=1095, 
                  num_layers=learning['layerNum']):
        super(SRU_FedNP, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.Dense_layer1 = Dense(input_size, self.hidden_size)

        self.sru = SRU(input_size=self.hidden_size, 
                       hidden_size=self.hidden_size,
                       num_layers=self.num_layers,
                       dropout=0.1, 
                       use_tanh=True)

        self.Dense_layer3 = Dense(self.hidden_size, 1024)

        self.Dense_layer4 = Dense(1024, output_size)
        
        self.dropout = nn.Dropout(p=0.1)
        self.eps = 1e-3
        self.len_hist = 1.0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(
                        learning['layerNum'], 
                        learning['batchSize'], 
                        learning['hiddenDim']
                            
            ).cuda()

        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer1(x)
        x = self.dropout(x)
        x = torch.reshape(x, (b, t, self.hidden_size))
        x = x.permute(1, 0, 2)
        x, hidden_after = self.sru(x, hidden)
        x = x.permute(1, 0, 2)
        b, t, h = x.size()
        x = torch.reshape(x, (b * t, h))
        fx = self.Dense_layer3(x)
        x = torch.relu(fx)
        x = self.dropout(x)
        x = self.Dense_layer4(x).view(-1, self.output_size)

        return x, fx, hidden_after


# If you run this code on CPU, please remove the '.cuda()'
model = SRU_FedNP(input_size=feaDim, hidden_size=learning['hiddenDim'],
                    output_size=learning['targetDim']).cuda()


model_total_params = sum(p.numel() for p in model.parameters())
print("Total SRU parameters:", model_total_params)
N_S_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total SRU parameters to update:", N_S_PARAMS)

aux_model = Hybrid(z_size, len(list(model.parameters()))).cuda()

loss_classify = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_model.parameters()), \
     lr=learning['rate'], betas=(0.5, 0.999))

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=turns * epoch_per_turn // 2,
                                        gamma=1) # turn of the LR scheduling


def train_one_epoch(global_model, model, aux_model, cms, css, client, epoch, hidden):
    acc = 0

    model.train()
    aux_model.train()

    for batch_idx, (x,y) in enumerate(tqdm.tqdm(trGens[client])):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, t, h = x.size()
        if b == learning['batchSize']:
            """
                Update Classifier
            """
            model.train()
            D.train()
            G.eval()
            # Classifier optimization
            output, f_x, hidden_after = model(x, hidden_before)
            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()
            loss = loss_classify(output, y)

            # calculate extended loss
            m, s = get_ep_prior(cms[client], css[client], f_x)
            m, s = aux_model((m, s))

            loss_np = torch.sum([(p - mm) ** 2 / (2 * ss + EPS) for p, mm, ss in zip(model.parameters(), m, s)]) / model_total_params
        
            loss += labmda * loss_np
            if batch_idx % 100 == 0:
                print(f"Total_SRU_loss: {loss:.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output.data, 1)
            hidden_after = hidden_after.detach()
            acc += ((pred == y).sum()).cpu().numpy()

            



            nn.utils.clip_grad_norm_(parameters=model.parameters(), 
                                    max_norm=100, norm_type=2)
            nn.utils.clip_grad_norm_(parameters=G.parameters(), 
                                    max_norm=100, norm_type=2)
            nn.utils.clip_grad_norm_(parameters=D.parameters(), 
                                    max_norm=100, norm_type=2)

    local_mu, local_sigma = remove_cavity(m, s, cms[client], css[client])

    print("Train acc:", acc)
    print("Train numFeats:", trDatasets[client].numFeats)
    print("Train accuracy: %f"%(acc/trDatasets[client].numFeats))
    return model, aux_model, local_mu, local_sigma


def val(model, train_loader, hidden=None):
    if hidden is None:
        hidden = torch.zeros(
                learning['layerNum'], 
                learning['batchSize'], 
                learning['hiddenDim']
            ).cuda()

    model.eval()
    acc = 0
    val_loss = 0
    val_loss_list = []
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, t, h = x.size()
        if b == learning['batchSize']:
            #model.zero_grad()
            optimizer.zero_grad()
            if batch_idx == 0:
                # hidden_before = torch.from_numpy(hidden).cuda()
                hidden_before = hidden.cuda()

            else:
                # hidden_before = torch.from_numpy(hidden_after).cuda()
                hidden_before = hidden_after.cuda()

            #output, hidden_after, cell_after = model(x, hidden_before, cell_before)
            #print(hidden_before.dtype)
            with torch.no_grad():
                output, hidden_after = model(x, hidden_before)
                #output = model(x)

            _, pred = torch.max(output.data, 1)
            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()
            loss = loss_classify(output, y)
            val_loss += float(loss.item())
            val_loss_list.append(val_loss)
            acc += ((pred == y).sum()).cpu().numpy()
            # if (batch_idx % 1000 == 0):#1000/60
            #     print("val:\t\tepoch:%d ,step:%d, loss:%f" % (epoch + 1, batch_idx, loss))

    print('Valid acc:', acc)
    print('Valid numFeats:', cvDataset.numFeats)
    val_acc = acc/cvDataset.numFeats
    print("Valid accuracy: %f" % (val_acc))
    print("Valid lOSS: %f" % (val_loss / len(val_loss_list)))
    return float(val_loss / len(val_loss_list)), val_acc


def train_net_local(model, aux_model, optimizer, cms, css,
                    client, vals, turn, max_epoch):
    val_loss_before = vals[client]
    tol_epoch = 0
    for epoch in range(max_epoch):
        print(f"====== Client: {client} trun-ep: {turn}-{epoch} =====")
        h = torch.zeros(
                learning['layerNum'], 
                learning['batchSize'], 
                learning['hiddenDim']
            )

        time_start = time.time()
        _, _, m, s = train_one_epoch(model, aux_model, optimizer, cms, css, client, epoch, h)
        # val_loss_after, _ = val(model, cvGen, loss_classify, optimizer, h)
        # if(val_loss_before - val_loss_after < 0):
            # scheduler.step()
            # tol_epoch += 1
        # else:
            # val_loss_before = val_loss_after
            # tol_epoch = 0
        # torch.save(model.state_dict(), exp + '/dnn.nnet.pth')
        time_end = time.time()
        time_cost = time_end - time_start
        print("Time Cost : %f"%(time_cost))
        if tol_epoch > 3:
            print(f"Client {client} local optimization finished"
                  f" at epoch {epoch}")
            break

    val_loss_after, _ = val(model, cvGen, h)
    vals[client]=val_loss_after
    return vals, m, s


def train_fl_net(model, aux_model, optimizer, clients, turns, epoch_per_turn):
    server_model_parameters = [param.cpu() for param in model.parameters()]
    server_aux_model_parameters = [param.cpu() for param in aux_model.parameters]

    cms = [torch.zeros(z_size, requires_grad=False).cuda() for _ in range(clients)]
    css = [torch.ones(z_size, requires_grad=False).cuda() for _ in range(clients)]

    vals = [10000 for i in range(clients)]
    for turn in range(turns):
        print('>>> current learning rate:', optimizer.param_groups[0]['lr'])
        client_model_parameters = []
        client_aux_model_parameters = []
        client_ms = []
        client_ss = []
        
        for client in range(clients):
            model.zero_grad()
            aux_model.zero_grad()
            # broadcasting parameters
            for p, sp in zip(model.parameters(), server_model_parameters):
                p.data.copy_(sp.cuda())
            for p, sp in zip(aux_model.parameters(), server_aux_model_parameters):
                p.data.copy_(sp.cuda())
            print(f"Runing turn {turn} on client {client}.")
            vals, m, s = train_net_local(model, aux_model, optimizer, cms, css, client, vals, turn, epoch_per_turn)
            client_model_parameters.append(
                [param.cpu() for param in model.parameters()])
            client_aux_model_parameters.append(
                [param.cpu() for param in aux_model.parameters()])
            client_ms.append(m)
            client_ss.append(s)
        server_model_parameters = FedAvg(client_model_parameters, client_weights)
        server_aux_model_parameters = FedAvg(client_aux_model_parameters, client_weights)
        _, _, cavities = merge_ep_prior(clients_ms, client_ss)

        scheduler.step()
        for p, sp in zip(model.parameters(), server_model_parameters):
            p.data.copy_(sp.cuda())
        for p, sp in zip(aux_model.parameters(), server_aux_model_parameters):
            p.data.copy_(sp.cuda())
        
        val_loss_after, _ = val(model, D, cvGen)
        print(f"Save the {turn} model with valid loss {val_loss_after}")
        torch.save(model.state_dict(), exp + '/dnn.nnet.pth')
        torch.save(aux_model.state_dict(), exp + '/dnn.nnetA.pth')
    print("Finished ... ")



train_fl_net(model, aux_model, optimizer, clients, turns, epoch_per_turn)
