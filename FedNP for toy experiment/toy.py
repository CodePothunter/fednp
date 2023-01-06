import os
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as pplot
import torch
import random
import copy


sns.set_theme()

client_1_n = 30
client_2_n = 20
client_3_n = 50

client_weight = [client_1_n / 100, client_2_n / 100, client_3_n / 100]

global_x = np.concatenate(
    [np.linspace(-7, -5, client_1_n).astype(np.float32),
     np.linspace(-2, 2, client_2_n).astype(np.float32),
     np.linspace(4, 8, client_3_n).astype(np.float32)], axis=0)
y =  - global_x * global_x + 2 * global_x + 4
global_y = y + np.random.randn(y.shape[-1]).astype(np.float32) * 5

client_1_x, client_1_y = global_x[:client_1_n], global_y[:client_1_n]
client_2_x, client_2_y = global_x[client_1_n:client_1_n+client_2_n], global_y[client_1_n:client_1_n+client_2_n]
client_3_x, client_3_y = global_x[client_1_n+client_2_n:], global_y[client_1_n+client_2_n:]


def draw_background(ax):
    ax.plot(client_1_x, client_1_y, 'rx')
    ax.plot(client_2_x, client_2_y, 'gx')
    ax.plot(client_3_x, client_3_y, 'bx')
    # p = pplot.plot(global_x, y, 'black')
    return ax 

def draw_curve(ax, a, b, c, e, color='black'):
    ax = draw_background(ax)
    x = np.linspace(-7, 8, 60)
    xx = x * x
    xxx = xx * x
    # xxxx = xxx * x
    y = a * xxx + b * xx + c * x + e
    ax.plot(x, y, color)
    
    return ax

fig, ax = pplot.subplots(figsize=(8,6))
# fig.show(draw_background(ax))
fig.show(draw_curve(ax, 0,-1,2,5))


class PolyNet(torch.nn.Module):
    def __init__(self):
        super(PolyNet, self).__init__()
        self.coef_ = torch.nn.Linear(4, 1, bias=False)
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, x):
        construct_input = torch.cat([x*x*x, x*x, x, torch.ones(x.size(0), 1)], dim=1)
        # construct_input = torch.cat([x, torch.ones(x.size(0),1)], dim=1)
        return self.coef_(construct_input)

def make_dataset(x):
    x = torch.tensor(x).unsqueeze(1)
    xx = x * x
    xxx = xx * x 
    # xxxx = xxx * x
    X = torch.cat([xxx, xx, x], dim=1)
    X_min = torch.min(X, dim=0).values.expand(X.size())
    X_max = torch.max(X, dim=0).values.expand(X.size())
    X = (X - X_min) / (X_max - X_min)
    return X


batch_size = 10
losses = []



def train_one_epoch(client, x, y, batch_size, alg='fedavg', last_global=None, c_local=None, c_global=None, previous_local=None):
    algs_need_last_global = ['fedprox', 'scaffold', 'moon']
    if alg in algs_need_last_global:
        assert last_global is not None, f"For algorithms in {str(algs_need_last_global)}, the 'last_global' should not be NoneType."
    if alg == 'scaffold':
        assert (c_local is not None) and (c_global is not None), f"For SCAFFOLD, the c parameters should not be NoneType."
    if alg == 'moon':
        assert (previous_local is not None), f"For MOON, the previous local model must be provided."

    steps = len(x) // batch_size
    shuffle_index = np.random.permutation(len(x))
    # x = make_dataset(x)
    x = x[shuffle_index]
    y = y[shuffle_index]
    cu_loss = 0
    total_step = 0
    tmp_losses = []
    prox_model = copy.deepcopy(model)
    for i in range(steps):
        batch_x = torch.tensor(x[i*batch_size:(i + 1) * batch_size]).unsqueeze(1)
        batch_y = torch.tensor(y[i*batch_size:(i + 1) * batch_size]).unsqueeze(1)
        y_pred = model(batch_x)
        loss = critereon(y_pred, batch_y)
        if alg == 'fedprox':
            reg_loss = (0.01 / 2) * torch.norm(model.coef_.weight - last_global) ** 2
            loss += reg_loss
        if alg == 'moon':
            if type(previous_local) == type(-1):
                previous_local = last_global
            prox_model.coef_.weight.data = previous_local
            prev_pred = prox_model(batch_x)
            prox_model.coef_.weight.data = last_global
            glob_pred = prox_model(batch_x)
            sim_g = (y_pred - glob_pred) ** 2
            sim_p = (y_pred - prev_pred) ** 2
            tau = 10                        
            sim_g = torch.exp((1 - (sim_g - sim_g.min()) / (sim_g.max() - sim_g.min() + 1e-7)) / tau)
            sim_p = torch.exp((1 - (sim_p - sim_p.min()) / (sim_p.max() - sim_p.min() + 1e-7)) / tau)
            reg_loss = - torch.log(sim_g / (sim_g + sim_p)).mean()
            loss += reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3, norm_type=2)
        optimizer.step()
        if alg == 'scaffold':
            model.coef_.weight.data -= 1e-2 * (c_global - c_local)
        cu_loss += loss.data.item()
        total_step += 1
        tmp_losses.append(cu_loss / total_step)
    if alg == 'scaffold':
        return tmp_losses, total_step
    else:
        return tmp_losses


def train_fl(n, l, t, alg='fedavg'):
    assert alg in ['fedavg', 'fedprox', 'scaffold', 'moon', 'ours'], "Only support the algorithms: ['fedavg', 'fedprox', 'scaffold', 'moon', 'ours']."
    server_parameters = model.coef_.weight.data.clone()
    fig, ax = pplot.subplots(figsize=(8,6))
    (draw_curve(ax, *server_parameters.tolist()[0], color='black'))
    pplot.show(fig)
    client_x = [client_1_x, client_2_x, client_3_x]
    client_y = [client_1_y, client_2_y, client_3_y]
    # client_x = [client_3_x, client_3_x, client_3_x]
    # client_y = [client_3_y, client_3_y, client_3_y]
    colors = ['r', 'g', 'b']
    fig, ax = pplot.subplots(nrows=2, ncols=5, figsize=(30, 10))
    if alg == 'scaffold':
        c_global = model.coef_.weight.data.clone().zero_()
        c_locals = [c_global.clone() for _ in range(n)]
    for turn in range(t):
        if alg == 'moon':
            try:
                prev_client_parameters = copy.deepcopy(client_parameters)
            except:
                prev_client_parameters = [-1 for _ in range(n)]
        client_parameters = []
        client_losses = [[] for _ in range(n)]
        if alg == 'scaffold':
            total_delta = 0
        for client in range(n):

            model.coef_.weight.data.copy_(server_parameters)
            coef = model.coef_.weight.tolist()[0]

            if alg == 'scaffold':
                local_steps = 0
            for epoch in range(l):
                if alg == 'scaffold':
                    local_losses, steps = train_one_epoch(client, client_x[client], client_y[client], batch_size, alg, server_parameters, c_global, c_locals[client])
                    local_steps += steps
                elif alg == 'moon':
                    local_losses = train_one_epoch(client, client_x[client], client_y[client], batch_size, alg, server_parameters, previous_local=prev_client_parameters[client])
                else:
                    local_losses = train_one_epoch(client, client_x[client], client_y[client], batch_size, alg, server_parameters)
                client_losses[client] += local_losses
            if alg == 'scaffold':
                c_new_para = (c_locals[client] - c_global + (server_parameters - model.coef_.weight.data) / (local_steps * 0.1)).detach().clone()
                total_delta += c_new_para - c_locals[client]
                c_locals[client] = c_new_para

            coef = model.coef_.weight.tolist()[0]

            draw_curve(ax[turn//5][turn%5], *coef, color=colors[client])
            client_parameters.append(model.coef_.weight.data.clone() * client_weight[client])

        if alg == 'scaffold':
            total_delta /= 300 # i.e. N
            c_global += total_delta

        server_parameters = sum(client_parameters)
        scheduler.step()
        draw_curve(ax[turn//5][turn%5], *server_parameters.tolist()[0], color='black')
        for loss_pairs in zip(*client_losses):
            losses.append(sum(loss_pairs) / len(loss_pairs))

    pplot.show(fig)
    return server_parameters
   
model = PolyNet()
critereon = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
batch_size = 10
losses = []

server_parameters = train_fl(3, 4, 10)
pplot.plot(list(range(len(losses))), losses)
pplot.ylim([0,2000])




model = PolyNet()
critereon = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
batch_size = 10
losses = []

train_fl(3, 4, 10, 'fedprox')

pplot.plot(list(range(len(losses))), losses)
pplot.ylim([0,2000])


model = PolyNet()
critereon = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
batch_size = 10
losses = []

server_parameters = train_fl(3, 4, 10, 'scaffold')

pplot.plot(list(range(len(losses))), losses)
pplot.ylim([0,2000])


model = PolyNet()
critereon = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
batch_size = 10
losses = []

train_fl(3, 4, 10, 'moon')

pplot.plot(list(range(len(losses))), losses)
pplot.ylim([0,2000])


import math

EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)


def calc_template(a, b, c, d):
    return (((a + b) * c) - (b * d))


def calc_statistics(x, y, a, b):
    ksei_square = math.pi / 8
    nu = a * (x + b)
    de = torch.sqrt(1 + ksei_square * a * a * y)

    return torch.sigmoid(nu / (de + EPS))


def get_ep_prior(theta_m, theta_s, fx):
    fx = fx.mean(0).squeeze()
    fxsq = fx * fx

    cm, cs = fx * theta_m, fxsq * theta_s
    # cb, cd = cm / (cs + EPS), -0.5 / (cs + EPS)
    e_1 = calc_statistics(cm, cs, 1, 0)
    e_2 = calc_statistics(cm, cs, 4 - 2 * SQRT2, math.log(SQRT2 + 1))
    e_3 = calc_statistics(cm, cs, 6 * (1 - 1 / POW2_3O2), math.log(POW2_3O2 - 1))

    _p_1 = calc_template(cm, cs, e_1, e_2)
    _p_2 = calc_template(cm, 2 * cs, e_2, e_3)
    s_0 = e_1
    s_1 = _p_1 / (s_0 * fx + EPS)
    s_2 = (cs * e_1 + calc_template(cm, cs, _p_1, _p_2)) / (s_0 * fxsq + EPS)


    # cb, cd = s_1 / (s_2 - s_1 * s_1), -1 / (2 * (s_2 - s_1 * s_1))
    theta_m, theta_s = s_1, torch.relu(s_2 - s_1 * s_1) + EPS

    try:
        assert theta_s.min() >= 0, "negative value found in theta_s"
    except:
        exit(0)

    del cm, cs, e_1, e_2, e_3, s_0, s_1, s_2
    return theta_m, theta_s



def merge_ep_prior_and_get_cavity(client_m, client_s):
    lmd = [(1/(s+EPS)) for s in client_s]
    theta_s = 1/(sum(lmd)+EPS)
    theta_m = theta_s * sum([m*l for m,l in zip(client_m, lmd)]) 
    cavities = []
    for i in range(len(client_m)):
        rest_lmd = lmd[:i] + lmd[i+1:]
        rest_m = client_m[:i] + client_m[i+1:]
        c_s = (1 / (sum(rest_lmd) + EPS))
        c_m = (c_s * sum([m*l for m,l in zip(rest_m, rest_lmd)]))
        cavities.append((c_m, c_s))
    return theta_m, theta_s, cavities

class NPNLinearLite(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dual_input = True, init_type = 0):
        super(NPNLinearLite, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dual_input = dual_input

        self.W_m = torch.nn.Parameter(2 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * (torch.rand(in_channels, out_channels) - 0.5))
        self.bias_m = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.dual_input:
            x_m, x_s = x
        else:
            x_m = x
            x_s = x.clone()
            x_s = 0 * x_s

        o_m = torch.mm(x_m.unsqueeze(0), self.W_m)
        o_m = o_m + self.bias_m.expand_as(o_m)

        o_s = torch.mm(x_s.unsqueeze(0), self.W_m * self.W_m)

        return o_m.squeeze(0), o_s.squeeze(0)

class NPNSigmoid(torch.nn.Module):
    def __init__(self):
        super(NPNSigmoid, self).__init__()
        self.xi_sq = math.pi / 8
        self.alpha = 4 - 2 * math.sqrt(2)
        self.beta = - math.log(math.sqrt(2) + 1)

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = torch.sigmoid(o_m / (1 + self.xi_sq * o_s) ** 0.5)
        a_s = torch.sigmoid(self.alpha * (o_m + self.beta) / (1 + self.xi_sq * self.alpha ** 2 * o_s) ** 0.5) - a_m ** 2
        return a_m, a_s

class PolyNet(torch.nn.Module):
    def __init__(self):
        super(PolyNet, self).__init__()
        self.coef_ = torch.nn.Linear(4, 1, bias=False)
        self.f = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            torch.nn.ReLU(),
            torch.nn.Linear(1,1)
        )
        self.decoder = torch.nn.Sequential(
            NPNLinearLite(1, 4, True),
            NPNSigmoid(),
            NPNLinearLite(2, 4, True)
        )

        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
        

    def forward(self, x):
        construct_input = torch.cat([x*x*x, x*x, x, torch.ones(x.size(0), 1)], dim=1)
        # construct_input = torch.cat([x, torch.ones(x.size(0),1)], dim=1)
        return self.coef_(construct_input)

    def aux_f(self, x, y):
        return torch.tanh(self.f((self.forward(x) - y) ** 2))
    
    def decode(self, theta_m, theta_s):
        return self.decoder((theta_m, theta_s))

def remove_cavity(tm, ts, cm, cs):
    tb = tm / ts
    td = -0.5 / ts
    cb = cm / cs
    cd = -0.5 / cs
    qb = tb - cb
    qd = (td - cd)
    assert qd < 0, "gg"
    qs = - 2  / qd
    qm = qb * qs
    return qm, qs


def make_dataset(x):
    x = torch.tensor(x).unsqueeze(1)
    xx = x * x
    xxx = xx * x 
    # xxxx = xxx * x
    X = torch.cat([xxx, xx, x], dim=1)
    X_min = torch.min(X, dim=0).values.expand(X.size())
    X_max = torch.max(X, dim=0).values.expand(X.size())
    X = (X - X_min) / (X_max - X_min)
    return X


# model = PolyNet()
# critereon = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
batch_size = 10
losses = []


def train_one_epoch(x, y, batch_size, alg='fedavg', last_global=None, c_local=None, c_global=None, previous_local=None, c_m=None, c_s=None, theta_m=None, theta_s=None):
    algs_need_last_global = ['fedprox', 'scaffold', 'moon']
    if alg in algs_need_last_global:
        assert last_global is not None, f"For algorithms in {str(algs_need_last_global)}, the 'last_global' should not be NoneType."
    if alg == 'scaffold':
        assert (c_local is not None) and (c_global is not None), f"For SCAFFOLD, the c parameters should not be NoneType."
    elif alg == 'moon':
        assert (previous_local is not None), f"For MOON, the previous local model must be provided."
    elif alg == 'ours':
        assert (
            c_m is not None and
            c_s is not None and 
            theta_m is not None and 
            theta_s is not None
        ), f"need EP priors."

    steps = len(x) // batch_size
    shuffle_index = np.random.permutation(len(x))
    # x = make_dataset(x)
    x = x[shuffle_index]
    y = y[shuffle_index]
    cu_loss = 0
    total_step = 0
    tmp_losses = []
    prox_model = copy.deepcopy(model)
    for i in range(steps):
        batch_x = torch.tensor(x[i*batch_size:(i + 1) * batch_size]).unsqueeze(1)
        batch_y = torch.tensor(y[i*batch_size:(i + 1) * batch_size]).unsqueeze(1)
        y_pred = model(batch_x)
        loss = critereon(y_pred, batch_y)
        if alg == 'fedprox':
            reg_loss = (0.01 / 2) * torch.norm(model.coef_.weight - last_global) ** 2
            loss += reg_loss
        if alg == 'moon':
            if type(previous_local) == type(-1):
                previous_local = last_global
            prox_model.coef_.weight.data = previous_local
            prev_pred = prox_model(batch_x)
            prox_model.coef_.weight.data = last_global
            glob_pred = prox_model(batch_x)
            sim_g = (y_pred - glob_pred) ** 2
            sim_p = (y_pred - prev_pred) ** 2
            tau = 1                        
            sim_g = torch.exp((1 - (sim_g - sim_g.min()) / (sim_g.max() - sim_g.min() + 1e-7)) / tau)
            sim_p = torch.exp((1 - (sim_p - sim_p.min()) / (sim_p.max() - sim_p.min() + 1e-7)) / tau)
            reg_loss = - torch.log(sim_g / (sim_g + sim_p)).mean()
  
            loss += reg_loss
        if alg == 'ours':
            fx = model.aux_f(batch_x, batch_y) 
            tm, ts = get_ep_prior(c_m.detach().clone(), c_s.detach().clone(), fx)
            sample_m, sample_s = model.decode(tm, ts)

            kld_loss = torch.mean(-0.5 * ((1 + torch.log(ts) - tm ** 2 - ts)), dim = 0)
            ep_loss = ((model.coef_.weight - sample_m) ** 2 / (2 * sample_s)).mean() + kld_loss + 0.5 * sample_s.mean()
            loss += 10 * ep_loss
        loss.backward()
        optimizer.step()
        if alg == 'scaffold':
            model.coef_.weight.data -= 1e-2 * (c_global - c_local)

        cu_loss += loss.data.item()
        total_step += 1
        tmp_losses.append(cu_loss / total_step)

    if alg == 'scaffold':
        return tmp_losses, total_step
    elif alg == 'ours':
        _fx = []
        for i in range(steps):
            batch_x = torch.tensor(x[i*batch_size:(i + 1) * batch_size]).unsqueeze(1)
            batch_y = torch.tensor(y[i*batch_size:(i + 1) * batch_size]).unsqueeze(1)
            _fx.append(model.aux_f(batch_x, batch_y).detach().clone())
        fx = torch.cat(_fx, dim=0)
        tm, ts = get_ep_prior(c_m.detach().clone(), c_s.detach().clone(), fx)
        return tmp_losses, tm.detach().clone(), ts.detach().clone()
    else:
        return tmp_losses


def train_fl(n, l, t, alg='fedavg'):
    assert alg in ['fedavg', 'fedprox', 'scaffold', 'moon', 'ours'], "Only support the algorithms: ['fedavg', 'fedprox', 'scaffold', 'moon', 'ours']."
    server_parameters = model.coef_.weight.data.clone()
    fig, ax = pplot.subplots(figsize=(8,6))



    (draw_curve(ax, *server_parameters.tolist()[0], color='black'))
    pplot.show(fig)
    client_x = [client_1_x, client_2_x, client_3_x]
    client_y = [client_1_y, client_2_y, client_3_y]
    colors = ['r', 'g', 'b']
    fig, ax = pplot.subplots(nrows=2, ncols=5, figsize=(30, 10))
    fig2, ax2 = pplot.subplots(2, 5, figsize=(30,10))
    if alg == 'scaffold':
        c_global = model.coef_.weight.data.clone().zero_()
        c_locals = [c_global.clone() for _ in range(n)]
    elif alg == 'ours':
        global_m = torch.zeros(1)
        global_s = torch.ones(1)
        cavities = [(global_m.clone(), global_s.clone()) for _ in range(n)]
    else:
        pass
    for turn in range(t):
        if alg == 'moon':
            try:
                prev_client_parameters = copy.deepcopy(client_parameters)
            except:
                prev_client_parameters = [-1 for _ in range(n)]
        elif alg == 'ours':
            tm = global_m.clone()
            ts = global_s.clone()
            client_m = []
            client_s = []
        else:
            pass
        client_parameters = []
        client_losses = [[] for _ in range(n)]
        if alg == 'scaffold':
            total_delta = 0
        for client in range(n):
            model.coef_.weight.data.copy_(server_parameters)
            coef = model.coef_.weight.tolist()[0]
            if alg == 'scaffold':
                local_steps = 0
            for epoch in range(l):
                if alg == 'scaffold':
                    local_losses, steps = train_one_epoch(client_x[client], client_y[client], batch_size, alg, server_parameters, c_global, c_locals[client])
                    local_steps += steps
                elif alg == 'moon':
                    local_losses = train_one_epoch(client_x[client], client_y[client], batch_size, alg, server_parameters, previous_local=prev_client_parameters[client])
                elif alg == 'ours':
                    c_m, c_s = cavities[client]
                    local_losses, tm, ts = train_one_epoch(client_x[client], client_y[client], batch_size, alg, server_parameters, theta_m=tm, theta_s=ts, c_m=c_m, c_s=c_s)
                else:
                    local_losses = train_one_epoch(client_x[client], client_y[client], batch_size, alg, server_parameters)
                client_losses[client] += local_losses
            if alg == 'scaffold':
                c_new_para = (c_locals[client] - c_global + (server_parameters - model.coef_.weight.data) / (local_steps * 0.1)).detach().clone()
                total_delta += c_new_para - c_locals[client]
                c_locals[client] = c_new_para
            elif alg == 'ours':
                client_m.append(tm.clone())
                client_s.append(ts.clone())
            else:
                pass
            coef = model.coef_.weight.tolist()[0]
            draw_curve(ax[turn//5][turn%5], *coef, color=colors[client])

            client_parameters.append(model.coef_.weight.data.clone() * client_weight[client])

        if alg == 'scaffold':
            total_delta /= 300 # i.e. N
            c_global += total_delta
        elif alg == 'ours':
            global_m, global_s, cavities = merge_ep_prior_and_get_cavity(client_s=client_s, client_m=client_m)
        else:
            pass
        server_parameters = sum(client_parameters)
        scheduler.step()
        draw_curve(ax[turn//5][turn%5], *server_parameters.tolist()[0], color='black')
        for loss_pairs in zip(*client_losses):
            losses.append(sum(loss_pairs) / len(loss_pairs))

        
        ax2[turn//5][turn%5].set_title(f'{turn}')
        (draw_curve(ax2[turn//5][turn%5], *model.decode(tm, ts)[0].tolist(), color='black'))
    pplot.show(fig)
    return server_parameters


model = PolyNet()
critereon = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
optimizer = torch.optim.RMSprop([
        {'params': model.coef_.parameters(), 'lr': 1e-2},
        {'params': model.f.parameters(), 'lr': 0.0015},
        {'params': model.decoder.parameters(), 'lr': 0.0015},
    ], lr=1e-2)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
batch_size = 10


server_parameters = train_fl(3, 4, 10, 'ours')

pplot.plot(list(range(len(losses))), losses)
pplot.ylim([0,2000])

