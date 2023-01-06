
import numpy as np
import numpy

import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
import torch.nn as nn
import math
import time
from sru import SRU
import warnings

warnings.filterwarnings('ignore')
##############################################################################
################implemenations of the basic functionality###################
##############################################################################

def error(y,pred):
    return torch.mean(torch.ne(y,pred))


def accuracy(y,pred):
    return torch.mean(torch.eq(y,pred))

def clip(x,min,max):
    return torch.clamp(x,min,max)


def floor(x):
    return torch.floor(x).int()


def ceil(x):
    return torch.ceil(x).int()

def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return F.relu(x)

def leaky_relu(x,negative_slope):
    return F.leaky_relu(x,negative_slope=negative_slope)


def softplus(x):
    return F.softplus(x)

def softmax(x):
    return F.softmax(x)


def tanh(x):
    return torch.tanh(x)

def l2_norm(x,epsilon = 0.00001):
    square_sum = torch.sum(torch.pow(x,exponent=2))
    norm = torch.sqrt(torch.add(square_sum,epsilon))
    return norm

def l2_norm_2d(x, epsilon = 0.00001):
    square_sum = torch.sum(torch.pow(x,exponent=2))
    norm = torch.mean(torch.sqrt(torch.add(square_sum,epsilon)))

    return norm

# we assume afa=beta
def neg_likelihood_gamma(x, afa ,epsilon = 0.00001):
    #norm = T.maximum(x, epsilon)
    norm = torch.add(x,epsilon)
    neg_likelihood = -(afa-1)*torch.log(norm)+afa*norm
    return  torch.mean(neg_likelihood)

# KL(lambda_t||lambda=1)
def kl_exponential(x, epsilon = 0.00001):
    norm = torch.add(x,epsilon)
    kl = -torch.log(norm)+norm
    return  torch.mean(kl)
 
def likelihood(x,y, epsilon = 0.00001):
    norm = torch.add(x,epsilon)
    kl = -torch.log(norm)+norm*y
    return  0.25*torch.mean(kl)



def shape(x):

    return x.shape

def reshape(x, shape):
    y = torch.reshape(x, shape).float()
    return y




def Linear_Function(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input,weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret



##############################################################################
################implemenations of the Neuro Networks##########################
##############################################################################



class Dense(Module):

    __constants__ = ['bias', 'features', 'features']
    def __init__(self, in_features, out_features, bias = True):
        super(Dense,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        if (self.in_features == self.out_features):
            init.orthogonal_(self.weight)
        else:
            init.uniform_(self.weight,a = -math.sqrt(1.0/self.in_features)*math.sqrt(3), b = math.sqrt(1.0/self.in_features)*math.sqrt(3) )

        #init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            #fam_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1 / math.sqrt(fam_in)
            init.uniform_(self.bias, -0, 0)
    def forward(self, input):
        return Linear_Function(input, self.weight,self.bias)
    '''def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features,self.out_features, self.bias is not None)'''


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers,n_layers_2, bias=True):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_layers_2 = n_layers_2
        # feature-extracting transformations
        self.phi_z = nn.Sequential(
            Dense(z_dim, h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            Dense(h_dim, h_dim),
            nn.ReLU(),
            Dense(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = Dense(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            Dense(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            Dense(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            Dense(h_dim, z_dim),
            nn.Softplus())

        # recurrence
        self.sru1 = SRU(input_size=self.h_dim*2, hidden_size=self.h_dim, num_layers=self.n_layers,dropout=0.1)
        self.sru2 = SRU(input_size=self.h_dim, hidden_size=self.h_dim, num_layers=self.n_layers_2, dropout=0.1,use_tanh=True)

    def forward(self, x, h1, h2):

        #sru_recurrence
        x_input4sru1 = x
        x_input4sru2 = x
        h_after1 = torch.zeros_like(h1)
        h_after2 = torch.zeros_like(h2)
        #before_pooling_hidden = before_pooling_hidden.detach()

        x_input4sru2, h_after2 = self.sru2(x_input4sru2, h2)



        time_step,batch_size,hidden_dim = x.size()

        x_input4sru2 = torch.reshape(x_input4sru2, (time_step * batch_size, hidden_dim))

        # encoder
        enc_t = self.enc(x_input4sru2)
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)

        # prior
        prior_t = self.prior(x_input4sru2)
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        phi_z_t = torch.reshape(self.phi_z(z_t), (time_step , batch_size, hidden_dim))


        x_input4sru1 = torch.cat([phi_z_t, x_input4sru1], 2)
        # decoder
        x_input4sru1, h_after1 = self.sru1(x_input4sru1, h1)


        kld_loss = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)


        #result = torch.reshape(dec_t, (time_step, batch_size, 2 * hidden_dim))

        return  x_input4sru1, h_after1, h_after2, kld_loss


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        return eps.mul(std).add_(mean)
        #return torch.mul(eps,std) + mean

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        #kld_element = (2 * torch.log(std_2+ 0.01) - 2 * torch.log(std_1+0.01) +
        #               ((std_1+0.01).pow(2) + (mean_1 - mean_2).pow(2)) /
        #               ((std_2+0.01).pow(2)) - 1)
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       ((std_2).pow(2)) - 1)

        return 0.5 * torch.sum(torch.abs(kld_element))

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        pass


class RTN(nn.Module):  # samlping only on utterence-level
    def __init__(self, x_dim, h_dim, z_dim, n_layers, n_layers2, window_size, dropout, bias=False):
        super(RTN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.graph_node_dim = 128
        self.n_layers = n_layers
        self.n_layers2 = n_layers2
        self.window_size = window_size
        self.total_window_size = int((window_size * (window_size + 1)) / 2)
        #print(self.total_window_size)
        self.dropout_rate = dropout
        # recurrence
        self.sru1 = SRU(input_size=self.h_dim * 2, hidden_size=self.h_dim, num_layers=self.n_layers, dropout=0.1)
        self.sru2 = SRU(input_size=self.h_dim, hidden_size=self.h_dim, num_layers=self.n_layers2, dropout=0.1)

        # encoder
        self.enc = nn.Sequential(
            Dense(self.graph_node_dim * (self.window_size+1), h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate))
        self.enc_g = nn.Sequential(
            Dense(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.enc_b = nn.Sequential(
            Dense(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.enc_mean = Dense(h_dim, self.total_window_size)
        self.enc_std = nn.Sequential(
            Dense(h_dim, self.total_window_size),
            nn.Softplus())
        self.enc_mean_b = Dense(h_dim, self.total_window_size)
        self.enc_std_b = nn.Sequential(
            Dense(h_dim, self.total_window_size),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            Dense(self.graph_node_dim * (self.window_size+1), h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate))
        self.prior_g = nn.Sequential(
            Dense(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.prior_mean = Dense(h_dim, self.total_window_size)
        self.prior_std = nn.Sequential(
            Dense(h_dim, self.total_window_size),
            nn.Softplus())
        self.prior_b = nn.Sequential(
            Dense(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.prior_lamada_b = nn.Sequential(
            Dense(h_dim, self.total_window_size)
        )

        # graph
        self.node_hidden = nn.Sequential(
            Dense(h_dim, self.graph_node_dim),
            nn.ReLU())
        self.gen_new_node = nn.Sequential(
            Dense(self.graph_node_dim, h_dim),
            nn.ReLU())
        self.gen_new_embediing4graph = nn.Sequential(
            Dense(self.graph_node_dim * 2, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            Dense(self.graph_node_dim, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            Dense(self.graph_node_dim, self.graph_node_dim),
            nn.ReLU()
        )
        self.graph_4_encoder = nn.Sequential(
            Dense(h_dim, self.graph_node_dim),
            nn.ReLU())

    def forward(self, x, f, h1, h2, ADJ_NODE, ADJ_id, before_pooling_hidden):

        # sru_recurrence
        x_input4sru1 = x
        x_input4sru2 = x
        h_after1 = torch.zeros_like(h1)
        h_after2 = torch.zeros_like(h2)


        x_input4sru2, h_after2 = self.sru2(x_input4sru2, h2)
        time_step, batch_size, hidden_dim = x.size()

        # graph
        x_input4sru2 = torch.reshape(x_input4sru2, (time_step, batch_size, self.h_dim)).permute(1, 0, 2)
        x_input4sru2 = torch.cat([x_input4sru2, before_pooling_hidden.unsqueeze(1)], dim=1)

        after_pooling_hidden = torch.nn.functional.adaptive_max_pool2d(input=x_input4sru2, output_size=(1, self.h_dim))
        after_pooling_hidden = torch.reshape(after_pooling_hidden, (batch_size, self.h_dim))

        after_graph_hidden = self.graph_4_encoder(after_pooling_hidden.clone())

        input_4_encoder = torch.cat([ADJ_NODE.clone(), after_graph_hidden.unsqueeze(1)], dim=1)
        input_4_encoder = torch.reshape(input_4_encoder,(batch_size,-1))

        input_4_encoder1 = input_4_encoder.clone()
        input_4_encoder2 = input_4_encoder.clone()

        # prior
        prior_t = self.prior(input_4_encoder1)
        prior_t_g = self.prior_g(prior_t)
        prior_mean_t = self.prior_mean(prior_t_g)
        prior_std_t = self.prior_std(prior_t_g)
        prior_t_b = self.prior_b(prior_t)
        prior_lamada_b = 0.4 * sigmoid(self.prior_lamada_b(prior_t_b))


        # encoder
        enc_t = self.enc(input_4_encoder2)
        enc_t_g = self.enc_g(enc_t)
        enc_t_b = self.enc_b(enc_t)
        enc_mean_t = self.enc_mean(enc_t_g)
        enc_std_t = self.enc_std(enc_t_g)
        enc_mean_t_b = self.enc_mean_b(enc_t_b)
        enc_std_t_b = self.enc_std_b(enc_t_b)



        enc_mean_t_b = softplus(enc_mean_t_b) + 0.01
        enc_k = enc_mean_t_b.mul(2.0 * enc_std_t_b.pow(2))
        enc_lamada_t = (1.0 + enc_k - torch.sqrt(enc_k.pow(2) + 1)) * 0.5


        # sampling and reparameterization
        z_t, z_lamada_t = self._reparameterized_sample4utterence(enc_mean_t, enc_std_t, enc_lamada_t, self.window_size)

        z_lamada_t = softplus(z_lamada_t)
        e_t = torch.relu(z_t).unsqueeze(2)
        #print(e_t)
        after_pooling_z_repeat = self.node_hidden(after_pooling_hidden.clone())
        after_pooling_z_repeat = after_pooling_z_repeat.unsqueeze(1).repeat(1, self.window_size, 1)

        for b in range(batch_size):
            a, t = f.size()
            t = t - 1
            if f[b][t] == 1:
                after_pooling_hidden[b] = torch.zeros_like(after_pooling_hidden[b]).cuda()

        ADJ_NODE_data = ADJ_NODE.clone()
        ADJ_now = torch.cat([after_pooling_z_repeat, ADJ_NODE_data], dim=2)
        ADJ_his = torch.zeros(batch_size,self.total_window_size-self.window_size,self.graph_node_dim*2).cuda()
        for i in range(self.window_size-1):
            ADJ_his[:,int((self.window_size-i-2)*(self.window_size-i-1)/2):int((self.window_size-i)*(self.window_size-i-1)/2),:] \
                = torch.cat([ADJ_NODE_data[:, self.window_size - i -1, :].unsqueeze(1).repeat(1, self.window_size - i -1, 1), ADJ_NODE_data[:,1:self.window_size - i,:]],dim=2)
        ADJ = torch.cat([ADJ_his,ADJ_now],dim=1)
        ADJ = self.gen_new_embediing4graph(torch.reshape(ADJ, (batch_size * self.total_window_size, self.graph_node_dim * 2)))
        ADJ = torch.reshape(ADJ, (batch_size, self.total_window_size, self.graph_node_dim))


        h_ei = torch.sum(torch.mul(e_t, ADJ), dim=1)


        #print(h_ei.size())
        h_vt = self.gen_new_node(h_ei).unsqueeze(0).repeat(time_step, 1, 1)
        x_input4sru1 = torch.cat([h_vt, x_input4sru1], dim=2)
        x_input4sru1, h_after1 = self.sru1(x_input4sru1, h1)

        # decoder
        #print(z_lamada_t)
        kld_loss_term1 = self._kld_gauss(enc_mean_t.mul(z_lamada_t), enc_std_t.mul(z_lamada_t),
                                         prior_mean_t.mul(z_lamada_t), prior_std_t.mul(z_lamada_t))
        kld_loss_term2 = self._kld_gauss_and_bernoulli_term1(enc_lamada_t, prior_lamada_b)

        for b in range(batch_size):
            ADJ_NODE[b, 0:self.window_size - 1, :] = ADJ_NODE[b, 1:self.window_size, :].clone()
            ADJ_NODE[b, -1, :] = h_ei[b]

        return x_input4sru1, h_after1, h_after2, ADJ_NODE, ADJ_id, after_pooling_hidden, kld_loss_term1, kld_loss_term2

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample4utterence(self, mean, std, lamada, window_size):
        """using std to sample"""
        # print(mean)
        # print(std)
        # print(lamada)
        mean_2 = lamada
        std_2 = torch.sqrt(torch.mul(lamada, (1.0 - lamada)))
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        sample = (eps.mul(std).add_(mean)).mul(eps_2.mul(std_2).add_(mean_2))
        sample_lamada = eps_2.mul(std_2).add_(mean_2)
        # print(sample)
        # print(sample_lamada)
        # print('=====================')
        return sample, sample_lamada

    def _reparameterized_sample(self, mean, std, lamada, window_size):
        """using std to sample"""
        d1, d2 = std.size()
        # print(std.size())
        sample = torch.zeros(window_size, d1, d2).cuda()
        for i in range(window_size):
            eps = torch.FloatTensor(std.size()).normal_().cuda()
            eps_2 = torch.rand(lamada.size()).cuda()
            ones = torch.ones(lamada.size()).cuda()
            zeros = torch.zeros(lamada.size()).cuda()
            eps_2 = torch.where(eps_2 >= lamada, eps_2, ones)
            eps_2 = torch.where(eps_2 < lamada, eps_2, zeros)
            sample[i] = eps.mul(std).add_(mean).mul(eps_2)

        return sample

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       ((std_1).pow(2) + (mean_1 - mean_2).pow(2)) /
                       (std_2).pow(2) - 1)
        #kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / ((std_2).pow(2)) - 1)

        return 0.5 * torch.sum(torch.abs(kld_element))

    def _kld_gauss_and_bernoulli_term1(self, lamada_t, prior_lamada):
        """Using std to compute KLD"""

        lamada_0 = prior_lamada
        kld_element_term1 = -(lamada_t * torch.log(lamada_t / lamada_0) + (1 - lamada_t) * torch.log(
            (1 - lamada_t + lamada_t.pow(2) / 2.0) / (1 - lamada_0 + lamada_t.pow(2) / 2.0)))

        return torch.sum(torch.abs(kld_element_term1))

    def _kld_gauss_and_bernoulli_term2(self, mean, std, lamada_t, prior_lamada):
        """Using std to compute KLD"""

        lamada_0 = prior_lamada
        kld_element_term2 = 0.5 * (2.0 * torch.log(10.0 * std) - 100.0 * (std.pow(2) - mean.pow(2)))
        return torch.sum(torch.abs(kld_element_term2))

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        pass



