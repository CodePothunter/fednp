#!/usr/bin/python3

import json
import sys
import numpy
import kaldiIO
from signal import signal, SIGPIPE, SIG_DFL
#from train_sru_nolap_12layer_1024_fbank import lstm
import torch
import torch.nn as nn
from lib.ops import Dense,UBGGRNN
#from lib.ops  import SpeechNet_Sru_Test as SpeechNet
#import lib as mylib
import time
import warnings
warnings.filterwarnings('ignore')
#print("========================================================")




if __name__ == '__main__':
    model = sys.argv[1]
    priors = sys.argv[2]
    # the default time step is 20
    #spliceSize = int(sys.argv[3])
    cfgFile = sys.argv[4]
    mfccDim = 40

    with open(cfgFile, 'r') as load_f:
        learning = json.load(load_f)

    feaDim = (learning['left'] + learning['right'] + 1) * mfccDim

    #print("======================================================================================================")
    #print("======================================================================================================")
    if not model.endswith('.pth'):
        raise TypeError ('Unsupported model type. Please use pth format.')


    class lstm(nn.Module):
        def __init__(self, batch_size=1, input_size=feaDim, hidden_size=1024, output_size=1095,
                     num_layers=learning['layerNum'], num_layers2=learning['layerNum2'], z_dim=learning['z_dim'],
                     window_size=learning['window_size']):
            super(lstm, self).__init__()

            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_layers2 = num_layers2
            self.window_size = window_size
            self.Dense_layer1 = Dense(input_size, self.hidden_size)

            self.vrnn_layers = UBGGRNN(x_dim=self.hidden_size, h_dim=self.hidden_size, z_dim=z_dim,
                                      n_layers=self.num_layers, n_layers2=self.num_layers2, window_size=window_size,
                                      dropout=0.1)

            self.Dense_layer3 = Dense(self.hidden_size, 1024)
            self.Dense_layer4 = Dense(1024, output_size)
            self.dropout = nn.Dropout(p=0.1)

        def forward(self, x, f, hidden1, hidden2, ADJ_NODE, ADJ_id, current_node):
            b, t, h = x.size()
            x = torch.reshape(x, (b * t, h))
            x = self.Dense_layer1(x)
            # print(f)

            x = self.dropout(x)
            x = torch.reshape(x, (b, t, self.hidden_size))
            hidden1 = torch.reshape(hidden1, (self.num_layers, self.batch_size, self.hidden_size))
            hidden2 = torch.reshape(hidden2, (self.num_layers2, self.batch_size, self.hidden_size))

            hidden1_after = torch.zeros_like(hidden1)
            hidden2_after = torch.zeros_like(hidden2)
            ADJ_NODE_after = torch.zeros_like(ADJ_NODE)
            current_node_after = torch.zeros_like(current_node)

            x = x.permute(1, 0, 2)
            x, hidden1_after, hidden2_after, ADJ_NODE_after, ADJ_id_after, current_node_after, kl_loss1, kl_loss2 = self.vrnn_layers(
                x, f, hidden1, hidden2, ADJ_NODE, ADJ_id, current_node)
            x = x.permute(1, 0, 2)

            b, t, h = x.size()
            x = torch.reshape(x, (b * t, h))
            x = self.Dense_layer3(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.Dense_layer4(x)
            x = torch.softmax(x, dim=1)
            return x,ADJ_NODE_after, ADJ_id_after
    

    p = numpy.genfromtxt (priors, delimiter=',')
    p[p==0] = 1e-5 ## Deal with zero priors

    arkIn = sys.stdin.buffer
    arkOut = sys.stdout.buffer
    encoding = sys.stdout.encoding
    signal (SIGPIPE, SIG_DFL)

    ## Load a feature matrix (utterance)
    uttId, featMat = kaldiIO.readUtterance(arkIn)
    #print("==================================================")
    row,col = featMat.shape
    f = torch.zeros(1,row).cuda()
    f[0][-1] = 1
    #cal_model = lstm(input_size=feaDim, hidden_size=learning['hiddenDim'], output_size=learning['targetDim']).cuda()
    cal_model = lstm(input_size=feaDim, hidden_size=learning['hiddenDim'], output_size=learning['targetDim']).cuda()
    #inputs_variable = T.fmatrix(name="inputs variable")
    #t_c0 = T.tensor3(name="init_c0_test",dtype=theano.config.floatX)
    cal_model.load_state_dict(torch.load(model))
    cal_model.eval()
    ##load the model for testing
    ADJ_node = torch.zeros(1, learning['window_size'], learning['graph_dim']).cuda()
    ADJ_id = []
    for i in range(1):
        ADJ_id.append(0)
    count = 0
    while uttId:
        count += 1
        #start_time = time.time()
        featMat = torch.from_numpy(featMat).unsqueeze(0).cuda()
        #h = torch.zeros(learning['layerNum'], 1, 1, learning['hiddenDim']).cuda()
        h1 = torch.zeros(learning['layerNum'], 1, 1, learning['hiddenDim']).cuda()
        h2 = torch.zeros(learning['layerNum2'], 1, 1, learning['hiddenDim']).cuda()
        current_node = torch.zeros(1, learning['hiddenDim']).cuda()

        with torch.no_grad():
            output,ADJ_node,ADJ_id = cal_model(featMat,f,h1,h2,ADJ_node,ADJ_id,current_node)
            #output = cal_model(featMat)
        pred = output.data.cpu().numpy()
        pred = numpy.where(pred > 0 ,pred,  1e-5 )
        logProbMat = numpy.log (pred / p)
        logProbMat [logProbMat == -numpy.inf] = -100
        row = logProbMat.flatten().shape[0]//learning['targetDim']
        logProbMat = numpy.reshape(logProbMat,(row,learning['targetDim']))
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance(arkIn)



