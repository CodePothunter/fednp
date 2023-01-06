#!/usr/bin/python3


import json
import sys
import numpy
import kaldiIO
from signal import signal, SIGPIPE, SIG_DFL
#from train_sru_nolap_12layer_1024_fbank import lstm
import torch
import torch.nn as nn
from lib.ops import Dense
#from lib.ops  import SpeechNet_Sru_Test as SpeechNet
#import lib as mylib
import time

from sru import SRU
##!!! please modify these hyperprameters manually
import warnings

warnings.filterwarnings('ignore')
#print("========================================================")




if __name__ == '__main__':
    model = sys.argv[1]
    D = sys.argv[2]
    priors = sys.argv[3]
    # the default time step is 20
    #spliceSize = int(sys.argv[3])
    cfgFile = sys.argv[5]
    mfccDim = 40

    with open(cfgFile, 'r') as load_f:
        learning = json.load(load_f)

    feaDim = (learning['left'] + learning['right'] + 1) * mfccDim
    disDim = learning['disDim']

    #print("======================================================================================================")
    #print("======================================================================================================")
    if not model.endswith('.pth'):
        raise TypeError ('Unsupported model type. Please use pth format.')
    
    #learning=[]

    class SRU_FedNP(nn.Module):
        def __init__(self, input_size=feaDim, hidden_size=1024, output_size=1095, 
                    num_layers=learning['layerNum']):
            super(SRU_FedNP, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.Dense_layer1 = Dense(input_size,self.hidden_size)

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
            x = self.Dense_layer3(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.Dense_layer4(x).view(-1, self.output_size)
            x = torch.softmax(x, dim=1)

            return x, hidden_after

    

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


    cal_model = SRU_FedNP(
                    input_size=feaDim, 
                    hidden_size=learning['hiddenDim'],
                    output_size=learning['targetDim'], 
                ).cuda()

    #inputs_variable = T.fmatrix(name="inputs variable")
    #t_c0 = T.tensor3(name="init_c0_test",dtype=theano.config.floatX)
    cal_model.load_state_dict(torch.load(model))
    cal_model.eval()
    ##load the model for testing

    count = 0
    while uttId:
        count += 1
        #start_time = time.time()
        featMat = torch.from_numpy(featMat).unsqueeze(0).cuda()
        b, t, _ = featMat.size()
        h = torch.zeros(learning['layerNum'], 1, learning['hiddenDim']).cuda()
        with torch.no_grad():
            output , _ = cal_model(featMat, h)
            #output = cal_model(featMat)
        pred = output.data.cpu().numpy()
        pred = numpy.where(pred > 0 ,pred,  1e-5 )
        logProbMat = numpy.log (pred / p)
        logProbMat [logProbMat == -numpy.inf] = -100
        row = logProbMat.flatten().shape[0]//learning['targetDim']
        logProbMat = numpy.reshape(logProbMat,(row,learning['targetDim']))
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance(arkIn)



