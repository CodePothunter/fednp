#!/usr/bin/python3





import subprocess
from subprocess import Popen, PIPE
import tempfile
import kaldiIO
import pickle
import numpy
import os
import shutil
import torch.utils.data as data
import torch

## Data generator class for Kaldi

class dataGenSequences(data.Dataset):

    def __init__(self, data, ali, exp, batch_size=40, timeSteps=20, inputDim=195, left=0, right=0, my_sess=None):
        self.data = data
        self.ali = ali
        self.exp = exp
        self.batch_size=batch_size

        self.lable_list = [0]
        self.left = left
        self.right = right
        self.timeSteps = timeSteps

        ## Number of utterances loaded into RAM.
        ## Increase this for speed, if you have more memory.
        self.maxSplitDataSize = 1000

        ## Parameters for initialize the iteration
        self.item_counter = 0
        self.timeSteps_Num = 0


        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
        #aliPdf = '/home/glen/alipdf.txt'
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()
        if my_sess:
            my_sess = '|'.join(my_sess)
            os.system(f'cat {aliPdf} | grep -E \'{my_sess}\' > {aliPdf}_')
            os.system(f'mv {aliPdf}_ {aliPdf}')
        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)

        ## Determine the number of steps
        ## need to re calculate The last patch will be deleted


        self.numSteps = -(-self.numFeats // ( self.timeSteps))
      
        self.inputFeatDim = inputDim ## IMPORTANT: HARDCODED. Change if necessary.
        self.singleFeatDim = inputDim//(1+self.left+self.right)
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        #print out the configuration
        print ("NumFeats:%d"%(self.numFeats))
        print("NumSteps:%d" % (self.numSteps))
        print ("FeatsDim:%d"%(self.inputFeatDim))
        print ("TimeSteps:%d"%(self.timeSteps))
        print("OutputFeatDim:%d"%(self.outputFeatDim))
        
        self.x = numpy.empty ((0, self.inputFeatDim), dtype=numpy.float32)
        self.y = numpy.empty (0, dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        self.f = numpy.empty(0, dtype=numpy.uint16)
        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)
        print("numUtterances:%d"%(self.numUtterances))
        print("numSplit:%d" % (self.numSplit))

        ## Split data dir per utterance (per speaker split may give non-uniform splits)
        if os.path.isdir (data + 'split' + str(self.numSplit)):
            shutil.rmtree (data + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()
        # print(labels)
        ## Save split labels and delete label
       # print(labels.keys())
        self.splitSaveLabels(labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
        
    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['am-info', '%s/final.mdl' % self.exp], stdout=PIPE)
        modelInfo = p1.stdout.read().splitlines()
        for line in modelInfo:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        FilledNumFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1

            if (len(line)-1)%self.timeSteps!=0:
                FilledNumFeats += (self.timeSteps -(len(line)-1)%self.timeSteps) 
            
            labels[line[0]] = numpy.array([int(i) for i in line[1:]], dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        return labels, numFeats+FilledNumFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + 'utt/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        # print(uid)
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                #print(self.data + '/split' + str(self.numSplit) + 'utt/' + str(sdc) + '/utt2spk', len(splitLabels))
                pickle.dump (splitLabels, f)


    ## Return split of data to work on
    ## There
    def getNextSplitData (self):
        # print(self.splitDataCounter)
        # print(' '.join(['apply-cmvn','--print-args=false','--norm-vars=true',
        #          '--utt2spk=ark:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/utt2spk',
        #          'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/cmvn.scp',
        #          'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/feats.scp','ark:-']))
        # print(' '.join(['splice-feats','--print-args=false','--left-context='+str(self.left),'--right-context='+str(self.right),'ark:-','ark:-']))
        p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE, stderr=subprocess.DEVNULL)
        #print("Here is the p1 stdout")
        #print(p1.stdout.readlines())
        p2 = Popen (['splice-feats','--print-args=false','--left-context='+str(self.left),'--right-context='+str(self.right),'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        
        p1.stdout.close()	
        

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
        #print(labels)
        featList = []
        labelList = []
        flaglist = []
        while True:
            #print(p2.stdout)
            uid, featMat = kaldiIO.readUtterance (p2.stdout)
            #print(uid, labels, len(featList), len(labelList), len(flaglist))
            #print("=====111111111111111111111===================")
            #print(uid)
            #print(featMat)
            if uid == None:
                self.lable_list = labelList
                return (numpy.vstack(featList), numpy.hstack(labelList), numpy.hstack(flaglist))
            if uid in labels:
                row,col = featMat.shape
                fillNum = self.timeSteps - (row % self.timeSteps)
                fillRight = fillNum//2
                fillLeft = fillNum - fillRight
                featMat = numpy.concatenate([numpy.tile(featMat[0],(fillLeft,1)), featMat, numpy.tile(featMat[-1],(fillRight,1))])
                labels4uid = labels[uid]
                labels4uid = numpy.concatenate([numpy.tile(labels4uid[0],(fillLeft,)), labels4uid, numpy.tile(labels4uid[-1],(fillRight,))])
                flags4uid = numpy.zeros(labels4uid.shape)
                flags4uid[-1] = 1
                flaglist.append(flags4uid)
                featList.append (featMat)
                labelList.append (labels4uid)


    def __len__(self):
        return self.numSteps


    def __getitem__(self, item):

        while (self.item_counter >= self.timeSteps_Num):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True

                # return the last group of data, may repeated several times but not matter
                return (self.xMini,self.yMini)
                # break

            self.splitDataCounter += 1
            x, y, f = self.getNextSplitData()
            #print("=======================")
            #print(x.shape)
            #print(y.shape)
            self.split_counter = 0

            self.batchPointer = len(self.x) - len(self.x) % self.timeSteps
            self.timeSteps_Num = self.batchPointer//self.timeSteps
            self.x = numpy.concatenate((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate((self.y[self.batchPointer:], y))
            self.f = numpy.concatenate((self.f[self.batchPointer:], f))
            self.item_counter = 0
            self.batchnum = (len(self.x) - len(self.x) % (self.timeSteps)) // (self.timeSteps * self.batch_size)


            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False

        item = item % ((len(self.x) - len(self.x) % self.timeSteps)//self.timeSteps)
        item = (item % self.batch_size) * self.batchnum + (item // self.batch_size)

        self.xMini = self.x[item * self.timeSteps:item * self.timeSteps +  self.timeSteps]
        self.yMini = self.y[item * self.timeSteps:item * self.timeSteps +  self.timeSteps]
        self.fMini = self.f[item * self.timeSteps:item * self.timeSteps + self.timeSteps]
        self.item_counter += 1

        self.xMini = torch.from_numpy(self.xMini)
        self.yMini = self.yMini.astype(numpy.int16)
        self.yMini = torch.from_numpy(self.yMini)
        self.fMini = self.fMini.astype(numpy.int16)
        self.fMini = torch.from_numpy(self.fMini)


        return (self.xMini,self.yMini)



