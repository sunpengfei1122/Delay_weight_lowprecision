import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../../src")
import torch.nn.functional as F
import  os
import  h5py
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torch.utils import data

from utils import get_shd_dataset
from datetime import datetime
import numpy as np
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from spikeLoss import spikeLoss, gradLog
import matplotlib.cm as cm
# Dataset definition

import os
from torch.utils import data
from torchvision import datasets
from torch.utils.data import Dataset
import torch
import numpy as np

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

class shdDataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.samples = np.loadtxt(self.data_paths+'.txt').astype('int')
        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):  
        inputindex = self.samples[index, 0]
        label = self.samples[index, 1]
        path = self.data_paths + '/'+str(inputindex.item()) + '.npy'
        input = torch.from_numpy(np.load(path))
        label = torch.from_numpy(np.asarray(label))
        #if self.transform:
            
        #    x = self.transform(x)
        emptytensor = torch.zeros((1,1,700,350))
        yy = input.transpose(1,0).nonzero()
        emptytensor[0,0,yy[:,0],yy[:,1]]  = 1
        
        desiredClass = torch.zeros((20, 1, 1, 1))
        desiredClass[label,...] = 1
        return emptytensor, desiredClass, label







def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha1, beta):
      if k == 32:
        out = input
      elif k == 1:

        delta = 0.7 * torch.mean(torch.abs(input)).detach()
        ctx.save_for_backward(input, alpha1, beta,delta)
        alpha = torch.mean(torch.abs(input)).detach()
        # learnable tenary weights
        out = torch.zeros_like(input)
        out[input > delta] = alpha1
        out[input < -delta] = -beta
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      input, alpha, beta,delta = ctx.saved_tensors
      grad_input = grad_output.clone()
      indicator_pos = (input > delta).type_as(grad_output)
      indicator_neg = (input < -delta).type_as(grad_output)

      grad_alpha = torch.sum(grad_output * indicator_pos) 
      grad_beta  = -torch.sum(grad_output * indicator_neg) 

      return grad_input, grad_alpha, grad_beta

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)
    self.alpha = nn.Parameter(torch.tensor(1.0))
    self.beta  = nn.Parameter(torch.tensor(1.0))
    self.alpha_initialized = False
  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      
      alpha1 = F.softplus(self.alpha )
      beta1 = F.softplus(self.beta )
      weight_q = self.uniform_q(x, alpha1, beta1 ) 
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    
    return weight_q

class Network1(torch.nn.Module):
    def __init__(self, netParams):
        super(Network1, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        self.preHookFx1 = weight_quantize_fn(w_bit=1)
        self.preHookFx2 = weight_quantize_fn(w_bit=1)
        # define network functions
        self.fc1   = torch.nn.utils.weight_norm(slayer.dense( 700, 128, preHookFx= self.preHookFx1), name='weight')
        self.fc2   = torch.nn.utils.weight_norm(slayer.dense(128, 128, preHookFx= self.preHookFx2), name='weight')
        self.fc3   = torch.nn.utils.weight_norm(slayer.dense(128,  20), name='weight')

        self.delay1 = slayer.delay_minmax(128, step=1)  #learnable lowest delay th, the quantized  delay will be [th, win+th, 2*win_th]  
        #self.delay1_1 = slayer.delay(128)   # non learnable, the quantized delay will be [o, win, 2*win], win is the step window(defined in /src/slayer)
        self.delay2 = slayer.delay_minmax(128,step =1) £ step can be 1,2,3,4,5,10,quantize the delays
        #self.delay2_1 = slayer.delay(128)

    def forward(self, spike):
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
        spike = self.slayer.spike(self.fc1(self.slayer.psp(spike)))
        spike1 = spike

        spike = self.delay1(spike)

        spike1_1 = spike
        spike = self.slayer.spike(self.fc2(self.slayer.psp(spike)))

        spike2 = spike

        spike = self.delay2(spike)

        spike = self.slayer.spike(self.fc3(self.slayer.psp(spike)))

        return spike, spike1,spike1_1, spike2, spike2

    def clamp(self):
        self.delay1.mind.data.clamp_(0) #causal
        self.delay2.mind.data.clamp_(0)
        self.delay1.delay.data.clamp_(0)
        self.delay2.delay.data.clamp_(0)
                

    def gradFlow(self, path):
        gradNorm = lambda x: torch.norm(x).item()/torch.numel(x)

        grad = []
        grad.append(gradNorm(self.fc1.weight_g.grad))  #check the gradient
        grad.append(gradNorm(self.fc2.weight_g.grad))
        grad.append(gradNorm(self.fc3.weight_g.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('-gpu', type=int, default=[0] )
    parser.add_argument('-seed', type=int, default=2256 )
    parser.add_argument('-f', type=str, default='delay0_win10' )    
    parser.add_argument('-l', type=str, default='delay0_win10' )
    parser.add_argument('-lr1', type=str, default=0.0 )   #penalty term for the first layer
    parser.add_argument('-lr2', type=str, default=0.0 )   #penalty term for the second layer
    args = parser.parse_args()
    trainedFolder = args.f
    #'Trained01'
    logsFolder    = args.l


    os.makedirs(trainedFolder, exist_ok=True)
    os.makedirs(logsFolder   , exist_ok=True)
    
    # Read network configuration
    netParams = snn.params('network.yaml')
    
    # Define the cuda device to run the code on. 
    device = torch.device('cuda')

    # define network instance
    net = Network1(netParams).to(device)
    module = net
    
    # # Use multiple GPU's, uncomment it
    #deviceIds = [0, 1, 2,3]
    #net = torch.nn.DataParallel(Network1(netParams).to(device), device_ids=deviceIds)
    #module = net.module


    # Create snn loss instance.
    error = spikeLoss(netParams).to(device)

    # Define optimizer module
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.5)
    # Dataset and dataLoader instances.
    trainingSet = shdDataset('data/train')
    trainLoader = DataLoader(dataset=trainingSet, batch_size=128, shuffle=True, num_workers=4)


    testingSet = shdDataset('/data/test')
    testLoader = DataLoader(dataset=testingSet, batch_size=128, shuffle=False, num_workers=4)
    
    # Learning stats instance.
    stats = snn.utils.stats()
   
    total = sum([param.nelement() for param in module.parameters()])
    print("Number of parameter: %.2fM" % (total))
    # Main loop
    for epoch in range(1000):
    #for epoch in range(1):
        tSt = datetime.now()    
        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            net.train()
                        
            loss1 = 0
            loss2 = 0
            # Move the input and target to correct GPU.
            input  = input.to(device)
            target = target.to(device) 
            
            # Forward pass of the network.
            output, s1,s11,s2,s22 = net.forward(input)
                    
            # Gather the training stats.
            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)
            for name, parameters in net.named_parameters():

                if name == 'delay1.delay':   # if utilize multiple GPUs, utilize the name 'module.delay1.delay'
                    loss1 += torch.norm(parameters, 2)  # l2 loss, try to minize the values, not to zero.
                elif name == 'delay2.delay':
                    loss2 += torch.norm(parameters, 2)
                else:
                    pass

            

                        
            # Calculate loss.
            loss =  error.probSpikes(output, target) + float(args.lr1)* loss1 + float(args.lr2) * loss2
                       
            # Reset gradients to zero.
            optimizer.zero_grad()
            
            # Backward pass of the network.
            loss.backward()
            
            # Update weights.
            optimizer.step()

            # Clamp delays
            module.clamp()
            #scheduler.step()
            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            stats.print(
                epoch, i, 
            
            )
       
        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
        for i, (input, target, label) in enumerate(testLoader, 0):    
            with torch.no_grad():
                input  = input.to(device)
                target = target.to(device) 

                output, s1,s11,s2,s22 = net.forward(input)
            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

            loss =  error.probSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()            
            stats.print(
                epoch, i,
            )
        
        # Update stats.
        stats.update()
        stats.plot(saveFig=True, path = trainedFolder + '/')
        module.gradFlow(path= trainedFolder + '/')
        if stats.testing.bestAccuracy is True:  torch.save(module.state_dict(), trainedFolder + '/NTIDIGITS.pt')
     
            
        stats.save(trainedFolder + '/')
