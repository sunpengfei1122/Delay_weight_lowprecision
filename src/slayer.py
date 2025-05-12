import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
# import slayer_cuda
import slayerCuda
# import matplotlib.pyplot as plt

# # Consider dictionary for easier iteration and better scalability
# class yamlParams(object):
#   '''
#   This class reads yaml parameter file and allows dictionary like access to the members.
    
#   Usage:

#   .. code-block:: python
        
#       import slayerSNN as snn
#       netParams = snn.params('path_to_yaml_file') # OR
#       netParams = slayer.yamlParams('path_to_yaml_file')

#       netParams['training']['learning']['etaW'] = 0.01
#       print('Simulation step size        ', netParams['simulation']['Ts'])
#       print('Spiking neuron time constant', netParams['neuron']['tauSr'])
#       print('Spiking neuron threshold    ', netParams['neuron']['theta'])

#       netParams.save('filename.yaml')
#   '''
#   def __init__(self, parameter_file_path):
#       with open(parameter_file_path, 'r') as param_file:
#           self.parameters = yaml.safe_load(param_file)

#   # Allow dictionary like access
#   def __getitem__(self, key):
#       return self.parameters[key]

#   def __setitem__(self, key, value):
#       self.parameters[key] = value

#   def save(self, filename):
#       with open(filename, 'w') as f:
#           yaml.dump(self.parameters, f)

# class spikeLayer():
class spikeLayer(torch.nn.Module):
    '''
    This class defines the main engine of SLAYER.
    It provides necessary functions for describing a SNN layer.
    The input to output connection can be fully-connected, convolutional, or aggregation (pool)
    It also defines the psp operation and spiking mechanism of a spiking neuron in the layer.

    **Important:** It assumes all the tensors that are being processed are 5 dimensional. 
    (Batch, Channels, Height, Width, Time) or ``NCHWT`` format.
    The user must make sure that an input of correct dimension is supplied.

    *If the layer does not have spatial dimension, the neurons can be distributed along either
    Channel, Height or Width dimension where Channel * Height * Width is equal to number of neurons.
    It is recommended (for speed reasons) to define the neuons in Channels dimension and make Height and Width
    dimension one.*

    Arguments:
        * ``neuronDesc`` (``slayerParams.yamlParams``): spiking neuron descriptor.
            .. code-block:: python

                neuron:
                    type:     SRMALPHA  # neuron type
                    theta:    10    # neuron threshold
                    tauSr:    10.0  # neuron time constant
                    tauRef:   1.0   # neuron refractory time constant
                    scaleRef: 2     # neuron refractory response scaling (relative to theta)
                    tauRho:   1     # spike function derivative time constant (relative to theta)
                    scaleRho: 1     # spike function derivative scale factor
        * ``simulationDesc`` (``slayerParams.yamlParams``): simulation descriptor
            .. code-block:: python

                simulation:
                    Ts: 1.0         # sampling time (ms)
                    tSample: 300    # time length of sample (ms)   
        * ``fullRefKernel`` (``bool``, optional): high resolution refractory kernel (the user shall not use it in practice)  
    
    Usage:

    >>> snnLayer = slayer.spikeLayer(neuronDesc, simulationDesc)
    '''
    def __init__(self, neuronDesc, simulationDesc, fullRefKernel = False):
        super(spikeLayer, self).__init__()
        self.neuron = neuronDesc
        self.simulation = simulationDesc
        self.fullRefKernel = fullRefKernel
        
        # self.srmKernel = self.calculateSrmKernel()
        # self.refKernel = self.calculateRefKernel()
        self.register_buffer('srmKernel', self.calculateSrmKernel())
        self.register_buffer('refKernel', self.calculateRefKernel())
        
    def calculateSrmKernel(self):
        srmKernel = self._calculateAlphaKernel(self.neuron['tauSr'])
        # TODO implement for different types of kernels
        return torch.FloatTensor(srmKernel)
        # return torch.FloatTensor( self._zeroPadAndFlip(srmKernel)) # to be removed later when custom cuda code is implemented
        
    def calculateRefKernel(self):
        if self.fullRefKernel:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'], EPSILON = 0.0001)
            # This gives the high precision refractory kernel as MATLAB implementation, however, it is expensive
        else:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'])
        
        # TODO implement for different types of kernels
        return torch.FloatTensor(refKernel)
        
    def _calculateAlphaKernel(self, tau, mult = 1, EPSILON = 0.01):
        # could be made faster... NOT A PRIORITY NOW
        #print('do we really COme here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        eps = []
        # tauSr = self.neuron['tauSr']
        for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
            epsVal = mult * t / tau * math.exp(1 - t / tau)
            if abs(epsVal) < EPSILON and t > tau:
                break
            eps.append(epsVal)
        return eps
    
    def _zeroPadAndFlip(self, kernel):
        if (len(kernel)%2) == 0: kernel.append(0)
        prependedZeros = np.zeros((len(kernel) - 1))
        return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()
        
    def psp(self, spike):
        '''
        Applies psp filtering to spikes.
        The output tensor dimension is same as input.

        Arguments:
            * ``spike``: input spike tensor.

        Usage:

        >>> filteredSpike = snnLayer.psp(spike)
        '''
        return _pspFunction.apply(spike, self.srmKernel, self.simulation['Ts'])

    def pspLayer(self):
        '''
        Returns a function that can be called to apply psp filtering to spikes.
        The output tensor dimension is same as input.
        The initial psp filter corresponds to the neuron psp filter.
        The psp filter is learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.
        
        Usage:
        
        >>> pspLayer = snnLayer.pspLayer()
        >>> filteredSpike = pspLayer(spike)
        '''
        return _pspLayer(self.srmKernel, self.simulation['Ts'])

    def pspFilter(self, nFilter, filterLength, filterScale=1):
        '''
        Returns a function that can be called to apply a bank of temporal filters.
        The output tensor is of same dimension as input except the channel dimension is scaled by number of filters.
        The initial filters are initialized using default PyTorch initializaion for conv layer.
        The filter banks are learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.
        
        Arguments:
            * ``nFilter``: number of filters in the filterbank.
            * ``filterLength``: length of filter in number of time bins.
            * ``filterScale``: initial scaling factor for filter banks. Default: 1.

        Usage:
        
        >>> pspFilter = snnLayer.pspFilter()
        >>> filteredSpike = pspFilter(spike)
        '''
        return _pspFilter(nFilter, filterLength, self.simulation['Ts'], filterScale)

    def replicateInTime(self, input, mode='nearest'):
        Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
        N, C, H, W = input.shape
        # output = F.pad(input.reshape(N, C, H, W, 1), pad=(Ns-1, 0, 0, 0, 0, 0), mode='replicate')
        if mode == 'nearest':
            output = F.interpolate(input.reshape(N, C, H, W, 1), size=(H, W, Ns), mode='nearest')
        return output
    
    def dense(self, inFeatures, outFeatures, weightScale=10, preHookFx=None):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _denseLayer(inFeatures, outFeatures, weightScale, preHookFx)  
    def dense_selfconnected(self, inFeatures, outFeatures, weightScale=10, preHookFx=None):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _denseLayer_selfconnect(inFeatures, outFeatures, weightScale, preHookFx)  
        
    def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, preHookFx=None):    # default weight scaling of 100
        '''
        Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.conv2d`` applied for each time instance.

        Arguments:
            * ``inChannels`` (``int``): number of channels in input
            * ``outChannels`` (``int``): number of channls produced by convoluion
            * ``kernelSize`` (``int`` or tuple of two ints): size of the convolving kernel
            * ``stride`` (``int`` or tuple of two ints): stride of the convolution. Default: 1
            * ``padding`` (``int`` or tuple of two ints):   zero-padding added to both sides of the input. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
            * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
            * ``weightScale``: sale factor of default initialized weights. Default: 100
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> conv = snnLayer.conv(2, 32, 5) # 32C5 flter
        >>> output = conv(input)           # must have 2 channels
        '''
        return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, preHookFx) 
        
    def pool(self, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        '''
        Returns a function that can be called to apply pool layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.``:sum pooling applied for each time instance.

        Arguments:
            * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to pool over
            * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
            * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.
            
        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> pool = snnLayer.pool(4) # 4x4 pooling
        >>> output = pool(input)
        '''
        return _poolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation, preHookFx)

    def convTranspose(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, preHookFx=None):
        '''
        Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
        It behaves the same as ``torch.nn.ConvTranspose3d`` applied for each time instance.

        Arguments:
            * ``inChannels`` (``int``): number of channels in input
            * ``outChannels`` (``int``): number of channels produced by transposed convolution
            * ``kernelSize`` (``int`` or tuple of two ints): size of ransposed convolution kernel
            * ``stride`` (``int`` or tuple of two ints): stride of the transposed convolution. Default: 1
            * ``padding`` (``int`` or tuple of two ints): amount of implicit zero-padding added to both sides of the input. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
            * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
            * ``weightScale`` : scale factor of default initialized weights. Default: 100
            * ``preHookFx``: a function that operates on weights before applying it. Could be used for quantization etc.
        
        The parameters kernelSize, stride, padding, dilation can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a `tuple` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second is used for the width dimension

        Usage:

        >>> convT = snnLayer.convTranspose(32, 2, 5) # 2T5 flter, the opposite of 32C5 filter
        >>> output = convT(input)
        '''
        return _convTransposeLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, preHookFx)

    def unpool(self, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        '''
        Returns a function that can be called to apply unpool layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.`` unpool layers.

        Arguments:
            * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to unpool over
            * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
            * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dialtion`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> unpool = snnLayer.unpool(2) # 2x2 unpooling
        >>> output = unpool(input)
        '''
        return _unpoolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation, preHookFx)

    def dropout(self, p=0.5, inplace=False):
        '''
        Returns a function that can be called to apply dropout layer to the input tensor.
        It behaves similar to ``torch.nn.Dropout``.
        However, dropout over time dimension is preserved, i.e.
        if a neuron is dropped, it remains dropped for entire time duration.

        Arguments:
            * ``p``: dropout probability.
            * ``inplace`` (``bool``): inplace opeartion flag.

        Usage:

        >>> drop = snnLayer.dropout(0.2)
        >>> output = drop(input)
        '''
        return _dropoutLayer(p, inplace)

    def delayShift(self, input, delay, Ts=1):
        '''
        Applies delay in time dimension (assumed to be the last dimension of the tensor) of the input tensor.
        The autograd backward link is established as well.

        Arguments:
            * ``input``: input Torch tensor.
            * ``delay`` (``float`` or Torch tensor): amount of delay to apply.
              Same delay is applied to all the inputs if ``delay`` is ``float`` or Torch tensor of size 1.
              If the Torch tensor has size more than 1, its dimension  must match the dimension of input tensor except the last dimension.
            * ``Ts``: sampling time of the delay. Default is 1.
        
        Usage:

        >>> delayedInput = slayer.delayShift(input, 5)
        '''
        return _delayFunctionNoGradient.apply(input, delay, Ts)

    def delay(self, inputSize):
        '''
        Returns a function that can be called to apply delay opeartion in time dimension of the input tensor.
        The delay parameter is available as ``delay.delay`` and is initialized uniformly between 0ms  and 1ms.
        The delay parameter is stored as float values, however, it is floored during actual delay applicaiton internally.
        The delay values are not clamped to zero.
        To maintain the causality of the network, one should clamp the delay values explicitly to ensure positive delays.

        Arguments:
            * ``inputSize`` (``int`` or tuple of three ints): spatial shape of the input signal in CHW format (Channel, Height, Width).
              If integer value is supplied, it refers to the number of neurons in channel dimension. Heighe and Width are assumed to be 1.   

        Usage:

        >>> delay = snnLayer.delay((C, H, W))
        >>> delayedSignal = delay(input)

        Always clamp the delay after ``optimizer.step()``.

        >>> optimizer.step()
        >>> delay.delay.data.clamp_(0)  
        '''
        return _delayLayer(inputSize, self.simulation['Ts'])

    def delay_minmax(self, inputSize, step=10):
        '''
        Returns a function that can be called to apply delay opeartion in time dimension of the input tensor.
        The delay parameter is available as ``delay.delay`` and is initialized uniformly between 0ms  and 1ms.
        The delay parameter is stored as float values, however, it is floored during actual delay applicaiton internally.
        The delay values are not clamped to zero.
        To maintain the causality of the network, one should clamp the delay values explicitly to ensure positive delays.

        Arguments:
            * ``inputSize`` (``int`` or tuple of three ints): spatial shape of the input signal in CHW format (Channel, Height, Width).
              If integer value is supplied, it refers to the number of neurons in channel dimension. Heighe and Width are assumed to be 1.   

        Usage:

        >>> delay = snnLayer.delay((C, H, W))
        >>> delayedSignal = delay(input)

        Always clamp the delay after ``optimizer.step()``.

        >>> optimizer.step()
        >>> delay.delay.data.clamp_(0)  
        '''
        return _delayLayer_minmax(inputSize,  self.simulation['Ts'], step=step)  
    # def applySpikeFunction(self, membranePotential):
    #   return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

    def spike(self, membranePotential):
        '''
        Applies spike function and refractory response.
        The output tensor dimension is same as input.
        ``membranePotential`` will reflect spike and refractory behaviour as well.

        Arguments:
            * ``membranePotential``: subthreshold membrane potential.

        Usage:

        >>> outSpike = snnLayer.spike(membranePotential)
        '''
        return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

class _denseLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None):
        '''
        '''
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        # print('Kernel Dimension:', kernel)
        # print('Input Channels  :', inChannels)
        
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx

    
    def forward(self, input):
        '''
        '''
        if self.preHookFx is None:
            #print('come1')
            return F.conv3d(input, 
                            self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            #import numpy as np
            #np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            #print('come here', np.shape(self.preHookFx(self.weight)), self.preHookFx(self.weight)[:10])
            #mm  =  F.conv3d(input, self.preHookFx(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
            #print('shape', np.shape(self.weight))
            #if np.shape(input) == [1,700,1,1,800]:
            #print('what is', self.preHookFx(self.weight).cpu().numpy()[:20,:20,:,:,:])
            return F.conv3d(input, 
                            self.preHookFx(self.weight), self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)




class _denseLayer_selfconnect(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None):
        '''
        '''
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        # print('Kernel Dimension:', kernel)
        # print('Input Channels  :', inChannels)
        
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        super(_denseLayer_selfconnect, self).__init__(inChannels, outChannels, kernel, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx

    
    def forward(self, input):
        '''
        '''
        mask = torch.eye(self.weight.shape[0]).view(self.weight.shape[0],self.weight.shape[0],1,1,1).to(input.device)
        #print('shape', mask.shape,self.weight.shape )
        if self.preHookFx is None:
            #print('come1')
            return F.conv3d(input, 
                            self.weight*mask, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.preHookFx(self.weight)*mask, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)


class _convLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, preHookFx=None):
        inChannels = inFeatures
        outChannels = outFeatures
        
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # groups
        # no need to check for groups. It can only be int

        # print('inChannels :', inChannels)
        # print('outChannels:', outChannels)
        # print('kernel     :', kernel, kernelSize)
        # print('stride     :', stride)
        # print('padding    :', padding)
        # print('dilation   :', dilation)
        # print('groups     :', groups)

        super(_convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In conv, using weightScale of', weightScale)

        self.preHookFx = preHookFx

    def forward(self, input):
        '''
        '''
        if self.preHookFx is None:
            return F.conv3d(input, 
                            self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.preHookFx(self.weight), self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)

class _poolLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
        
        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # print('theta      :', theta)
        # print('kernel     :', kernel, kernelSize)
        # print('stride     :', stride)
        # print('padding    :', padding)
        # print('dilation   :', dilation)
        
        super(_poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)   

        # set the weights to 1.1*theta and requires_grad = False
        self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)
        # print('In pool layer, weight =', self.weight.cpu().data.numpy().flatten(), theta)

        self.preHookFx = preHookFx


    def forward(self, input):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        
        # add necessary padding for odd spatial dimension
        # if input.shape[2]%2 != 0:
            # input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], 1, input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        # if input.shape[3]%2 != 0:
            # input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], 1, input.shape[4]), dtype=dtype).to(device)), 3)
        if input.shape[2]%self.weight.shape[2] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        if input.shape[3]%self.weight.shape[3] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]), dtype=dtype).to(device)), 3)

        dataShape = input.shape

        if self.preHookFx is None:
            result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                              self.weight, self.bias, 
                              self.stride, self.padding, self.dilation)
        else:
            result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                          self.preHooFx(self.weight), self.bias, 
                          self.stride, self.padding, self.dilation)
        # print(result.shape)
        return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))

class _convTransposeLayer(nn.ConvTranspose3d):
    '''
    '''
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, preHookFx=None):
        inChannels = inFeatures
        outChannels = outFeatures

        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # groups
        # no need to check for groups. It can only be int

        super(_convTransposeLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, 0, groups, False, dilation)

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed

        self.preHookFx = preHookFx

    def forward(self, input):
        '''
        '''
        if self.preHookFx is None:
            return F.conv_transpose3d(
                input,
                self.weight, self.bias,
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )
        else:
            return F.conv_transpose3d(
                input,
                self.preHookFx(self.weight), self.bias,
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )

class _unpoolLayer(nn.ConvTranspose3d):
    '''
    '''
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
        
        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))
        
        super(_unpoolLayer, self).__init__(1, 1, kernel, stride, padding, 0, 1, False, dilation)

        self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad=False)

        self.preHookFx = preHookFx

    def forward(self, input):
        '''
        '''
        # device = input.device
        # dtype  = input.dtype
        # # add necessary padding for odd spatial dimension
        # This is not needed as unpool multiplies the spatial dimension, hence it is always fine
        # if input.shape[2]%self.weight.shape[2] != 0:
        #     input = torch.cat(
        #         (
        #             input, 
        #             torch.zeros(
        #                 (input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]),
        #                 dtype=dtype
        #             ).to(device)
        #         ),
        #         dim=2,
        #     )
        # if input.shape[3]%self.weight.shape[3] != 0:
        #     input = torch.cat(
        #         (
        #             input,
        #             torch.zeros(
        #                 (input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]),
        #                 dtype=dtype
        #             ),
        #             dim=3,
        #         )
        #     )

        dataShape = input.shape

        if self.preHookFx is None:
            result = F.conv_transpose3d(
                input.reshape((dataShape[0], 1, -1, dataShape[3], dataShape[4])),
                self.weight, self.bias, 
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )
        else:
            result = F.conv_transpose3d(
                input.reshape((dataShape[0], 1, -1, dataShape[3], dataShape[4])),
                self.preHookFx(self.weight), self.bias, 
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )

        return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))

class _dropoutLayer(nn.Dropout3d):
    '''
    '''
    # def __init__(self, p=0.5, inplace=False):
    #   super(_dropoutLayer, self)(p, inplace)

    '''
    '''
    def forward(self, input):
        inputShape = input.shape
        return F.dropout3d(input.reshape((inputShape[0], -1, 1, 1, inputShape[-1])),
                           self.p, self.training, self.inplace).reshape(inputShape)

class _pspLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, filter, Ts):
        inChannels  = 1
        outChannels = 1
        kernel      = (1, 1, torch.numel(filter))

        self.Ts = Ts

        super(_pspLayer, self).__init__(inChannels, outChannels, kernel, bias=False) 

        # print(filter)
        # print(np.flip(filter.cpu().data.numpy()).reshape(self.weight.shape)) 
        # print(torch.FloatTensor(np.flip(filter.cpu().data.numpy()).copy()))

        flippedFilter = torch.FloatTensor(np.flip(filter.cpu().data.numpy()).copy()).reshape(self.weight.shape)

        self.weight = torch.nn.Parameter(flippedFilter.to(self.weight.device), requires_grad = True)

        self.pad = torch.nn.ConstantPad3d(padding=(torch.numel(filter)-1, 0, 0, 0, 0, 0), value=0)

    def forward(self, input):
        '''
        '''
        inShape = input.shape
        inPadded = self.pad(input.reshape((inShape[0], 1, 1, -1, inShape[-1])))
        # print((inShape[0], 1, 1, -1, inShape[-1]))
        # print(input.reshape((inShape[0], 1, 1, -1, inShape[-1])).shape)
        # print(inPadded.shape)
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape(inShape)

class _pspFilter(nn.Conv3d):
    '''
    '''
    def __init__(self, nFilter, filterLength, Ts, filterScale=1):
        inChannels  = 1
        outChannels = nFilter
        kernel      = (1, 1, filterLength)
        
        super(_pspFilter, self).__init__(inChannels, outChannels, kernel, bias=False) 

        self.Ts  = Ts
        self.pad = torch.nn.ConstantPad3d(padding=(filterLength-1, 0, 0, 0, 0, 0), value=0)

        if filterScale != 1:
            self.weight.data *= filterScale

    def forward(self, input):
        '''
        '''
        N, C, H, W, Ns = input.shape
        inPadded = self.pad(input.reshape((N, 1, 1, -1, Ns)))
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape((N, -1, H, W, Ns))

class _spikeFunction(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
        '''
        '''
        device = membranePotential.device
        dtype  = membranePotential.dtype
        threshold      = neuron['theta']
        oldDevice = torch.cuda.current_device()

        # if device != oldDevice: torch.cuda.set_device(device)
        # torch.cuda.device(3)

        # spikeTensor = torch.empty_like(membranePotential)

        # print('membranePotential  :', membranePotential .device)
        # print('spikeTensor        :', spikeTensor       .device)
        # print('refractoryResponse :', refractoryResponse.device)
            
        # (membranePotential, spikes) = slayer_cuda.get_spikes_cuda(membranePotential,
        #                                                         torch.empty_like(membranePotential),  # tensor for spikes
        #                                                         refractoryResponse,
        #                                                         threshold,
        #                                                         Ts)
        import numpy as np
        np.set_printoptions(threshold=np.inf)
        import numpy as np
        import matplotlib.pyplot as plt
        spikes = slayerCuda.getSpikes(membranePotential.contiguous(), refractoryResponse, threshold, Ts)
        '''
        if membranePotential.shape[1] == 20:
        # 去掉多余维度，保留 [20, 300]，对应 20 条曲线，每条有 300 个时间步
            membranePotential_reshaped = membranePotential.cpu().detach().numpy().reshape(20, 300)
            plt.figure(figsize=(12, 8))

        # 绘制 20 条曲线，每一条对应 20 个神经元在 300 时间步内的电位变化
            for i in range(8,12):
                plt.plot(membranePotential_reshaped[i], label=f'Neuron {i+1}')

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.title('Membrane Potential over Time (20 Neurons, 300 Time Steps)', fontsize=14)
            plt.xlabel('Time Steps', fontsize=12)
            plt.ylabel('Membrane Potential', fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('before',dpi=600)
        #if membranePotential.shape[1] == 20:
        #    print('before',membranePotential.cpu().detach().numpy()[:,10,] )

        #if membranePotential.shape[1] == 20:
        #    print('after',membranePotential.cpu().detach().numpy()[:,10,]  )  
       
        import numpy as np
        import matplotlib.pyplot as plt
        
        if membranePotential.shape[1] == 20:
        # 去掉多余维度，保留 [20, 300]，对应 20 条曲线，每条有 300 个时间步
            membranePotential_reshaped = membranePotential.cpu().detach().numpy().reshape(20, 800)
            plt.figure(figsize=(25, 12))
            #plt.xlim(200, 700)  # 时间范围
        # 绘制 20 条曲线，每一条对应 20 个神经元在 300 时间步内的电位变化
            for i in range(10,11):
                plt.plot(membranePotential_reshaped[i], label=f'Neuron {i+1}')

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.title('Membrane Potential over Time (20 Neurons, 300 Time Steps)', fontsize=14)
            plt.xlabel('Time Steps', fontsize=14)
            plt.ylabel('Membrane Potential', fontsize=14)
            plt.grid(True)
        
            plt.tight_layout()
            plt.savefig('membrane_overtime_true',dpi=600)
         '''
     
        pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
        # pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho']                   , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
        # torch.cuda.synchronize()
        
        # if device != oldDevice: torch.cuda.set_device(oldDevice)
        # torch.cuda.device(oldDevice)
        
        return spikes
        
    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
        spikePdf = pdfScale / pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)
        # return gradOutput, None, None, None # This seems to work better!
        return gradOutput * spikePdf, None, None, None
        # plt.figure()
        # plt.plot(gradOutput[0,5,0,0,:].cpu().data.numpy())
        # print   (gradOutput[0,0,0,0,:].cpu().data.numpy())
        # plt.plot(membranePotential[0,0,0,0,:].cpu().data.numpy())
        # plt.plot(spikePdf         [0,0,0,0,:].cpu().data.numpy())
        # print   (spikePdf         [0,0,0,0,:].cpu().data.numpy())
        # plt.show()
        # return gradOutput * spikePdf, None, None, None

class _pspFunction(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, spike, filter, Ts):
        device = spike.device
        dtype  = spike.dtype
        psp = slayerCuda.conv(spike.contiguous(), filter, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(filter, Ts)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (filter, Ts) = ctx.saved_tensors
        gradInput = slayerCuda.corr(gradOutput.contiguous(), filter, Ts)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass
            
        return gradInput, gradFilter, None

class _delayLayer(nn.Module):
    '''
    '''
    def __init__(self, inputSize, Ts):
        super(_delayLayer, self).__init__()

        if type(inputSize) == int:
            inputChannels = inputSize
            inputHeight   = 1
            inputWidth    = 1
        elif len(inputSize) == 3:
            inputChannels = inputSize[0]
            inputHeight   = inputSize[1]
            inputWidth    = inputSize[2]
        else:
            raise Exception('inputSize can only be 1 or 2 dimension. It was: {}'.format(inputSize.shape))

        self.delay = torch.nn.Parameter(torch.rand((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        # self.delay = torch.nn.Parameter(torch.empty((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        # print('delay:', torch.empty((inputChannels, inputHeight, inputWidth)))
        self.Ts = Ts

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        if input.numel() != self.delay.numel() * input.shape[-1] * input.shape[0]:
            return _delayFunction.apply(input, self.delay.repeat((1, H, W)), self.Ts) # different delay per channel
        else:
            return _delayFunction.apply(input, self.delay, self.Ts) #different delay per neuron

class _delayFunction(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, delay, Ts, step=10):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input.contiguous(), delay.data, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(output, delay.data, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        # autograd tested and verified
        (output, delay, Ts) = ctx.saved_tensors
        diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
        outputDiff = slayerCuda.conv(output.contiguous(), diffFilter, 1)
        # the conv operation should not be scaled by Ts. 
        # As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
        gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
        # no minus needed here, as it is included in diffFilter which is -1 * [1, -1]

        return slayerCuda.shift(gradOutput.contiguous(), -delay, Ts), gradDelay, None
class _delayLayer_minmax(nn.Module):
    '''
    '''
    def __init__(self, inputSize, Ts, step=10):
        super(_delayLayer_minmax, self).__init__()

        if type(inputSize) == int:
            inputChannels = inputSize
            inputHeight   = 1
            inputWidth    = 1
        elif len(inputSize) == 3:
            inputChannels = inputSize[0]
            inputHeight   = inputSize[1]
            inputWidth    = inputSize[2]
        else:
            raise Exception('inputSize can only be 1 or 2 dimension. It was: {}'.format(inputSize.shape))

        self.delay = torch.nn.Parameter(torch.rand((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        self.mind = torch.nn.Parameter(torch.tensor(0.0))
        self.maxd = torch.nn.Parameter(torch.tensor(10.0))
        # self.delay = torch.nn.Parameter(torch.empty((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        # print('delay:', torch.empty((inputChannels, inputHeight, inputWidth)))
        self.Ts = Ts
        self.step = step

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        delta = torch.mean(torch.abs(self.delay)).detach()
        maxd = delta + F.softplus(self.maxd)
        if input.numel() != self.delay.numel() * input.shape[-1] * input.shape[0]:
 
            return _delayFunction1.apply(input, self.delay.repeat((1, H, W)), self.Ts, self.mind, maxd,self.step) # different delay per channel
        else:
            return _delayFunction1.apply(input, self.delay, self.Ts, self.mind, maxd,self.step) #different delay per neuron


class DelayClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, l, U):
        # 前向传播：硬性剪切操作
        # 如果 input > U，则输出 U；如果 input < l，则输出 l；否则输出 input
        #out = torch.clamp(input, l, U)
        out = input.clone()
        #out[input>U] = U
        out[input <l] = 0
    
        ctx.save_for_backward(input, l, U)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：采用直通估计器，让梯度“透传”
        input, l, U = ctx.saved_tensors

        # 对 input 的梯度，直接把 grad_output 透传，不管是否在边界外
        grad_input = grad_output.clone()

        # 对 l，当 input < l 时，原始 forward 输出为 l，
        # 因此对于 l 的梯度应为 grad_output 在这些位置的和（d(clamp)/dl = 1）
        indicator_lower = (input < l).type_as(grad_output)
        grad_l = torch.sum(grad_output * indicator_lower)

        # 对 U，当 input > U 时，原始 forward 输出为 U，
        # 因此对于 U 的梯度也为 grad_output 在这些位置的和（d(clamp)/dU = 1）
        indicator_upper = (input > U).type_as(grad_output)
        grad_U = torch.sum(grad_output * indicator_upper)
        ctx.grad_l = grad_l
        ctx.grad_U = grad_U
        # 返回的梯度分别对应 input, l, U
        return grad_input, grad_l, grad_U

class DelayClampSTE1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, l, U):
        # 前向传播：硬性剪切操作
        # 如果 input > U，则输出 U；如果 input < l，则输出 l；否则输出 input
        #out = torch.clamp(input, l, U)
        delta = 0.7*torch.mean(torch.abs(input)).detach()
        
        # 使用 torch.where 进行条件赋值：
        # 当 input < self.l 时，将输出设为 0
        # 当 self.l <= input < delta 时，将输出设为 self.l
        # 当 input >= delta 时，将输出设为 delta
        output = torch.where(
            input < l,
            torch.zeros_like(input),
            torch.where(
                input < delta,
                l * torch.ones_like(input),
                U * torch.ones_like(input)
            )
        )
        ctx.save_for_backward(input, l, U,delta)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：采用直通估计器，让梯度“透传”
        input, l, U,delta = ctx.saved_tensors

        # 对 input 的梯度，直接把 grad_output 透传，不管是否在边界外
        grad_input = grad_output.clone()

        # 对 l，当 input < l 时，原始 forward 输出为 l，
        # 因此对于 l 的梯度应为 grad_output 在这些位置的和（d(clamp)/dl = 1）
        indicator_l = ((input >= l) & (input < delta)).type_as(grad_output)
        grad_l = torch.sum(grad_output * indicator_l)

        # 对 U 的梯度：仅在 forward 输出为 U 的区域，即当 input >= delta 时，梯度传递为 1。
        indicator_U = (input >= delta).type_as(grad_output)
        grad_U = torch.sum(grad_output * indicator_U)
        ctx.grad_l = grad_l
        ctx.grad_U = grad_U

        # 返回的梯度分别对应 input, l, U
        return grad_input, grad_l, grad_U


class _delayFunction1(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, delay, Ts, mind, maxd,step=10):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        step = torch.autograd.Variable(torch.tensor(step, device=device, dtype=dtype), requires_grad=False)
        #delay_inference = quantize_delay(delay,0, step=step)
        delay_inference = quantize_delay(delay,mind, step=step)
        delay_inference = DelayClampSTE.apply(delay_inference, mind, maxd) 

        #delay_inference = DelayClampSTE1.apply(delay, mind, maxd) 

        #with open("delay_valuess_win80_inference.txt", "a") as f:

        #    f.write("secodn: {} \n".format( torch.round(delay_inference).squeeze()))
        output = slayerCuda.shift(input.contiguous(), delay_inference, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(output, delay.data, Ts)
        ctx.mind = mind
        ctx.maxd = maxd  
        ctx.delay_inference =  delay_inference     
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        # autograd tested and verified
        (output, delay, Ts) = ctx.saved_tensors
        diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
        outputDiff = slayerCuda.conv(output.contiguous(), diffFilter, 1)
        # the conv operation should not be scaled by Ts. 
        # As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
        gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
        # no minus needed here, as it is included in diffFilter which is -1 * [1, -1]
        '''#learable delay threshold
        indicator_lower = (delay < ctx.mind).type_as(gradDelay)
        grad_mind = torch.sum(gradDelay * indicator_lower)
        # 当原始 delay 大于 maxd 时，输出被置为 maxd，因此梯度应传递到 maxd
        indicator_upper = (delay > ctx.maxd).type_as(gradDelay)
        grad_maxd = torch.sum(gradDelay * indicator_upper)
        '''
        grad_mind = None
        grad_maxd = None 
        
        return slayerCuda.shift(gradOutput.contiguous(), -delay, Ts), gradDelay, None, grad_mind, grad_maxd, None

class DelayQuantizeCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mind, step=10):
        # 量化操作：对 (input - l)/step 进行向下取整后乘回 step，并加上 l
        quantized = torch.floor((input - mind) / step) * step + mind
        ctx.save_for_backward(input, mind, torch.tensor(step, device=input.device, dtype=input.dtype))
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # 采用直通估计器：反向传播时直接将梯度传递给 input 和 l
        input, l, step = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_l = grad_output.clone()  # 直接传递梯度到 l（作为一种 STE 近似）
        return grad_input, grad_l, None

def quantize_delay(input, l, step=10):
    return DelayQuantizeCustom.apply(input, l, step)



class _delayFunctionNoGradient(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, delay, Ts=1):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input.contiguous(), delay, Ts)
        Ts     = torch.autograd.Variable(torch.tensor(Ts   , device=device, dtype=dtype), requires_grad=False)
        delay  = torch.autograd.Variable(torch.tensor(delay, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(delay, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (delay, Ts) = ctx.saved_tensors
        return slayerCuda.shift(gradOutput.contiguous(), -delay, Ts), None, None
