import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W=input_dim
    pool_size=2
    stride=1
    pad=(filter_size-1)/2
    H2=(1+(H+2*pad-filter_size)/stride)/pool_size
    W2=(1+(W+2*pad-filter_size)/stride)/pool_size
    self.params['W1']=np.random.normal(size=(num_filters,C,filter_size,filter_size),scale=weight_scale)
    self.params['b1']=np.zeros(num_filters)
    self.params['W2']=np.random.normal(size=(num_filters*H2*W2,hidden_dim),scale=weight_scale)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=np.random.normal(size=(hidden_dim,num_classes),scale=weight_scale)
    self.params['b3']=np.zeros(num_classes)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    cache={}
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    h1,cache['h1']=conv_relu_pool_forward(X,self.params['W1'],self.params['b1'],conv_param,pool_param)
    h2,cache['h2']=affine_relu_forward(h1,self.params['W2'],self.params['b2'])
    scores, cache['h3']=affine_forward(h2,self.params['W3'],self.params['b3'])



    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dout=softmax_loss(scores,y)
    dout,grads['W3'],grads['b3']=affine_backward(dout,cache['h3'])
    dout,grads['W2'],grads['b2']=affine_relu_backward(dout,cache['h2'])
    dx,grads['W1'],grads['b1']=conv_relu_pool_backward(dout,cache['h1'])

    loss+=0.5*self.reg*(np.sum(np.square(self.params['W1']),axis=(0,1,2,3))+np.sum(np.square(self.params['W2']),axis=(0,1))+np.sum(np.square(self.params['W3']),axis=(0,1)))
    grads['W3']+=self.reg*self.params['W3']
    grads['W2']+=self.reg*self.params['W2']
    grads['W1']+=self.reg*self.params['W1']

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass


class N_layer_convnet(object):


  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dims=[100]*2, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,conv_layers=3,affine_layers=2,use_batchnorm=True):
    """
    Initialize a new network.
    [conv-relu-pool]XN - [affine-relu]XM-1- affine - [softmax or SVM]
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.filter_size=filter_size
    self.conv_layers=conv_layers
    self.affine_layers=affine_layers
    self.batch_size=100

    self.bn_params_conv = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in xrange(conv_layers - 1)]



    C,H,W=input_dim
    pool_size=2
    stride=1
    pad=(filter_size-1)/2

    for i in range(1,conv_layers+1):
        H=(1+(H+2*pad-filter_size)/stride)/(pool_size)
        W=(1+(W+2*pad-filter_size)/stride)/pool_size
    for i in range(1,conv_layers+1):
        if (i==1):
            self.params['W_conv'+repr(i)]=np.random.normal(size=(num_filters,C,filter_size,filter_size),scale=weight_scale)
            self.params['b_conv'+repr(i)]=np.zeros(num_filters)
        else:
            self.params['W_conv'+repr(i)]=np.random.normal(size=(num_filters,num_filters,filter_size,filter_size),scale=weight_scale)
            self.params['b_conv'+repr(i)]=np.zeros(num_filters)
        if (self.use_batchnorm and i==1):
            self.params['gamma_conv'+repr(i)]=np.ones((num_filters))
            self.params['beta_conv'+repr(i)]=np.zeros((num_filters))
        elif (self.use_batchnorm and i<conv_layers):
            self.params['gamma_conv'+repr(i)]=np.ones((((self.batch_size*num_filters*num_filters/(pool_size**i),1))))
            self.params['beta_conv'+repr(i)]=np.zeros((self.batch_size*num_filters*num_filters/(pool_size**i),1))

    #input_affine_dim=

    for i in range(1,affine_layers+1):
        if (i==1):
            self.params['W'+repr(i)]=np.random.normal(size=(32*H*W,hidden_dims[i-1]),scale=weight_scale)
            self.params['b'+repr(i)]=np.zeros(hidden_dims[i-1])
        elif (i==affine_layers):
            self.params['W'+repr(i)]=np.random.normal(size=(hidden_dims[i-2],num_classes),scale=weight_scale)
            self.params['b'+repr(i)]=np.zeros(num_classes)
        else:
            self.params['W'+repr(i)]=np.random.normal(size=(hidden_dims[i-2],hidden_dims[i-1]),scale=weight_scale)
            self.params['b'+repr(i)]=np.zeros(hidden_dims[i-1])
        if (self.use_batchnorm and i<affine_layers):
            self.params['gamma'+repr(i)]=np.ones((hidden_dims[i-1]))
            self.params['beta'+repr(i)]=np.zeros((hidden_dims[i-1]))
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    cache={}
    ############################################################################
    # TODO: Implement the forward pass for the N-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    forward_inputs=X
    for i in range(1,self.conv_layers+1):
        forward_inputs,cache['conv'+repr(i)]=conv_forward_fast(forward_inputs,self.params['W_conv'+repr(i)],self.params['b_conv'+repr(i)],conv_param)
        if (self.use_batchnorm and i<self.conv_layers):
            forward_inputs,cache['batch_conv'+repr(i)] = spatial_batchnorm_forward(forward_inputs,self.params['gamma_conv'+repr(i)],self.params['beta_conv'+repr(i)],bn_param)
        forward_inputs,cache['crelu'+repr(i)]=relu_forward(forward_inputs)
        forward_inputs,cache['pool'+repr(i)]=max_pool_forward_fast(forward_inputs,pool_param)

    for i in range(1,self.affine_layers+1):
        forward_inputs, cache['affine'+repr(i)]=affine_forward(forward_inputs,self.params['W'+repr(i)],self.params['b'+repr(i)])
        if (self.use_batchnorm and i<self.affine_layers):
            forward_inputs,cache['bnorm'+repr(i)]=batchnorm_forward(forward_inputs,self.params['gamma'+repr(i)],self.params['beta'+repr(i)],self.bn_params[i-1])
        if (i<self.affine_layers):
            forward_inputs,cache['relu'+repr(i)]=relu_forward(forward_inputs)
    scores=forward_inputs

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dout=softmax_loss(scores,y)
    for i in range(self.affine_layers,0,-1):
        if (i!=self.affine_layers):
            dout=relu_backward(dout,cache['relu'+repr(i)])
        if (self.use_batchnorm and i<self.affine_layers):
            dout,grads['gamma'+repr(i)],grads['beta'+repr(i)] = batchnorm_backward_alt(dout,cache['bnorm'+repr(i)])
        dout,grads['W'+repr(i)],grads['b'+repr(i)]=affine_backward(dout,cache['affine'+repr(i)])
        grads['W'+repr(i)]+=self.reg*self.params['W'+repr(i)]
        loss+=0.5*self.reg*np.sum((np.sum(np.square(self.params['W'+repr(i)]))))


    for i in range(self.conv_layers,0,-1):
        dout=max_pool_backward_fast(dout,cache['pool'+repr(i)])
        dout=relu_backward(dout,cache['crelu'+repr(i)])
        if (self.use_batchnorm and i<self.conv_layers):
            dout,grads['gamma_conv'+repr(i)],grads['beta_conv'+repr(i)]=spatial_batchnorm_backward(dout,cache['batch_conv'+repr(i)])
        dout, grads['W_conv'+repr(i)], grads['b_conv'+repr(i)]=conv_backward_fast(dout,cache['conv'+repr(i)])
        loss+=0.5*self.reg*np.sum(np.square(self.params['W_conv'+repr(i)]),axis=(0,1,2,3))
        grads['W_conv'+repr(i)]+=self.reg*self.params['W_conv'+repr(i)]
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
