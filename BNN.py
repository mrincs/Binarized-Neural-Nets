# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 15:05:37 2016

@author: mrins

Input - MNIST dataset 28 * 28 pixels flattened
Output - 10 digits classification
Architecture -  Fully connected layers. 
                Model pretrained with compressed weights in round 1. 
                Binarized neural network has been trained on next round.
                2 hidden layers FC.
                1024 hidden units per layer.
                Drop Out parameter set 0.95 for input layer and 0.8 for hidden layers.
                Momentum for update : 0.95
                Learning Rate : ???
                Mini batch Size : 100 samples per batch
                Number of epoch : 5000
                Cost Function: ???

"""

import numpy as np
import theano
import theano.tensor as T
import theano.scalar as t
from step import binarize

# Activation Functions

def ReLU(x):
    return T.maximum(0.0, x)
    
def Sigmoid(x):
    return T.nnet.sigmoid(x)
    
def HardSigmoid(x):
    return T.clip((x+1.)/2., 0, 1)
    
def Tanh(x):
    return T.tanh(x)
    
def HardTanh(x):
    return T.clip(x, -1, 1)
    
def Sign(x):
    return binarize(x)
    
    

def SkipHiddenUnits(rng, layer, dropOut_prob):
     """ Drop out units from layers during training phase
        'mask' creates binary vector of zeros and ones to suppress any node 
     """
     retain_prob = 1 - dropOut_prob
     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
     mask = srng.binomial(n=1, p=retain_prob, size=layer.shape, dtype=theano.config.floatX)
     return T.cast(layer * mask, theano.config.floatX)
     
     
def ScaleWeight(layer, dropOut_prob):
    """ Rescale weights for averaging of layers during validation/test phase
    """
    retain_prob = 1 - dropOut_prob
    return T.cast(retain_prob * layer, theano.config.floatX)
     
     
def gradientUpdates(cost, params, learningRate):
        
    assert learningRate > 0 and learningRate < 1
           
    gparams = [T.grad(cost, param) for param in params]   
    updates = [
        (param, param - learningRate * gparam)
        for param, gparam in zip(params, gparams)
        ]
    return updates
    
    
def gradientUpdateUsingMomentum(cost, params, learningRate, momentum):
    
    assert learningRate > 0 and learningRate < 1
    assert momentum > 0 and momentum < 1
    
    updates = []
    for param in params:
        paramUpdate = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)
        updates.append((param, param - learningRate * paramUpdate))
        updates.append((paramUpdate, momentum * paramUpdate + (1 - momentum) * T.grad(cost, param)))
    return updates
    

class WeightCompressionLayer(object):
    
    """In the first round, we train an ordinary DNN with tanh activation functions. 
    The only additional step in this part is weight compression,
    with which we can ensure that the weights and biases are bound between -1 and
    +1. The weight compression is done by wrapping the parameters with tanh, and
    the use the wrapped versions during feedforward.
    """
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
            
        self.input = input
        self.W, self.b = self.initializeParams(W, b, rng, n_in, n_out)
        self.params = [self.W, self.b]
                
        # BNN First Round Training FP: real valued network with weight compression #
        lin_output = T.dot(input, Tanh(self.W)) + Tanh(self.b)
        self.output = Tanh(lin_output)

        
    def initializeParams(self, W, b, rng, n_in, n_out):
        
        assert (n_in+n_out) != 0
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6/(n_in+n_out)),
                    high = np.sqrt(6/(n_in+n_out)),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
                
            W = theano.shared(value = W_values, name = "W", borrow = True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = "b", borrow = True)
            
        return [W, b]
        
            
class BinarizedHiddenLayer(object):
    
    """Second round/ Actual Bitwise Neural Network Architecture
    """
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        
        self.input = input
        self.W, self.b = self.initializeParams(W, b, rng, n_in, n_out)            
        self.params = [self.W, self.b]
        
        
        # BNN Second Round Forward Propagation #
        lin_output = T.dot(input, Sign(self.W)) + Sign(self.b)
        self.output = Sign(lin_output)
        
        
    def initializeParams(self, W, b, rng, n_in, n_out):
        
        # Initialize W with trained values in first round
        if W is None:
            raise ValueError("Weight should be initialized from Round 1 training ")
        
        if b is None:
            raise ValueError("Bias should be initialized from Round 1 training  ")
            
        return [W, b]



class WeightCompressionLayerWithDropOut(WeightCompressionLayer):
    
    def __init__(self, rng, input, n_in, n_out, dropOut, train_phase=1, W=None, b=None):
        super(WeightCompressionLayerWithDropOut, self).__init__(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_out,
            W = W,
            b = b)
            
        training_output = SkipHiddenUnits(rng, self.output, dropOut)
        predict_output = ScaleWeight(self.output, dropOut)
        self.output = T.switch(T.neq(train_phase, 0), training_output, predict_output)


class BinarizedHiddenLayerWithDropOut(BinarizedHiddenLayer):
    
    def __init__(self, rng, input, n_in, n_out, dropOut,  train_phase=1, W=None, b=None):
        super(BinarizedHiddenLayerWithDropOut, self).__init__(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_out,
            W = W,
            b = b)
            
        training_output = SkipHiddenUnits(rng, self.output, dropOut)
        predict_output = ScaleWeight(self.output, dropOut)
        self.output = T.switch(T.neq(train_phase, 0), training_output, predict_output)


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.W, self.b = self.initializeParams(W, b, n_in, n_out)
         
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        
        self.params = [self.W, self.b]
        
        self.input = input
        
        
    def initializeParams(self, W, b, n_in, n_out):
        
        if W is None:
            W = theano.shared(
                value = np.zeros((n_in, n_out),
                                 dtype = theano.config.floatX),
                        name = "W",
                        borrow = True
                    )            
        if b is None:
            b = theano.shared(
                value = np.zeros((n_out,),
                                 dtype = theano.config.floatX),
                        name = "b",
                        borrow = True
                    )        
        return [W, b]
        
        
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
        
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
        
        
class WeightCompressionMLP(object):
    
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        self.n_hidden_layers = len(n_hidden)
        self.hiddenLayers = []
        
        for i in range(self.n_hidden_layers): 
            if i == 0:
                inputLayer = input
                inputLayerSize = n_in             
            else:
                inputLayer = self.hiddenLayers[i-1].output
                inputLayerSize = n_hidden[i-1]
            
            hiddenLayer = WeightCompressionLayer(
                rng = rng,
                input = inputLayer,
                n_in = inputLayerSize,
                n_out = n_hidden[i],
            )
            self.hiddenLayers.append(hiddenLayer)
        
        self.outputLayer = LogisticRegression(
            input = self.hiddenLayers[-1].output,            
            n_in = n_hidden[-1],
            n_out = n_out
        )
        
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        
        self.errors = self.outputLayer.errors
        
        self.params = []
        for i in range(self.n_hidden_layers):
            self.params.extend(self.hiddenLayers[i].params)
        self.params.extend(self.outputLayer.params)
                
        self.input = input
        

        
    def getCostAndUpdates(self, y, learningRate = 0.01):
        
        assert y is not None
        
        cost = self.negative_log_likelihood(y)
        updates = gradientUpdates(cost, self.params, learningRate)
        
        return (cost, updates)
        
        
class WeightCompressedMLPWithDropOut(object):
    
    def __init__(self, rng, input, n_in, n_hidden, dropOut, n_out, train_phase):
        
        self.n_hidden_layers = len(n_hidden)
        self.hiddenLayers = []
        
        for i in range(self.n_hidden_layers): 
            if i == 0:
                inputLayer = SkipHiddenUnits(rng, input, dropOut[0])
                inputLayerSize = n_in             
            else:
                inputLayer = self.hiddenLayers[i-1].output
                inputLayerSize = n_hidden[i-1]
            
            hiddenLayer = WeightCompressionLayerWithDropOut(
                rng = rng,
                input = inputLayer,
                n_in = inputLayerSize,
                n_out = n_hidden[i],
                dropOut = dropOut[i],
                train_phase = train_phase
            )
            self.hiddenLayers.append(hiddenLayer)
        
        self.outputLayer = LogisticRegression(
            input = self.hiddenLayers[-1].output,            
            n_in = n_hidden[-1],
            n_out = n_out
        )
        
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        
        self.errors = self.outputLayer.errors
        
        self.params = []
        for i in range(self.n_hidden_layers):
            self.params.extend(self.hiddenLayers[i].params)
        self.params.extend(self.outputLayer.params)
                
        self.input = input
        

        
    def getCostAndUpdates(self, y, learningRate = 0.01):
        
        assert y is not None
        
        cost = self.negative_log_likelihood(y)
        updates = gradientUpdates(cost, self.params, learningRate)
        
        return (cost, updates)
        
        
class BinarizedMLP(object):
    
    def __init__(self, rng, input, n_in, n_hidden, n_out, dropOut, train_phase, preTrainedModel=None):
        
        if preTrainedModel is None:
            raise ValueError("Params should be initialized from Round 1 training ")
        
        assert len(dropOut) == len(n_hidden)+1
        
        self.n_hidden_layers = len(n_hidden)
        self.hiddenLayers = []
        
        for i in range(self.n_hidden_layers): 
            if i == 0:
                inputLayer = SkipHiddenUnits(rng, input, dropOut[0])
                inputLayerSize = n_in             
            else:
                inputLayer = self.hiddenLayers[i-1].output
                inputLayerSize = n_hidden[i-1]
            
            hiddenLayer = BinarizedHiddenLayerWithDropOut(
                rng = rng,
                input = inputLayer,
                n_in = inputLayerSize,
                n_out = n_hidden[i],
                dropOut = dropOut[i+1],
                train_phase = train_phase,
                W = preTrainedModel.hiddenLayers[i].params[0],
                b = preTrainedModel.hiddenLayers[i].params[1]
            )
            self.hiddenLayers.append(hiddenLayer)
        
        self.outputLayer = LogisticRegression(
            input = self.hiddenLayers[-1].output,            
            n_in = n_hidden[-1],
            n_out = n_out,
            W = preTrainedModel.outputLayer.params[0],
            b = preTrainedModel.outputLayer.params[1]
        )
        
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        
        self.errors = self.outputLayer.errors
        
        self.params = []
        for i in range(self.n_hidden_layers):
            self.params.extend(self.hiddenLayers[i].params)
        self.params.extend(self.outputLayer.params)
                
        self.input = input
        
        
    
    def getCostAndUpdates(self, y, learningRate = 0.01, momentum = 0.01):
        
        assert y is not None
        
        cost = self.negative_log_likelihood(y)
        updates = gradientUpdateUsingMomentum(cost, self.params, 
                                              learningRate, momentum)
        
        return (cost, updates)