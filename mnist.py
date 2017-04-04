# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:44:13 2016

@author: mrins
"""

import os
import six.moves.cPickle as pickle
import gzip

import numpy as np
import theano
import theano.tensor as T

from BNN import BinarizedMLP, WeightCompressedMLPWithDropOut
from train import train, train_v2, train_v3


# For Debugging
#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

def test_BNN_on_MNIST(learning_rate=0.01, L1_reg=0, L2_reg=0.0001,
             dataset='mnist.pkl.gz', batch_size=20):
    
#    :type n_epochs: int
#    :param n_epochs: maximal number of epochs to run the optimizer
#
#    :type dataset: string
#    :param dataset: the path of the MNIST dataset file from
#                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

        
    datasets = load_data(dataset)
        
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
        
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y') 
    train_phase = T.iscalar('train_phase')

    
    # Experiment Parameters
    n_epochs_round_1 =  1000
    n_epochs_round_2 = 10000

    # Architecture Parameters
    n_in = 28*28
    n_hidden = [40]
    n_out = 10
    dropOut = [0.1, 0.5] # First index value is for dropout on input layer  
        
        
    # Optimization Parameters
    learning_rate = 0.01
    momentum = 0.95
    
    
    ###########################
    # BUILD PRETRAINING MODEL #
    ###########################
        
    print('... Designing the model with compressed weights')
    
    rng = np.random.RandomState(7341)
    
    classifier_BNN_pretrain = WeightCompressedMLPWithDropOut(
        rng = rng,
        input = x,
        n_in = n_in,
        n_hidden = n_hidden,
        dropOut = dropOut,
        n_out = n_out,
        train_phase = train_phase
        )
        
    classifier_BNN_pretrain.cost, classifier_BNN_pretrain.updates = \
                classifier_BNN_pretrain.getCostAndUpdates(y, learning_rate)

    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model_round_1 = theano.function(
        inputs=[index],
        outputs=classifier_BNN_pretrain.cost,
        updates=classifier_BNN_pretrain.updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            train_phase: np.cast['int32'](1)
        }
    )
    
    print('.... Model Graph Round 1')    
    theano.printing.debugprint(classifier_BNN_pretrain.outputLayer.p_y_given_x)
#    theano.printing.debugprint(cost)
    
    
    
    ###################
    # BUILD BNN MODEL #
    ###################
    
    print('... Designing the binarized model')
    
    classifier_BNN = BinarizedMLP(
        rng = rng,
        input = classifier_BNN_pretrain.input,
        n_in = n_in,
        n_hidden = n_hidden,
        n_out = n_out,
        dropOut = dropOut,
        train_phase = train_phase,
        preTrainedModel = classifier_BNN_pretrain
        )
        
        
    classifier_BNN.cost, classifier_BNN.updates = \
            classifier_BNN.getCostAndUpdates(y, learning_rate, momentum)
        
    
    
    train_model_round_2 = theano.function(
        inputs=[index],
        outputs=classifier_BNN.cost,
        updates=classifier_BNN.updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            train_phase: np.cast['int32'](1)
        }
    )
    
    print('.... Model Graph Round 2')    
    theano.printing.debugprint(classifier_BNN.outputLayer.p_y_given_x)
    

    test_model = theano.function(
        inputs=[index],
        outputs=classifier_BNN_pretrain.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            train_phase: np.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier_BNN_pretrain.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            train_phase: np.cast['int32'](0)
        }
    )

   

    ###############
    # TRAIN MODEL #
    ###############

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    numBatches = [n_train_batches, n_valid_batches, n_test_batches]
    
    # Pre training - Weight Compression
    print('... Training for round 1')
    train(numBatches, train_model_round_1, validate_model, test_model,
          numEpochs = n_epochs_round_1,
          validationFrequency = validation_frequency)
          
    # Binarized Model Training
    print('... Training for round 2')
    train(numBatches, train_model_round_2, validate_model, test_model,
          numEpochs = n_epochs_round_2,
          validationFrequency = validation_frequency)      


           
           
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... Loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    
    
    
if __name__ == '__main__':
    test_BNN_on_MNIST()