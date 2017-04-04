# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:35:31 2016

@author: mrins
"""

from .train import train, train_v2
from .BNN import BinarizedMLP, WeightCompressionMLP # Complete Architectures
from .BNN import BinarizedHiddenLayer, WeightCompressionLayer, BinarizedHiddenLayerWithDropOut, LogisticRegression # Layers
from .BNN import ReLU, Sigmoid, Tanh, Sign  #Ops
from .BNN import gradientUpdates, gradientUpdateUsingMomentum #Updates 
