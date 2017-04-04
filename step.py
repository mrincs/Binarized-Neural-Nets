# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 18:27:36 2016

@author: mrins
"""
from __future__ import print_function

import theano
import numpy
import theano.tensor as T
import theano.scalar as t

import unittest
from theano.tests.unittest_tools import verify_grad



class ScalarBinary(t.UnaryScalarOp):

    __props__ = ()
    itypes = [theano.scalar.convert_to_float32]
    otypes = [theano.scalar.convert_to_float32]
    
    @staticmethod
    def st_impl(x, rho):
        if x <= -rho:
            return -1.0
        elif x >= rho:
            return 1.0
        else:
            return 0
            
    def impl(self, x):
        return self.st_impl(x, 0.5)
    
    def make_node(self, x):
        x = t.as_scalar(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = self.impl(x)
                
    def grad(self, inputs, grads):
        x, = inputs
        gz, = grads
        rval = gz * (1-t.pow(t.tanh(x), 2))
        return [rval]
               
    # Optional methods
    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        z, = outputs
        if node.inputs[0].type in [t.float32, t.float64]:
            """%(z)s = %(x)s < -88.0f ? 0.0 : %(x)s > 15.0f ? 1.0f : 1.0f /(1.0f + exp(-%(x)s));""" % locals()
            return """%(z)s =
                %(x)s >= 0.5
                ? 1.0f
                : %(x)s <= -0.5 
                ? -1.0f 
                : 0.0f;""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
        
#    
#    def make_thunk(self, node, storage_map, _, _2):
#        pass
#    
#    def R_op(self, inputs, eval_points):
#        pass
#
#    def infer_shape(node, input_shapes):
#        pass
    
    

scalar_binarize = ScalarBinary(theano.scalar.convert_to_float32, name='scalar_binarize')
binarize = T.elemwise.Elemwise(scalar_binarize, name='binarize')    

binarize_inplace = T.elemwise.Elemwise(
    ScalarBinary(t.transfer_type(0)),
    inplace_pattern={0: 0},
    name='binarize_inplace',
)
        
        
theano.printing.pprint.assign(binarize, theano.printing.FunctionPrinter('binarize'))


#############################################
##################Test Modules###############
#############################################

class T_scalar_binarize(unittest.TestCase):
        
    def test_perform(self):
        x = theano.tensor.scalar()
        y = scalar_binarize(x)
        f = theano.function([x], [y])
        inp = numpy.random.random()
        inp = numpy.random.randint(-10, 10)
        print("Type of Input:", x.type(), "; Input: ",inp)
        out = f(inp)
        print("Type of Output:", out[0].type(), "; Output: ",out[0])
        
    def test_grad(self):
         verify_grad(scalar_binarize, [numpy.random.rand(5, 7, 2)])
        
class T_vector_binarize(unittest.TestCase):        

    def test_perform(self):    
        x = theano.tensor.dmatrix()
        y = binarize(x)
        f = theano.function([x], [y])
        inp = numpy.random.randn(10,4)
        print("Type of Input:", x.type(), "; Input: ",inp)
        out = f(inp)
        #print("Type of Output:", out[0].type(), "; Output: ",out[0])
        print("Output:", out[0])
        
    def test_grad(self):
        verify_grad(binarize, [numpy.random.rand(5, 7, 2)])
        
        
def test_ScalarBinarization():
    test_scalar_binarize = T_scalar_binarize()
    test_scalar_binarize.setUp()
    test_scalar_binarize.test_perform()
    test_scalar_binarize.test_grad()
    
    
def test_VectorBinarization():
    test_vector_binarize = T_vector_binarize()
    test_vector_binarize.setUp()
    test_vector_binarize.test_perform()
    test_vector_binarize.test_grad()

if __name__ == '__main__':
    #unittest.main()
    #test_ScalarBinarization()
    test_VectorBinarization()
        


