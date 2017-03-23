#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random 

class Network(object):
    
    def __init__(self,sizes):
        self.num_layers = len(sizes)#No. layer
        self.sizes = sizes
        self.bias = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    
    def feedforward(self,a):
        '''
        if a is input and output will be returned
        '''
        for w,b in zip(self.weights,self.bias):
            a = np.dot(w,a) + b
        return a
    
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        # epochs: how many times to train
        # mini_batch_size: every SGD use how many trainning data
        # eta: learning_rate
        if test_data: n_test = len(tets_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0,n,mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {0}:{1} / {2}".format(j,self.evaluate(test_data),n_test)
            else:
                print "EPoch {0} complete".format(j)

    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nbla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch)) * nw for w,nw in zip(self.weights,nabla_w)]
        self.bias = [b-(eta/len(mini_batch) * nb) for b,nb in zip(self.bias,nabla_b)]

    def backprop(self,x,y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = [x]#record the activation of every node layer by layer
        zs = [] #record the activation before the sigmoid function
        layer = x
        for w,b in zip(self.weights,self.bias):
            layer = np.dot(w,layer) + b
            zs.append(layer)
            actv = sigmoid(layer)
            activation.append(actv)
        #Delta = [] #list of error in every node layer by layer
        delta = (activation[-1] - y) * sigmoid_prime(zs[-1])#output delta(error)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activation[-2].T)
        #for w,b in zip(self.weights,self.bias):
        for i in xrange(2,self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].T,delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activation[-i-1].T)
        return (nabla_b,nabla_w)
def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
        return sigmoid(z) * (1-sigmoid(z))
    
    def evaluate(self,tets_data):
        test_results = [(np.argmax(self.feedforward(x),y) for x,y in test_data)]
        return sum(int(x == y) for (x, y) in test_results)



