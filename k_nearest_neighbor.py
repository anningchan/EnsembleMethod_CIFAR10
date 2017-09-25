
# coding: utf-8

# In[ ]:

import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with L1 distance """
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        #self.Xtr = X
        #self.ytr = y
        
    def predict(self, xtrain,ytrain,X):
        
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = ytrain.dtype)
        distances =[]
        # loop over all test samples
        for i in range(num_test):
            
          # find the nearest training image to the i'th test image
          # using the L1 distance (sum of absolute value differences)
            
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            distances = np.sum(np.abs(xtrain - X[i,:]), axis = 1)
            #distances = np.random.randn(50000)
           #distances = np.sqrt(np.sum((self.Xtr - X[i,:])^2))
        
            min_index = np.argmin(distances) # get the index with smallest distance

            Ypred[i] = ytrain[min_index] # predict the label of the nearest example
            
            #if i%100 ==0: print "Validating_Test%d" %i
            
        return Ypred

