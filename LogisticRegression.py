
# coding: utf-8

# In[6]:

import numpy as np
from numpy.linalg import inv
        


# In[ ]:

class LogisticRegression(object):
    
    """ LogisticRgressionClassification
    
    """
    
    def __init__(self):
        pass
    
    def train(self,xtrain,ytrain):
        
        
        """y=x*b>>>>>>>>y[m,k]=x[m,n]*_w[n,k]"""
        m = xtrain.shape[0] # number  of samples
        n = xtrain.shape[1] # number of features
        k = 10 # number of classfications        
        
        self._w = np.empty((n+1,k),dtype=np.float64)       
        x = np.concatenate((xtrain,np.ones((m,1),dtype=xtrain.dtype)),axis=1)

        inverse = inv(np.dot(x.T,x))
        for i in range(10):
            t = np.array([1.0 if i == y else 0.0 for y in ytrain])
            self._w[:,i] = np.dot(inverse,np.dot(x.T,t))
            
        
    def predict(self,xtest):
        #xtest is a m*n np.array, m is the number of test samples, n is the number of features,
        # which is 10000*3072 for cifar10
        
        m = xtest.shape[0]
        
        ypred  = np.empty(m,dtype=np.float64)
        
        x = np.concatenate((xtest,np.ones((m,1),dtype=xtest.dtype)),axis=1)
        
        f = np.dot(x,self._w)
        
        for i in range(f.shape[0]):
            scores = f[i]
            ypred[i] = np.argmax(scores)
        
        return ypred


# In[ ]:



