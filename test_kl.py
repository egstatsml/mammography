import numpy as np
X = np.load('/media/dperrin/preprocessed/preprocessedTrain/model_data/X.npy')
Y = np.load('/media/dperrin/preprocessed/preprocessedTrain/model_data/Y.npy')
from kl_divergence import *
a = kl_divergence(X[Y,:], X[Y==False,:],20)

#print(a)
print np.shape(a)
#print np.shape(X)
for ii in range(0, X.shape[1]):
    X[:,ii] = np.divide(X[:,ii], np.abs(np.max(X[:,ii])))
print(X[:,a])

