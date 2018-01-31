import numpy as np
from scipy import stats


"""
kl_divergence()

Description:

Compute the KL divergence between the feature vectors
for benign and malignant scans

"""


def kl_divergence(malignant, benign, num_features):
    
    #creating a row vector that will store the kl divergence for each feature
    kld = np.zeros((1, benign.shape[1]))
    #creating an index vector for the features we want
    feats = np.zeros((1,benign.shape[1])).astype(np.bool)
    
    print('test')
    #malignant = np.concatenate((malignant, malignant), axis=0)
    print benign.shape
    print malignant.shape
    num_bins=9
    for ii in range(0, kld.shape[1]):
        malignant_pdf, edge = np.histogram(malignant[:,ii],bins=num_bins,density=True)
        benign_pdf, edge = np.histogram(benign[:,ii],bins=num_bins,density=True)        
        #print(malignant_pdf)
        #print(benign_pdf.shape)
        kld[0,ii] = stats.entropy(malignant_pdf, benign_pdf)
    print kld
    
    #get the n most dissimilar features
    kld[np.isnan(kld)] = 0.0 #getting rid of Nans and Infs
    kld[np.isinf(kld)] = 0.0
    #take absolute value
    kld = np.abs(kld)
    #now sort it
    kld_sorted = np.sort(kld)
    #now lets get the n most dissimilar values
    feats[0,np.argsort(kld.ravel())[-num_features:-1]] = True
    print 'argsort'
    print np.argsort(kld.ravel())[-num_features:-1]
    print 'kld'
    print kld
    return feats.ravel()
