"""
feature_extract.py

Author: Ethan Goan

Description:

This file describes the methods used for feature extraction
and will define objects to hold these features.

Source will build on breast class 









"""








import dicom
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
from scipy import signal as signal
from scipy.ndimage import filters as filters
from scipy.ndimage import measurements as measurements
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)


from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage as ndi
import entropy

import pywt
from breast import breast










class feature(breast):

    def __init__(self, file_path, wavelet_type = 'haar', levels = 3):
        breast.__init__(self, file_path)

        
        self.packets = pywt.WaveletPacket2D(data=self.data, wavelet=wavelet_type, mode='sym')
        self.indicies = []            #way to index each decomposition level

        #check that the number of levels isnt to high
        #if it is, lets print something to let them know
        if(levels <= self.packets.maxlevel):
            self.levels = levels          #number of decompositions
        else:
            print('number of levels given is too large')
            print('Maximum level of %d, level decomp given is %d' %(self.packets.maxlevel, levels))
            print('Set to max level and will continue')
            self.levels = self.packets.maxlevel

        #now lets put the indicies for each level in a nice format that is pleseant to index
        #indicies will be a list of numpy arrays for each level
        self.find_indicies()


        #now listing some texture features at each level
        #these will be a list of arrays as well, the same size as the indicies
        #will calculate these features for each wavelet level for each branch using the
        #co-occurrance matrix


        self.homogeneity = []
        self.entropy = []
        self.energy = []
        self.contrast = []
        self.dissimilarity = []
        self.correlation = []

        

        
    """
    find_indicies()
    
    Description:
    Function that will create the strings used to index the wavelet packets
    will implement algorithm to find these for arbitrary number of wavelet decomps
0
    """


    def find_indicies(self):

        for ii in range(self.levels, 0, -1):
            #just getting the indecies and putting them in a square array
            temp = [self.packets.node.path for self.packets.node in self.packets.get_level(ii)]
            temp = np.asarray(temp)
            self.indicies.append( temp.reshape( [np.sqrt(np.shape(temp)), np.sqrt(np.shape(temp))] ))
            print('indicies')
            print(np.asarray(temp))
            
            
        
    


    """
    get_features()

    Description:
    wrapper function that will call get the features of specified level
    if no input level is specified, will get the features of every level

    @param level = level we want to evaluate. if selecting particular level, should be
                   integer, or maybe array or list.
                   Default = 'all' and is where we will do all of them
    

    """

    def get_features(self, level = 'all'):

        if(level == 'all'):
            for ii in range(0, self.levels):
                self._get_features_level(ii)

        #else will just get the features of a single level
        #not sure why would want to do this, but might be handy?
        else:
            self._get_features_level(level)





    """
    _get_features_level()

    Description:
    Will use the co-occurance matrix of the wavelet decomp level to find texture features

    @param level = level of decomp we are finding the features for

    """

    def _get_features_level(self, level):

        #initialise array for each decomp branch to hold the features
        self.homogeneity = np.zeros(np.shape(self.indicies[level]))
        self.entropy = np.zeros(np.shape(self.indicies[level]))
        self.energy = np.zeros(np.shape(self.indicies[level]))
        self.contrast = np.zeros(np.shape(self.indicies[level]))
        self.dissimilarity = np.zeros(np.shape(self.indicies[level]))
        self.correlation = np.zeros(np.shape(self.indicies[level]))


        #features above will be an array according to the level of decomp from
        #wavelet packet level we are at

        print(self.indicies[level])
        
        for ii in range(0, np.shape(self.indicies[level])[0]):
            for jj in range(0, np.shape(self.indicies[level])[1]):

                #find co-occurance matrix and convert to uint8
                temp = np.copy(self.packets[self.indicies[level][ii,jj]].data)
                temp = temp.astype('uint8')
                glcm = greycomatrix(temp, np.shape(temp), [0])

                self.homogeneity[ii,jj] = greycoprops(glcm, prop='homogeneity')[0,0]
                self.energy[ii,jj] = greycoprops(glcm, prop='energy')[0,0]
                self.contrast[ii,jj] = greycoprops(glcm, prop='contrast')[0,0]
                self.dissimilarity[ii,jj] = greycoprops(glcm, prop='dissimilarity')[0,0]
                self.correlation[ii,jj] = greycoprops(glcm, prop='correlation')[0,0]
                self.entropy[ii,jj] = entropy.shannon_entropy(temp)
        
