#!/bin/env python
"""
feature_extract.py

Author: Ethan Goan
Queensland University of Technology
DREAM Mammography Challenge
2016

Description:

feature class is described, which inherits all functionality from the breast class.
feature extraction is handled in this class through forms of wavelet packet decomposition
for finding textural features. Features from the wavelet packets are then found and saved
to be used for training the classifier.


"""


import dicom
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
from scipy import stats
from scipy import signal as signal
from scipy.ndimage import filters as filters
from scipy.ndimage import measurements as measurements
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import local_binary_pattern

from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage as ndi
import entropy

import pywt
from breast import breast






class feature(breast):

    def __init__(self, file_path = None, wavelet_type = 'haar', levels = 3, benign_scans = True, no_images = 1):
        breast.__init__(self, file_path)
        
        self.packets = []             #wavelet packets for each level of decomposition
        self.levels = levels          #level of wavelet packet decomposition wanted. If this
                                      #level is larger than the maximum possible level of decomposition, the
                                      #initialise function will throw an error message and set it to the maximum
        self.wavelet_type = wavelet_type
        #now listing some texture features at each level
        #these will be a list of arrays as well, the same size as the indicies
        #will calculate these features for each wavelet level for each branch using the
        #co-occurrance matrix
        
        #GLOBAL FEATURES
        self.homogeneity = []
        self.entropy = []
        self.energy = []
        self.contrast = []
        self.dissimilarity = []
        self.correlation = []
        self.density = []
        
        #Wavelet FEATURES
        self.wave_kurtosis = []
        self.wave_entropy = []
        self.wave_energy = []
        self.wave_contrast = []
        self.wave_dissimilarity = []
        self.wave_correlation = []
        
        
        
        #FEATURES FROM FIBROGLANDULAR DISK
        self.fibro_homogeneity = []
        self.fibro_entropy = []
        self.fibro_energy = []
        self.fibro_contrast = []
        self.fibro_dissimilarity = []
        self.fibro_correlation = []
        
        #FEATURES FROM MICROCALCIFICATIONS
        self.micro_homogeneity = []
        self.micro_entropy = []
        self.micro_energy = []
        self.micro_contrast = []
        self.micro_dissimilarity = []
        self.micro_correlation = []
        
        self.indicies = []            #way to index each decomposition level
        self.benign_scans = benign_scans
        self.no_images = no_images
        self.current_image_no = -1     #this will increment as we load in every individual scan
                
        
        self.__initialise_feature_lists()
        if(file_path != None):
            self.initialise(file_path)
            
        
        
        
        
    """
    initialise()
    
    Description:
    Function will be called either from the the __init__ function, or from the main function to initialise
    the wavelet decomposition of the data
    If the wavelet level is set too large, function will display an error message and set the level of decomposition
    to the maximum.
    The indicies (the labels for decomp eg. 'avd', 'hv' etc.) for the decomposition will also be found and saved as a list of arrays. 
    These will be used for to index the wavelet decomp levels.
    
    
    """
    
    
    
    def initialise(self, file_path, preprocessing):
        #call the breast initialise function first
        breast.initialise(self, file_path, preprocessing)
        #increment the image number
        #self.current_image_no = self.current_image_no + 1
        #do the wavelet packet decomposition
        
        
        
        
        
        
        
        
        
        
    """
    find_indicies()
    
    Description:
    Function that will create the strings used to index the wavelet packets
    will implement algorithm to find these for arbitrary number of wavelet decomps
    
    """
    
    
    def find_indicies(self):
        
        
        for ii in range(1,self.levels + 1):
            
            #just getting the indecies and putting them in a square array
            temp = [self.packets.node.path for self.packets.node in self.packets.get_level(ii)]
            temp = np.asarray(temp)
            self.indicies.append( temp.reshape( [np.sqrt(np.shape(temp)), np.sqrt(np.shape(temp))] ))
            
            
            
            
            
            
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
        
        #perform the wavelet decomposition
        a = np.copy(self.data)
        
        a[self.fibroglandular_mask == False ] = np.nan        
        
        self.packets = pywt.WaveletPacket2D(data=a, wavelet=self.wavelet_type, mode='sym')
        
        #check that the number of levels isnt to high
        #if it is, lets print something to let them know
        if(self.levels > self.packets.maxlevel):
            print('number of levels given is too large')
            print('Maximum level of %d, level decomp given is %d' %(self.packets.maxlevel, self.levels))
            print('Set to max level and will continue')
            self.levels = self.packets.maxlevel
            #will have to reinitialise the feature lists
            self.__initialise_feature_lists()
            
        #now lets put the indicies for each level in a nice format that is pleseant to index
        #indicies will be a list of numpy arrays for each level
        self.find_indicies()
        
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
    
    features will be saved per decomposition level, so will ha
    @param level = level of decomp we are finding the features for
    
    """
    
    def _get_features_level(self, level):
        
        #initialise array for each decomp branch to hold the features
        image_no = self.current_image_no
        self.homogeneity[image_no][level] = np.zeros(np.shape(self.indicies[level]))
        self.entropy[image_no][level] = np.zeros(np.shape(self.indicies[level]))
        self.energy[image_no][level] = np.zeros(np.shape(self.indicies[level]))
        self.contrast[image_no][level] = np.zeros(np.shape(self.indicies[level]))
        self.dissimilarity[image_no][level] = np.zeros(np.shape(self.indicies[level]))
        self.correlation[image_no][level] = np.zeros(np.shape(self.indicies[level]))
        
        
        #features above will be an array according to the level of decomp from
        #wavelet packet level we are at
        
        for ii in range(0, np.shape(self.indicies[level])[0]):
            for jj in range(0, np.shape(self.indicies[level])[1]):
                
                temp, glcm = self._comatrix(level, ii, jj)
                mask = np.isfinite(temp)
                
                self.homogeneity[image_no][level][ii,jj] = greycoprops(glcm, prop='homogeneity')[0,0]
                self.energy[image_no][level][ii,jj] = greycoprops(glcm, prop='energy')[0,0]
                self.contrast[image_no][level][ii,jj] = greycoprops(glcm, prop='contrast')[0,0]
                self.dissimilarity[image_no][level][ii,jj] = greycoprops(glcm, prop='dissimilarity')[0,0]
                self.correlation[image_no][level][ii,jj] = greycoprops(glcm, prop='correlation')[0,0]
                self.entropy[image_no][level][ii,jj] = entropy.shannon_entropy(temp)
                
                #self.wave_kurtosis[image_no][level][ii,jj] = stats.kurtosis(np.histogram( temp[mask].ravel() ))
                #self.wave_energy[image_no][level][ii,jj] = np.sum(temp[mask]**2)
                #self.wave_entropy[image_no][level][ii,jj] = entropy.shannon_entropy(temp[mask])
                
                
                #print('homogeneity = %f ' %self.homogeneity[image_no][level][ii,jj])
                #print('energy = %f ' %self.energy[image_no][level][ii,jj])
                #print('contrast = %f ' %self.contrast[image_no][level][ii,jj]) 
                #print('dissimilarity = %f ' %self.dissimilarity[image_no][level][ii,jj])
                #print('correlation = %f ' %self.correlation[image_no][level][ii,jj])
                #print('entropy = %f ' %self.entropy[image_no][level][ii,jj])
                
                
        """
        temp =  self.packets[self.indicies[0][0,1]].data * self.packets[self.indicies[0][1,0]].data# * self.packets[self.indicies[0][1,1]].data
        plt.figure()
        plt.imshow( temp)
        plt.colorbar()
        plt.show()        
        """
        
        
        
        
        
        
        
    def _comatrix(self,level, ii,jj):
        
        #find co-occurance matrix and convert to uint8
        temp = self.packets[self.indicies[level][ii,jj]].data
        temp = temp.astype(float)
        temp_mask = (np.isfinite(temp))
        
        #this will be a float value, so lets scale it to be within uint8 range
        #will also do some error checking, as sometimes the values may be negative
        
        min_val = np.min(temp[temp_mask])
        #probably be negative, but if it isn't, set it to zero
        if min_val > 0: min_val = 0
        #make all the values positive
        
        temp = temp + abs(min_val)
        max_val = np.max(temp[temp_mask]) 
        #probably be positive, but if it isn't, set it to one
        #wont set it to zero, as this is what we are using to scale it so we will be dividing by this
        if max_val < 0: max_val = 1
        
        #now lets scale it by and less than 2^8
        temp = np.multiply(temp, (255 /max_val))
        
        #print np.max(temp[np.isfinite(temp)])
        #now will set all the values equal to nan to 2**8
        temp[temp == np.nan] = 256
        
        #now find the co-occurance matrix
        glcm = greycomatrix(temp.astype('uint16'), [1],[0], levels=256, symmetric=True, normed=True)
        #now lets crop it to get rid of the readings that were interpreted as nan
        glcm = glcm[0:255, 0:255,:,:]        
        
        return temp[temp_mask], glcm
        
        
        
        
        
        
        
        
        
        
    def _crop_features(self, num_scans):
        
        self.homogeneity = self.homogeneity[0:num_scans][:][:] 
        self.energy = self.energy[0:num_scans][:][:] 
        self.contrast = self.contrast[0:num_scans][:][:] 
        self.dissimilarity = self.dissimilarity[0:num_scans][:][:] 
        self.correlation = self.correlation[0:num_scans][:][:] 
        self.entropy = self.entropy[0:num_scans][:][:]         
        self.wave_energy = self.wave_energy[0:num_scans][:][:] 
        self.wave_kurtosis = self.wave_kurtosis[0:num_scans][:][:]
        self.wave_entropy = self.wave_entropy[0:num_scans][:][:] 
        
        
        
    """
    __initialise_feature_lists()
    
    Description:
    Is a helper function called by the __init__ function to create a multidimensional
    list to store the features from the breast scan.
    This function is called only after the max level of decomposition has been set, and should not be called from anywhere other than __init__
    There will be a list for each level of wavelet decomposition that we do.
    
    Eg. if we want 3 levels of decomp, will be 
    [ [ np.array of features for each wavelet packet  ]    ]
    
    This allows us to have a list of features for each image we scan, with detail features from
    wavelet decomp from each level
    
    Features will look like
    self.homogeneity[file_number][level_wavelet_decomposition][array_wavelet_decomposition_indicies]
    
    Isnt that nice :)    
    
    """
    
    
    def __initialise_feature_lists(self):        

        #for python 2.7 and 3 compatability for range and xrange
        try:
            xrange
        except NameError:
            xrange = range
        
        self.homogeneity = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.energy = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.contrast = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.dissimilarity = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.correlation = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.entropy = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        
        self.wave_kurtosis = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.wave_energy = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.wave_contrast = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.wave_dissimilarity = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.wave_correlation = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]
        self.wave_entropy = [[0 for j in xrange(self.levels)] for i in xrange(self.no_images)]

