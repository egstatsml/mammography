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
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage as ndi
import entropy

import pywt
from mammogram.breast import breast





class feature(breast):

    def __init__(self, file_path = None, wavelet_type = 'haar', levels = 1, benign_scans = True, no_images = 1):
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
        
        self._initialise_feature_lists()
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
    
    
    
    def get_features(self, bbs = [-1], conf = [0],  level = 'all'):
        #Initially assume the data is valid
        #will only be set to false if we are using regions and there arent any good ones
        valid = True
        #copy the data across now
        scan = np.copy(self.data)        
        
        #check that we have the boundaries, if we dont will load them in
        self.check_boundaries()
        
        #get the fibroglandular mask
        self.fibroglandular_segmentation()
        
        #if there has been a problem with the preprocessed data, set the valid flag to false
        #so we skip this scan
        if(np.nansum(scan) < 1000):
            valid = False
            
            
        #turn everything thats nan to zero
        scan[np.isnan(scan)] = 0.0
        #see if we are using the regions found
        
        #sorting out the bounding boxes
        #they come to us as a single element numpy array of 2-D lists
        #this is confusing, but it allows us to get all of the bounding boxes for this
        #scan in one pop statement from the queue
        #so the format is like
        #   bbs = np.array([[xmin, ymin, xmax, ymax], ... , [xmin, ymin, xmax, ymax]])
        #to make the code neater now, am going to get rid of the numpy array layer
        #so it is bbs = 2d list
        bbs = bbs[0]
        #similar deal for the confidence values, is np array of 1d lists though
        conf = conf[0]
        
        #if the first element of bbs is set to -1, it means we aren't 
        if(bbs[0][0] != -1):
            
            #if the confidence values are equal to minus one, then we are using region extraction,
            #we just didnt find any suspicious regions for this image
            if(conf[0] == -1):
                print('Didnt find any regions for %s' %self.file_path)                
                #dont extract features from any region as none was found
                pass
            else:
                
                #lets crop the image so we only find features from this region
                best_bb = self.find_best_bb(bbs, conf)
                #if we did actually find a suitable bounding box
                if(best_bb[0] != -1):
                    scan = scan[best_bb[1]:best_bb[3], best_bb[0]:best_bb[2]]
                    self.plot_rcnn(best_bb)
                else:
                    #set the valid flag to false
                    valid = False
                
        if(bbs[0][0] != -1) & (valid):
            self.packets = pywt.WaveletPacket2D(data=scan, wavelet=self.wavelet_type, mode='sym')
            
            #check that the number of levels isnt to high
            #if it is, lets print something to let them know
            if(self.levels > self.packets.maxlevel):
                
                #skip this one
                valid = False
                
            else:
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
                    
                #estimate the density of the breast
                self.density.append(np.divide(np.nansum(self.data[self.fibroglandular_mask]), np.nansum( self.data[np.isfinite(self.data)] > 1)))

            
        #return the boolean that said whether the scan was valid or not
        return valid
    
    
    
    
    
    
    
    
    """
    find_best_bb():
    
    Description:
    In finding the best region, will just make sure that we arent including any unnesecary artifacts
    such as the nipple. To exclude the nipple and other artifacts near the edge of the breast,
    will compare the found regions with that of the breast boundary
    
    If preprocessing wasnt done in this step, will load the boundary data from file
    """
    
    def find_best_bb(self, bbs, conf):
        
        good_bbs = []
        good_conf = []
        
        #make sure the boundaries are in an array format
        boundary = np.array(self.boundary)
        boundary_y = np.array(self.boundary_y)
            
        #now lets loop over all of the bounding boxes
        #will check if the bounding box covers the boundary by seeing if the x value
        #of the boundary and the y value of the boundary appear in the same location
        
        for ii in range(0, len(bbs)):
            x = (boundary > bbs[ii][0]) &  (boundary < bbs[ii][2])
            y = (boundary_y > bbs[ii][1]) &  (boundary_y < bbs[ii][3])
            
            on_boundary = np.sum(x & y)
            
            if not on_boundary:
                good_bbs.append(bbs[ii])
                good_conf.append(conf[ii])
                
                
        if(len(good_conf) == 0):
            return [np.array(-1)]
        
        else:
            #now lets find the bounding box with the highest confidence
            best = np.where(np.array(good_conf) == np.max(good_conf))[0][0]
            return good_bbs[best]
        
        
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
                
                #print('ii =  %d, jj = %d, indicies = %s' %(ii,jj,self.indicies[level][ii,jj]))
                
                #im = self.packets[self.indicies[level][ii,jj]].data
                #fig = plt.figure()
                #ax2 = fig.add_subplot(1,1,1)
                #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
                #cax = ax2.imshow(im, cmap='gray')
                #plt.axis('off')
                #plt.tight_layout()
                #ax2.set_adjustable('box-forced')
                #fig.savefig(os.getcwd() + '/figs/wav_' + self.file_path[-10:-4] + '_' + str(ii) + '_' + str(jj) + '_'  + 'png')
                #plt.close()
                
                
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
        try:
            min_val = np.min(temp[temp_mask])
        except Exception:
            print temp
            print('level = %d, ii = %d, jj = %d' %(level,ii,jj))
            min_val = 0
            print(self.file_path)
            print('sum temp = %d' %(np.sum(temp[temp_mask])))
            print('sum temp_mask = %d' %(np.sum(temp_mask)))        
            temp_mask[temp_mask == False] = True
            temp[temp_mask] = 1
            
            
            """    
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            ax1.imshow(self.packets[self.indicies[level][ii,jj]].data)
            fig.savefig((os.getcwd() + '/figs/' + 'l_%d_i_%d_j_%d_data_' + self.file_path[-10:-3] + 'png') %(level, ii, jj))
            fig.clf()
            plt.close()
            """
            
            
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
        temp[temp == np.nan] = 0
        
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
        
        
        
    
    def reinitialise_feature_lists(self):
        
        #clear all  of the feature lists
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
        
        #now initialise the feature lists
        self._initialise_feature_lists()
        
        
        
    """
    _initialise_feature_lists()
    
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
    
    
    
    
    def _initialise_feature_lists(self):        
        
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
        
        
        
        
        
        
    def plot_rcnn(self, bbox):
        plt.cla()
        plt.axis('off')
        plt.imshow(self.data, cmap='gray')
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='r', linewidth=2))
        plt.savefig('./vis/%s.png' %os.path.basename(self.file_path), bbox_inches='tight')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    ###########################################################
    #
    # Just moved this code from breast.py, will test now to see if it is nicer
    #
    ##########################################################
    
    """
    fibroglandular_segmentation()
    
    Description:
    Will attempt to segment the fibroglandular disk out of the mammogram
    using the minimum cross entropy threshold method.
    
    Will not search near the breast boundary, as the skin component in the image can contain
    a high intensity pixels which will confound segmentation.
    
    
    """
    
    
    def fibroglandular_segmentation(self):
        
        #create a mask that contains pixels at and near the boundary
        #to do this, will just be blurring the boundary we have previously defined
        
        boundary_blur = np.zeros(self.data.shape)
        breast_pixels = np.zeros(self.data.shape)
        
        print self.data.shape
        print np.max(self.boundary_y)
        boundary_blur[self.boundary_y,self.boundary] = 1
        #blur with Gaussian filter that has large standard deviation
        boundary_blur = filters.gaussian_filter(boundary_blur,40) 
        boundary_blur[boundary_blur != 0] = 1
        #convert it to a boolean mask
        boundary_blur = boundary_blur.astype(np.bool)
        #use this to create an array of just the breast pixels, not any
        #of the skin
        breast_pixels[boundary_blur == False] = self.data[boundary_blur == False]
        #set the pixels at and near the skin boundary to nan
        breast_pixels[boundary_blur == True] = np.nan
        #pass this to the cross entropy threshold function to find the optimal
        #threshold that minimises cross entropy
        #threshold value is saved in the self.threshold member variable
        self.cross_entropy_threshold(breast_pixels)
        
        #now will use this to create a binary mask of the fibroglandular disk/dense tissue
        self.fibroglandular_mask = (self.data > self.threshold)
        self.fibroglandular_mask[boundary_blur == True] = False

        if(self.plot):
        
            fig = plt.figure()
            plt.axis('off')
            ax1 = fig.add_subplot(1,1,1)
            im1 = ax1.imshow(self.data, cmap='gray')
            fig.savefig(os.getcwd() + '/figs/' + 'im_' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            fig.clf()


            fig = plt.figure()
            plt.axis('off')
            ax1 = fig.add_subplot(1,1,1)
            im1 = ax1.imshow(self.fibroglandular_mask)
            fig.savefig(os.getcwd() + '/figs/' + 'fibro_' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            fig.clf()




            #fig = plt.figure()
            #ax1 = fig.add_subplot(1,1,1)
            #ax1.imshow(self.fibroglandular_mask)
            #fig.savefig(os.getcwd() + '/figs/' + 'fibro_' + self.file_path[-10:-3] + 'png')
            #fig.clf()
            #plt.close()
            
            
        
        
        
        
        
    """
    cross_entropy_threshold()
    
    Description:
    Creating adaptive threshold for segmenting fibroglandular disk
    and dense breast tissue.
    
    The function uses the minimum cross entropy between the two classes
    to find the threshold. Minimum cross entropy developed by Li, C. and Lee, C
    Is a wrapper function, the cross entropy (eta)  is calculated using a helper function
    
    
    Also creating a mask of the breast boundary as well. The breast skin has higher intensity, 
    and won't contain any useful information, so wont use the breast skin region for thresholding
    
    
    @param breast_pixels = preprocessed scan that has had skin removed, and the areas near the breast
                  boundary masked out so we only look at the breast tissue
    
    
    Reference
    
    @article{Li_1993,
            doi = {10.1016/0031-3203(93)90115-d},
            url = {http://dx.doi.org/10.1016/0031-3203(93)90115-d},
            year = 1993,
            month = {apr},
            publisher = {Elsevier {BV}},
            volume = {26},
            number = {4},
            pages = {617--625},
            author = {C.H. Li and C.K. Lee},
            title = {Minimum cross entropy thresholding},
            journal = {Pattern Recognition}
    }
    """
    
    def cross_entropy_threshold(self, breast_pixels):
        
        
        temp = breast_pixels[np.isfinite(breast_pixels)].ravel()
        #lets get rid of zero value pixels
        temp = temp[temp > 0]        
        eta = 10e10        #large value of eta that will be overwritten
        
        hist = np.histogram(temp, bins = 4096)
        hist = hist[0]
        for t in range(300, 3000):
            current_eta = self.calc_eta(t, hist)
            if(current_eta < eta):
                eta = current_eta
                self.threshold = t
                
                
        
        
        
        
        
    """
    calc_eta()
    
    Description:
    Helper function used to calculate the cross entropy (eta) for different threshold values
    Cross entropy calculations from paper referenced for cross_entropy_threshold()
    
    The cross entropy between the data above and below the threshod value is found
    
    @param t = threshold value
    @param hist = histogram of data
    
    @retval eta = cross entropy
    
    """
    
    def calc_eta(self, t, hist):
        
        #creating arrays for the two data sets
        mu_1_data = hist[1:t-1]
        mu_2_data = hist[t:-1]
        
        #creating index values
        mu_1_range = np.arange(1,t-1)
        mu_2_range = np.arange(t,4096-1)
        #numerator parts for cross entropy
        mu_1_num = mu_1_data*mu_1_range
        mu_2_num = np.multiply(mu_2_data, mu_2_range)
        
        mu_1 = (np.sum(mu_1_num)) / (np.sum(mu_1_data))
        mu_2 = (np.sum(mu_2_num)) / (np.sum(mu_2_data))
        
        #calculating cross entropy
        eta = np.sum( np.multiply(mu_1_num, np.divide(np.log(mu_1_range), mu_1))) + np.sum( np.multiply(mu_2_num, np.divide(np.log(mu_2_range), mu_2)))
        
        return eta                                  
    
    
    
    
    
    
    
    def check_boundaries(self):
    
        #if the boundary variables are empty, load the boundary
        #saved during preprocessing
        if(len(self.boundary) < 1):
            boundary_path = os.path.join(os.path.dirname(self.file_path), 'boundaries',
                                         os.path.basename(self.file_path))
            print boundary_path
            boundary_full = np.load(boundary_path)
            self.boundary = boundary_full[0,:].astype(np.int)
            self.boundary_y = boundary_full[1,:].astype(np.int)
            
