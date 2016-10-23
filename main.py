"""
main.py

Author: Ethan Goan
Queensland University of Technology
DREAM Mammography Challenge
2016

Description:

source file to run program for training SVM classifier for 
breast cancer detection. Uses breast, spreadsheet and feature classes
defined in read_files.py, breast.py and feature_extract.py, which contain
main functionality for reading in the files



"""


import dicom
import os
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
from scipy import signal as signal
from scipy.ndimage import filters as filters
from scipy.ndimage import measurements as measurements
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import pywt
from skimage import measure
from sklearn import svm
from sklearn.externals import joblib
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet




"""
create_classifier()

Description:
just a helper function to put the features in a flat array
will change later as use improved features

@param benign_scan = feature object containing data from benign scans
@param malignant_scan = feature object containing data from malignant scans

@retval feature array and class label arrays to be used for SVM classifier

"""

def create_classifier_arrays(benign_scan, malignant_scan):
    no_features = 6
    no_packets = 4
    X = np.zeros((benign_scan.no_images + malignant_scan.no_images, no_features * no_packets)) 
    Y = np.zeros((1,benign_scan.no_images + malignant_scan.no_images))
    Y[0:benign_scan.no_images] = 1
    
    for ii in range(0,benign_scan.no_images):
        homogeneity = benign_scan.homogeneity[ii][2].reshape(1,-1)
        entropy = benign_scan.entropy[ii][2].reshape(1,-1)
        energy = benign_scan.energy[ii][2].reshape(1,-1)
        contrast = benign_scan.contrast[ii][2].reshape(1,-1)
        dissimilarity = benign_scan.dissimilarity[ii][2].reshape(1,-1)
        correlation = benign_scan.correlation[ii][2].reshape(1,-1)
        for jj in range(0,no_packets):
            X[ii, jj*no_features] = homogeneity[0,jj]
            X[ii, jj*no_features + 1] = entropy[0,jj]
            X[ii, jj*no_features + 2] = energy[0,jj]
            X[ii, jj*no_features + 3] = contrast[0,jj]
            X[ii, jj*no_features + 4] = dissimilarity[0,jj]            
            X[ii, jj*no_features + 5] = correlation[0,jj]
            

    for ii in range(benign_scan.no_images, benign_scan.no_images + malignant_scan.no_images ):
        homogeneity = malignant_scan.homogeneity[ii][2].reshape(1,-1)
        entropy = malignant_scan.entropy[ii][2].reshape(1,-1)
        energy = malignant_scan.energy[ii][2].reshape(1,-1)
        contrast = malignant_scan.contrast[ii][2].reshape(1,-1)
        dissimilarity = malignant_scan.dissimilarity[ii][2].reshape(1,-1)
        correlation = malignant_scan.correlation[ii][2].reshape(1,-1)
        for jj in range(0,no_packets):
            X[ii, jj*no_features] = homogeneity[0,jj]
            X[ii, jj*no_features + 1] = entropy[0,jj]
            X[ii, jj*no_features + 2] = energy[0,jj]
            X[ii, jj*no_features + 3] = contrast[0,jj]
            X[ii, jj*no_features + 4] = dissimilarity[0,jj]            
            X[ii, jj*no_features + 5] = correlation[0,jj]
            
    return X,Y        




RUN_SYNAPSE = False




descriptor = spreadsheet(benign_files=True, run_synapse = RUN_SYNAPSE)

#creating a list that will store the scan filename if any of them create an error
#will then save this list and have a look to see where they failed
error_files = []
benign_scan = feature(levels = 3, wavelet_type = 'haar', no_images = descriptor.benign_count)



while(descriptor.exam_pos < descriptor.total_no_exams):
    #load in a file
    file_path = descriptor.next_scan()
    print(file_path)
    try:
        benign_scan.initialise(file_path)
        benign_scan.preprocessing()
        benign_scan.get_features()
        plt.figure()
        #plt.imshow(benign_scan.data)
        plt.imshow(benign_scan.packets[benign_scan.indicies[2][0,0]].data)
        plt.show()
    except:
        print('Error with current file %s' %(file_path))
        error_files.append(file_path)



#load in the malignant scans
descriptor = spreadsheet(benign_files=False, run_synapse = RUN_SYNAPSE)
malignant_scan = feature(levels = 3, wavelet_type = 'haar', benign_scans = False, no_images = descriptor.malignant_count)


while(descriptor.exam_pos < descriptor.total_no_exams):
    #load in a file
    file_path = descriptor.next_scan()
    print(file_path)
    try:
        malignant_scan.initialise(file_path)
        malignant_scan.preprocessing()
        malignant_scan.get_features()
    except:
        print('Error with current file %s' %(file_path))
        error_files.append(file_path)
        

#will be here after have run through everything
#lets save the features we found, and the file ID's of any that
#created any errors so I can have a look later
#will just convert the list of error files to a pandas database to look at later
error_database = pd.DataFrame(error_files)
#will save the erroneous files as csv
error_database.to_csv('error_files.csv')

#now lets train our classifier
#will just use the features from the approximation wavelet decomps

X , Y = create_classifier_arrays(benign_scan, malignant_scan)
clf = svm.SVC()
clf.fit(X,Y)


#now will save the model, then can have a look how it all performed a try it out
joblib.dump(clf,'filename.pkl')















            
## some old file paths that can be handy whilst testing
#file_path = '/home/ethan/DREAM/pilot_images/111359.dcm' #image without pectoral muscle
#file_path = '/home/ethan/DREAM/pilot_images/134060.dcm' #image with pectoral muscle
#file_path = '/home/ethan/DREAM/pilot_images/502860.dcm' #malignant case
#file_path = '/home/ethan/DREAM/pilot_images/151849.dcm' #malignant case with pectoral muscle
#file_path = '/home/ethan/DREAM/pilot_images/485343.dcm' #scan that broke my initial boundary scan
