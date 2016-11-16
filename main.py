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

@param scan_data = feature object containing data from all scans
@param cancer_status = list that holds the status of the cancer in a scan

@retval feature array and class label arrays to be used for SVM classifier

"""

def create_classifier_arrays(scan_data, cancer_status):

    num_files = len(cancer_status)
    no_features = 6
    no_packets = 4
    X = np.zeros((num_files, no_features * no_packets)) 
    Y = np.zeros((num_files,1))
    Y[cancer_status == True] = 1
    Y = np.ravel(Y)
    
    for ii in range(0,num_files):
        homogeneity = scan_data.homogeneity[ii][2].reshape(1,-1)
        entropy = scan_data.entropy[ii][2].reshape(1,-1)
        energy = scan_data.energy[ii][2].reshape(1,-1)
        contrast = scan_data.contrast[ii][2].reshape(1,-1)
        dissimilarity = scan_data.dissimilarity[ii][2].reshape(1,-1)
        correlation = scan_data.correlation[ii][2].reshape(1,-1)
        for jj in range(0,no_packets):
            X[ii, jj*no_features] = homogeneity[0,jj]
            X[ii, jj*no_features + 1] = entropy[0,jj]
            X[ii, jj*no_features + 2] = energy[0,jj]
            X[ii, jj*no_features + 3] = contrast[0,jj]
            X[ii, jj*no_features + 4] = dissimilarity[0,jj]            
            X[ii, jj*no_features + 5] = correlation[0,jj]
                   
    return X,Y


RUN_SYNAPSE = False
descriptor = spreadsheet(training=True, run_synapse = RUN_SYNAPSE)

#creating a list that will store the scan filename if any of them create an error
#will then save this list and have a look to see where they failed
error_files = []
cancer_status = [] #list that will hold the cancer status of the scans
scan_data = feature(levels = 3, wavelet_type = 'haar', no_images = descriptor.no_scans)





while(descriptor.file_pos < descriptor.no_scans):

    #load in a file
    file_path = descriptor.next_scan()
    print(file_path)
    try:
        scan_data.initialise(file_path)
        scan_data.preprocessing()
        scan_data.get_features()
        cancer_status.append(descriptor.cancer)

        #plotting some figures here, can uncomment to see how the preprocessing is going, and a histogram of the mammogram
        #plt.figure()
        #plt.subplot(211)
        #hist, bin_edges = np.histogram(scan_data.data[np.isfinite(scan_data.data)], bins=400)
        #plt.plot(hist)
        #plt.xlim([1,len(hist)])
        #plt.ylim([0, np.max(hist[1:-1]) * 1.1])
        #plt.subplot(212)
        #scan_data.data[scan_data.data < 600] = 0
        #plt.imshow(scan_data.data)
        #plt.show()
        #plt.title('histogram of breast tissue')

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

X,Y = create_classifier_arrays(scan_data, cancer_status)
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
