#!/usr/bin/python

import dicom
import os
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
from scipy import signal as signal
from scipy.optimize import minimize, rosen, rosen_der
from scipy.ndimage import filters as filters
from scipy.ndimage import measurements as measurements
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
import pywt
from skimage import measure
from sklearn import svm
from sklearn.externals import joblib
from skimage import feature
from skimage.morphology import disk
from skimage.filters import roberts, sobel, scharr, prewitt, threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi


import threading
import Queue
import time
import multiprocessing
from my_thread import my_thread

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet





"""
create_classifier()

Description:
just a helper function to put the features in a flat array
will change later as use improved features

@param threads = list of all threads that contains the features from each scan it processed
@param num_scans = the number of scans that were processed

@retval feature array and class label arrays to be used for SVM classifier

"""

def create_classifier_arrays(threads, num_scans):
    
    no_features = 6
    no_packets = 4
    X = np.zeros((num_scans, no_features * no_packets))
    Y = np.zeros((num_scans,1))
    
    ii = 0   #scan number index for all the scans
    t = 0    #this is the thread index. Once we have looked at all of the scans in the current
             #thread, will move on to the next one
    t_ii = 0 #scan index for the current thread
    
    while (ii < num_scans) & (t < len(threads)):    
        
        #get the features from the current scan in the current thread
        print(threads[t].scan_data.homogeneity[t_ii][2])
        print(np.shape(threads[t].scan_data.homogeneity[t_ii][2]))
        homogeneity = threads[t].scan_data.homogeneity[t_ii][2].reshape(1,-1)
        entropy = threads[t].scan_data.entropy[t_ii][2].reshape(1,-1)
        energy = threads[t].scan_data.energy[t_ii][2].reshape(1,-1)
        contrast = threads[t].scan_data.contrast[t_ii][2].reshape(1,-1)
        dissimilarity = threads[t].scan_data.dissimilarity[t_ii][2].reshape(1,-1)
        correlation = threads[t].scan_data.correlation[t_ii][2].reshape(1,-1)
        
        t_ii += 1
        #see if it is time to move on to the next thread
        if(t_ii == threads[t].scans_processed):
            #increment the thread index
            t += 1
            #set the scan index for the thread back to zero(the firt scan in the new thread)
            t_ii = 0
        
        for jj in range(0,no_packets):
            print('packet number = %d' %jj)
            X[ii, jj*no_features] = homogeneity[0,jj]
            X[ii, jj*no_features + 1] = entropy[0,jj]
            X[ii, jj*no_features + 2] = energy[0,jj]
            X[ii, jj*no_features + 3] = contrast[0,jj]
            X[ii, jj*no_features + 4] = dissimilarity[0,jj]
            X[ii, jj*no_features + 5] = correlation[0,jj]
            
            
        #set the cancer statues for this scan
        Y[ii,0] = threads[t].cancer_status[t_ii]    
        ii += 1
        
    #make Y a 1-d array so the SVM classifier can handle it properly
    #and also set it to a 1/0 binary array instead of True/False boolean array
    Y[Y == True] = 1
    y[Y == False] = 0
    Y = np.ravel(Y)
    return X,Y




#####################################################
#
#                Main Loop of Program
#
#####################################################

#start the program timer
program_start = time.clock()

descriptor = spreadsheet(training=True, run_synapse = False)
threads = []
id = 0
num_threads = multiprocessing.cpu_count() - 2

# Create new threads
for ii in range(0,num_threads):
    thread = my_thread(id, descriptor.no_scans)
    thread.start()
    threads.append(thread)
    id += 1
    
    
#setting up the queue for all of the threads, which contains the filenames
for ii in range(0, descriptor.no_scans):
    my_thread.t_lock.acquire()
    my_thread.q.put(descriptor.filenames[ii])
    my_thread.q_cancer.put(descriptor.cancer_list[ii])
    my_thread.t_lock.release()
    
    
#now some code to make sure it all runs until its done
#keep this main thread open until all is done
while (not my_thread.q.empty()):
    pass

#queue is empty so we are just about ready to finish up
#set the exit flag to true
my_thread.exit_flag = True
#wait until all threads are done
for t in threads:
    t.join()
    
    
#all of the threads are done, so we can now we can use the features found to train
#the classifier

#will be here after have run through everything
#lets save the features we found, and the file ID's of any that
#created any errors so I can have a look later
#will just convert the list of error files to a pandas database to look at later
error_database = pd.DataFrame(my_thread.error_files)
#will save the erroneous files as csv
error_database.to_csv('error_files.csv')

#now lets train our classifier
#will just use the features from the approximation wavelet decomps

X,Y = create_classifier_arrays(threads, descriptor.no_scans)
clf = svm.SVC()
clf.fit(X,Y)

#now will save the model, then can have a look how it all performed a try it out
joblib.dump(clf,'filename.pkl')

print('---- TIME INFORMATION ----')
#lets print the time info for each thread
print('Time for each thread to process a single scan')
for t in threads:
    print('Thread %d :' %(t.t_id))
    print('Average Time  = %f s' %(np.mean(t.processing_time)))
    print('Max Time  = %f s' %(np.max(t.processing_time)))
    print('Min Time  = %f s' %(np.min(t.processing_time)))
    print(' ')

#printing the total run time of the program
run_total_sec = time.clock() - program_start
run_hours = run_total_sec % 3600
run_mins = (run_total_sec - run_hours * 60) % 60
run_secs = (run_total_sec - run_hours * 3600 -  run_mins * 60)

print('Run Time for %d scans = %hours %d minutes and %f seconds' %(descriptor.no_scans, run_hours, run_mins, run_secs))

print "Exiting Main Thread"
