#!/usr/bin/python
import dicom
import os
import numpy as np
import pandas as pd
import sys

#this allows us to create figures over ssh and save to file
import matplotlib as mpl
mpl.use('Agg')
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
import timeit
from multiprocessing import Process, Lock, Queue, cpu_count
from my_thread import my_thread, shared, my_manager, feature_manager, feature_data
from multiprocessing.managers import BaseManager
#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet
from log import logger

from itertools import chain

def flatten(x):
    "Flatten one level of nesting"
    #if it is an int, dont need it flatten, just send it back
    if type(x) == int:
        print x
        return x
    
    else:
        print 'here'
        return chain.from_iterable(x)
    
    
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
    
    no_features = 8
    no_packets = 4
    levels = 2
    X = []
    #X = np.zeros((num_scans, no_features * no_packets * levels))
    Y = np.zeros(num_scans)
    
    ii = 0   #scan number index for all the scans
    t = 0    #this is the thread index. Once we have looked at all of the scans in the current
             #thread, will move on to the next one
    t_ii = 0 #scan index for the current thread
    first = True        
    
    
    
    
    
    while (ii < num_scans) & (t < len(threads) - 1):    
        X.append([])
        prev_no_feats = 0   #this will tell us the previous number of features available
        #see if we need to get the features from the next thread
        if(first | (t_ii >= threads[t].feature_manager.get_scans_processed()-1)):
            
            #increment the thread index
            t += 1
            #set the scan index for the thread back to zero(the firt scan in the new thread)
            t_ii = 0
            homogeneity_all = threads[t].feature_manager.get_homogeneity()
            entropy_all = threads[t].feature_manager.get_entropy()
            energy_all = threads[t].feature_manager.get_energy()
            contrast_all = threads[t].feature_manager.get_contrast()
            dissimilarity_all = threads[t].feature_manager.get_dissimilarity()
            correlation_all = threads[t].feature_manager.get_correlation()
            wave_entropy_all = threads[t].feature_manager.get_wave_entropy()
            wave_energy_all = threads[t].feature_manager.get_wave_energy()
            first = False
            
           
        for lvl in range(0, levels):
            #get the features from the current scan in the current thread
            homogeneity = homogeneity_all[t_ii][lvl]
            entropy = entropy_all[t_ii][lvl]
            energy = energy_all[t_ii][lvl]
            contrast = contrast_all[t_ii][lvl]
            dissimilarity = dissimilarity_all[t_ii][lvl]
            correlation = correlation_all[t_ii][lvl]
            wave_energy = wave_energy_all[t_ii][lvl]
            wave_entropy = wave_entropy_all[t_ii][lvl]
            
            #wave_kurtosis = threads[t].scan_data.wave_kurtosis[t_ii][lvl]
            
            
            X[ii].extend(flatten(homogeneity))
            X[ii].extend(flatten(entropy))
            X[ii].extend(flatten(energy))
            X[ii].extend(flatten(contrast))
            X[ii].extend(flatten(dissimilarity))
            X[ii].extend(flatten(correlation))
            X[ii].extend(flatten(wave_energy))
            X[ii].extend(flatten(wave_entropy))
            #X[ii].extend(flatten(wave_kurtosis))
            #X[ii, lvl + lvl*jj*no_features] = homogeneity[0,jj]
            #X[ii, jj*no_features + 1 + kk] = entropy[0,jj]
            #X[ii, jj*no_features + 2*feat_len] = energy[0,jj]
            #X[ii, jj*no_features + 3*feat_len] = contrast[0,jj]
            #X[ii, jj*no_features + 4*feat_len] = dissimilarity[0,jj]
            #X[ii, jj*no_features + 5*feat_len] = correlation[0,jj]
            
            
        #set the cancer statues for this scan
        print('t = %d' %(t))
        print('t_ii = %d' %(t_ii))
        
        
        Y[ii] = threads[t].feature_manager.get_cancer_status_individual(t_ii)
        #increment scan counter
        ii += 1
        #increment the scan counter for this thread
        t_ii += 1
        
    #make Y a 1-d array so the SVM classifier can handle it properly
    #and also set it to a 1/0 binary array instead of True/False boolean array
    
    Y[Y == True] = 1
    Y[Y == False] = 0
    
    #have to convert 2d list for X to an array first
    #print X
    print np.shape(X)
    X = np.array(X)
    
    return X, Y[0:X.shape[0]]




def get_performance(test, Y_classify):
    test_b = np.copy(test).astype(bool)
    Y_classify_b = np.copy(Y_classify).astype(bool)
    
    acc = accuracy(test_b, Y_classify_b)
    sens = sensitivity(test_b, Y_classify_b)
    spec = specificity(test_b, Y_classify_b)
    return acc, sens, spec



def accuracy(test, Y_classify):
    num  = np.sum(test == Y_classify)
    den = np.size(test)
    return np.divide(num, den)


def sensitivity(test, Y_classify):
    num  = np.sum(test & Y_classify)
    den = num + np.sum(test & (not Y_classify))
    return np.divide(num, den)

def specificity(test, Y_classify):
    num  = np.sum((not test) & (not Y_classify))
    den = num + np.sum((not test) & Y_classify)
    return np.divide(num, den)









#####################################################
#
#                Main Loop of Program
#
#####################################################

#start the program timer
program_start = timeit.default_timer()

sys.stdout = logger()

descriptor = spreadsheet(training=True, run_synapse = False)
threads = []
id = 0
num_threads = cpu_count() - 4
print num_threads    

#my_manager.register('shared',shared)


man = my_manager()
man.start()
shared = man.shared()



#setting up the queue for all of the threads, which contains the filenames
for ii in range(0, descriptor.no_scans):
    shared.t_lock_acquire()
    shared.q_put(descriptor.filenames[ii])
    shared.q_cancer_put(descriptor.cancer_list[ii])
    shared.t_lock_release()
    
    
#Create new threads
for ii in range(0,num_threads):
    temp = feature_manager()
    temp.start()
    feats = temp.feature_data()
    thread = my_thread(id, descriptor.no_scans, shared, feats)
    thread.start()
    threads.append(thread)
    id += 1
    
    
    
#now some code to make sure it all runs until its done
#keep this main thread open until all is done
while (not shared.get_exit_status()):
    pass
    
    
    
#queue is empty so we are just about ready to finish up
#set the exit flag to true

#wait until all threads are done
for t in threads:
    t.join()
    
    
    
#all of the threads are done, so we can now we can use the features found to train
#the classifier

#will be here after have run through everything
#lets save the features we found, and the file ID's of any that
#created any errors so I can have a look later
#will just convert the list of error files to a pandas database to look at later
error_database = pd.DataFrame(shared.get_error_files())
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
"""
for t in threads:
    print('Thread %d :' %(t.t_id))
    print('Average Time  = %f s' %(np.mean(t.time_process)))
    print('Max Time  = %f s' %(np.max(t.time_process)))
    print('Min Time  = %f s' %(np.min(t.time_process)))
    print(' ')
    
#printing the total run time of the program
run_total_sec = timeit.default_timer() - program_start
run_hours = run_total_sec / 3600
run_mins = (run_total_sec - run_hours * 60) / 60
run_secs = (run_total_sec - run_hours * 3600 -  run_mins * 60)


print('Run Time for %d scans = %hours %d minutes and %f seconds' %(descriptor.no_scans, run_hours, run_mins, run_secs))
"""



#################################################
# 
#                Validation
#
#################################################

#initialising the threads
my_thread.exit_flag = False
my_thread.error_files = []   #list that will have all of the files that we failed to process
my_thread.cancer_status = []
my_thread.scan_no = 0
my_thread.cancer_count = 0


threads = []
id = 0
shared.set_exit_status(False)
shared.set_cancer_count(0)
#scans validation set
scans_in_val = shared.q_size()

#Create new threads
for ii in range(0,num_threads):
    temp = feature_manager()
    temp.start()
    feats = temp.feature_data()
    thread = my_thread(id, scans_in_val, shared, feats)
    thread.start()
    threads.append(thread)
    id += 1
    
    
#now some code to make sure it all runs until its done
#keep this main thread open until all is done
while(not shared.get_exit_status()):
    pass


#queue is empty so we are just about ready to finish up
#set the exit flag to true
my_thread.exit_flag = True
#wait until all threads are done
for t in threads:
    t.join()
    
print('scans_in_val = %d' %scans_in_val)    
X_classify,Y_classify = create_classifier_arrays(threads, scans_in_val )

test = clf.predict(X_classify)

#find the accuracy
print((test == Y_classify))
for ii in range(0, len(Y_classify)):
    print("%r : %r " %(Y_classify[ii], test[ii]))

acc, sens, spec = get_performance(test, Y_classify)

np.save('X', X)
np.save('Y', Y)
np.save('X_classify', X_classify)
np.save('Y_classify', Y_classify)


print "Exiting Main Thread"
