#!/usr/bin/python

import dicom
import os
import numpy as np
import pandas as pd
import sys
import getopt

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

from itertools import chain
import threading
import Queue
import timeit
from multiprocessing import Process, Lock, Queue, cpu_count
from my_thread import my_thread, shared, my_manager
from multiprocessing.managers import BaseManager
#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet
from log import logger
from arguments import arguments


def flatten(x):
    "Flatten one level of nesting"
    return chain.from_iterable(x)


def create_classifier_arrays(shared):
    
    #How do we want to split it
    val = 250
    
    #convert the list of features and class discriptors into arrays
    X = np.array(shared.get_feature_array())
    Y = np.array(shared.get_class_array())
    print (X.shape)
    
    X_t = X[0:-val,:]
    Y_t = Y[0:-val]
    
    X_val = X[-val::,:]
    Y_val = Y[-val::]
    
    return X_t, Y_t, X_val, Y_val




#####################################################
#
#                Main Loop of Program
#
#####################################################

#start the program timer
program_start = timeit.default_timer()
command_line_args = arguments(sys.argv[1::])
sys.stdout = logger(arguments.log_path)

descriptor = spreadsheet(training=True, run_synapse = False)
threads = []
id = 0
num_threads = cpu_count() - 2
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
    
    
# Create new threads
for ii in range(0,num_threads):
    thread = my_thread(id, descriptor.no_scans, shared)
    thread.start()
    threads.append(thread)
    id += 1
    
    
    
#now some code to make sure it all runs until its done
#keep this main thread open until all is done
while (not shared.get_exit_status()):
    #a = threading.activeCount() 
    #if(a != 15):
    #print a
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


X, Y, X_v, Y_v = create_classifier_arrays(shared)
clf = svm.SVC()
clf.fit(X,Y)

#now will save the model, then can have a look how it all performed a try it out
joblib.dump(clf,'filename.pkl')

test = clf.predict(X_v)

#find the accuracy
print((test == Y_v))
for ii in range(0, Y_v.size):
    print("%r : %r " %(Y_v[ii], test[ii]))
    
    
    
np.save('X', X)
np.save('Y', Y)
np.save('X_val', X_v)
np.save('Y_val', Y_v)


"""
print('---- TIME INFORMATION ----')
#lets print the time info for each thread
print('Time for each thread to process a single scan')
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


print "Exiting Main Thread"
