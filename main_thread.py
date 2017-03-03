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
from sklearn.datasets import dump_svmlight_file
from skimage import feature
from skimage.morphology import disk
from skimage.filters import roberts, sobel, scharr, prewitt, threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi
import subprocess

from itertools import chain
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


def create_classifier_arrays(shared, validation):
    #How do we want to split it
    if(validation):
        val = validation
    else:
        val = 1
        
    #convert the list of features and class discriptors into arrays
    X = np.array(shared.get_feature_array())
    Y = np.array(shared.get_class_array())
    
    X_t = X[0:-val,:]
    Y_t = Y[0:-val]
    
    X_val = X[-val::,:]
    Y_val = Y[-val::]
    
    return X_t, Y_t, X_val, Y_val


def terminal_cmd(command):
    print(command)
    os.system(command)
    
    
    
    
    
def begin_processes(command_line_args):
    
    #sys.stdout = logger(command_line_args.log_path)
    descriptor = spreadsheet(command_line_args)
    threads = []
    id = 0
    num_threads = cpu_count() - 2
    print("Number of processes to be initiated = %d" %num_threads )
    
    man = my_manager()
    man.start()
    shared = man.shared()
    
    #setting up the queue for all of the threads, which contains the filenames
    print("Setting up Queue")
    print descriptor.no_scans
    for ii in range(0, descriptor.no_scans):
        shared.t_lock_acquire()
        shared.q_put(descriptor.filenames[ii])
        #if we are training, add the descriptor for cancer status
        if(command_line_args.training):
            shared.q_cancer_put(descriptor.cancer_list[ii])
        shared.t_lock_release()
        
    print('Set up Queue')
    
    # Create new threads
    print('Creating threads')
    for ii in range(0,num_threads):
        sys.stdout.flush()
        thread = my_thread(id, descriptor.no_scans, shared, command_line_args)
        thread.start()
        threads.append(thread)
        id += 1
        
    print('Started Threads')
        
    #now some code to make sure it all runs until its done
    #keep this main thread open until all is done
    while (not shared.get_exit_status()):
        #just so we flush all print statements to stdout
        sys.stdout.flush()
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
    #error_database.to_csv('error_files.csv')
    
    
    #now lets train our classifier
    #will just use the features from the approximation wavelet decomps
    
    X, Y, X_v, Y_v = create_classifier_arrays(shared, command_line_args.validation)
    #save this data in numpy format, and in the LIBSVM format
    print('Saving the final data')
    np.save(command_line_args.save_path + '/model_data/X', X)
    np.save(command_line_args.save_path + '/model_data/Y', Y)
    np.save(command_line_args.log_path + './X_val', X_v)
    np.save(command_line_args.log_path + './Y_val', Y_v)
    dump_svmlight_file(X,Y,command_line_args.save_path + '/model_data/train_file_libsvm')
    dump_svmlight_file(X_v,Y_v,'./predict_file_libsvm')
    
    
    
    
    
"""
preprocessing()

Description:

Will call the function to begin all threads and start preprocessing all of the data

"""    

def preprocessing(command_line_args):
    
    #basically can just run the main thread
    begin_processes(command_line_args)
    
    
    
    
    
    
    
"""
train_model()

Description:
If the command line arguments specified that we should train the model,
this function will be called. Uses the GPU enhanced version of LIBSVM to train the model.

In the command_line_args object, a path to a file containing the variables to train the models is found.

"""


def train_model(command_line_agrs):

    #if we are forcing to capture the features, we will do so now.
    #otherwise we will use the ones found during initial preprocessing
    if(command_line_args.extract_features):
        begin_processes(command_line_args)
        
    #if we aren't extracting files, lets check that there is already
    #a file in libsvm format. If there isn't we will make one
    elif(not os.path.isfile(command_line_args.input_path + '/model_data/train_file_libsvm')):
        #if this file doesn't exist, we have two options
        #use the near incomplete features if the preprocessing run finished completely
        #otherwise use the nearly complete feature set
        if( os.path.isfile(command_line_args.input_path + '/model_data/X.npy')):
            X = np.load(command_line_args.input_path + '/model_data/X.npy')
            Y = np.load(command_line_args.input_path + '/model_data/Y.npy')
        else:
            X = np.load(command_line_args.input_path + '/model_data/X_temp.npy')
            Y = np.load(command_line_args.input_path + '/model_data/Y_temp.npy')
            
            
        #now lets make an libsvm compatable file         
        dump_svmlight_file(X,Y,command_line_args.save_path + '/model_data/train_file_libsvm')
        
        
    #now features are extracted, lets classify using this bad boy
    terminal_cmd(command_line_args.train_string)
    
    
def validate_model(command_line_args):
    terminal_cmd(command_line_args.validation_string)
    
    
#####################################################
#
#                Main Loop of Program
#
#####################################################


if __name__ == '__main__':
    #start the program timer
    program_start = timeit.default_timer()
    command_line_args = arguments(sys.argv[1:])
    
    if(command_line_args.preprocessing):
        preprocessing(command_line_args)
        
    #if we want to train the model, lets start training
    if(command_line_args.training):
        train_model(command_line_args)
        
        
    #if we want to do some validation, lets dooo it :)
    if(command_line_args.validation):
        validate_model(command_line_args)
        
        
        
        
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
    
    
    print("Exiting Main Thread")
