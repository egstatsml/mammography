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
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
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


def create_classifier_arrays(shared):
        
    #convert the list of features and class discriptors into arrays
    print shared.get_laterality_array()
    X = np.array(shared.get_feature_array())
    Y = np.array(shared.get_class_array())
    lateralities = np.array(shared.get_laterality_array())
    exams = np.array(shared.get_exam_array(), dtype=np.int)
    subject_ids = np.array(shared.get_subject_id_array())
    
    return X, Y, lateralities, exams, subject_ids




def load_results(command_line_args):
    with open(command_line_args.save_path + '/model_data/results.txt') as f:
        lines = f.read().splitlines()
        
        
    #now map the results to an array of ints
    results = np.array( map(float, lines) )
    print results
    return results


def terminal_cmd(command):
    print(command)
    os.system(command)
    
    
    
    
    
def begin_processes(command_line_args, descriptor):
    #initialise list of threads
    threads = []
    id = 0
    num_threads = cpu_count() - 2
    print("Number of processes to be initiated = %d" %num_threads )
    
    man = my_manager()
    man.start()
    shared = man.shared()

    print("Number of Scans = %d" %(descriptor.no_scans))
    #setting up the queue for all of the threads, which contains the filenames
    #also add everything for the metadata queues
    print("Setting up Queues")
    
    for ii in range(0, descriptor.no_scans):
        shared.t_lock_acquire()
        shared.q_put(descriptor.filenames[ii])
        shared.q_cancer_put(descriptor.cancer_list[ii])
        shared.q_laterality_put(descriptor.laterality_list[ii])
        shared.q_exam_put(descriptor.exam_list[ii])
        shared.q_subject_id_put(descriptor.subject_id_list[ii])
        shared.t_lock_release()        
    print('Set up Queues')
    
    
    #Create new threads
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
    
    
    #now lets save the features found
    #will just use the features from the approximation wavelet decomps
    
    X, Y, lateralities, exams, subject_ids = create_classifier_arrays(shared)
    pca = PCA(n_components=20)#, whiten=True)
    X = pca.fit_transform(X)
    #print X.shape
    
    #save this data in numpy format, and in the LIBSVM format
    print('Saving the final data')
    np.save(command_line_args.save_path + '/model_data/X', X)
    np.save(command_line_args.save_path + '/model_data/Y', Y)
    np.save(command_line_args.save_path + '/model_data/lateralities', lateralities)
    np.save(command_line_args.save_path + '/model_data/exams', exams)
    np.save(command_line_args.save_path + '/model_data/subject_ids', subject_ids)
    dump_svmlight_file(X,Y,command_line_args.save_path + '/model_data/data_file_libsvm')
    
    
    
    
"""
preprocessing()

Description:

Will call the function to begin all threads and start preprocessing all of the data

"""    

def preprocessing(command_line_args, descriptor):
    
    #basically can just run all the processes
    begin_processes(command_line_args, descriptor)
    
    
    
    
    
    
    
"""
train_model()

Description:
If the command line arguments specified that we should train the model,
this function will be called. Uses the GPU enhanced version of LIBSVM to train the model.

In the command_line_args object, a path to a file containing the variables to train the models is found.

"""


def train_model(command_line_args, descriptor):
    
    #if we are forcing to capture the features, we will do so now.
    #otherwise we will use the ones found during initial preprocessing
    #force this with the -f argument 
    print command_line_args.extract_features
    if(command_line_args.extract_features):
        begin_processes(command_line_args, descriptor)
        
    #if we aren't extracting files, lets check that there is already
    #a file in libsvm format. If there isn't we will make one
    
    elif(not os.path.isfile(command_line_args.input_path + '/model_data/train_file_libsvm')):
        #if this file doesn't exist, we have two options
        #use the near incomplete features if the preprocessing run finished completely
        #otherwise use the nearly complete feature set
        if( os.path.isfile(command_line_args.model_path + '/X.npy')):
            X = np.load(command_line_args.model_path + '/X.npy')
            Y = np.load(command_line_args.model_path + '/Y.npy')
        else:
            X = np.load(command_line_args.model_path + '/X_temp.npy')
            Y = np.load(command_line_args.model_path + '/Y_temp.npy')
            
            
        #now lets make an libsvm compatable file         
        dump_svmlight_file(X,Y,command_line_args.model_path + '/train_file_libsvm')
        
        
    #now features are extracted, lets classify using this bad boy
    terminal_cmd(command_line_args.train_string)
    
    
    
    
    
    
def validate_model(command_line_args, descriptor):
    
    #if preprocessing was specified in this run (-p) than tho model will
    #have first preprocessed the data
    #may want to just extract features though, if we do, we have that ability
    if(command_line_args.extract_features):
        begin_processes(command_line_args, descriptor)
    
    #Otherwise will assume that preprocessing has already been done prior
    
    #run validation
    terminal_cmd(command_line_args.validation_string)
    #now load in the validation data
    #so we can group the prediction value from each of the scans
    #to get an overall prediction value for individual breasts
    lateralities = np.load(command_line_args.save_path + '/model_data/lateralities.npy')
    exams = np.load(command_line_args.save_path + '/model_data/exams.npy')
    subject_ids = np.load(command_line_args.save_path + '/model_data/subject_ids.npy')
    #load in the predicted values
    predicted = load_results(command_line_args)
    print np.shape(predicted)
    #load in the actual class values
    #WILL ONLY BE USEFUL IF VALIDATING ON OUR DATA
    #OTHERWISE I HAVE JUST SET THEM ALL TO ZERO :)
    actual = np.load(command_line_args.save_path + '/model_data/Y.npy')
    
    #creating a set of all the subjects used
    #set is used so we don't have repeated values
    subject_ids_set = set(subject_ids)
    
    #now will loop over all of the subjects in the validation set
    #will find scans for each breast and each examination
    #this will give us a prediction for that single breast of that patient per
    #single exam
    
    predicted_breast = []   #prediction based on individual breasts in the data set
    actual_breast = []     #actual classifier value based
                           #WILL ONLY BE USEFUL IF VALIDATING ON OUR DATA 
                           
                           
    for subject in subject_ids_set:
    #am going to initialise a mask of all the positions of scans for the current patient
        subject_mask = subject_ids == subject
        #now find the number of exams this patient has in the validation set
        num_exams = np.max(exams[subject_mask])
        #now iterate over all of these exams to get the data for these
        #start at 1 since the exams are not zero indexed
        for exam in range(1, num_exams+1):
            #now will loop over left and right
            for laterality in ['L', 'R']:
                #find a mask of all the scans from this patient
                #in this exam and this breast
                mask = subject_mask & (exams == exam) & (lateralities == laterality)
                #now use this to find the mean predicted score for this breast
                #but only try and do this is there is at least a single scan found in the final mask
                if(np.sum(mask) > 0):
                    #add the mean of the prediction value found from these scans to the list
                    predicted_breast.append( np.mean(predicted[mask]) )
                    actual_breast.append( np.mean(actual[mask]) )
                    
    print predicted_breast
    #now will find the AUROC score
    print('Area Under the Curve Prediction Score = %f' %(roc_auc_score(actual_breast, predicted_breast))) 
    
    
#####################################################
#
#                Main of Program
#
#####################################################


if __name__ == '__main__':
    #start the program timer
    program_start = timeit.default_timer()
    command_line_args = arguments(sys.argv[1:])
    
    #if we are preprocessing or validating, we will need the descriptor
    #with all of the relevant information
    #if we are just training, we shouldn't have to worry about it,
    #as we currently will have saved all of that data
    #
    #The only time we would want to redo this when training is if we are
    #going to extract features again.
    #Normally loading in this data again even if we weren't going to use it wouldn't be a big deal,
    #but when doing it as part of the challenge, since there are so many scans it can take a while,
    #so don't want to do it if I don't have to
    
    descriptor = []
    
    if(command_line_args.preprocessing | command_line_args.validation | command_line_args.extract_features):
        descriptor = spreadsheet(command_line_args)
    
    
    if(command_line_args.preprocessing):
        preprocessing(command_line_args, descriptor)
        
        
    #if we want to train the model, lets start training
    if(command_line_args.training):
        train_model(command_line_args, descriptor)
        
        
    #if we want to do some validation, lets dooo it :)
    if(command_line_args.validation):
        validate_model(command_line_args, descriptor)
        
        
    print("Exiting Main Thread")
