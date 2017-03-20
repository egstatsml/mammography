#!/usr/bin/python
import threading
import timeit
import time
import gc
from multiprocessing import Process, Lock, Queue
from multiprocessing.managers import BaseManager

import os
#import psutil


import dicom
import os
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
import networkx as nx
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
#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet


#for tracking memory
#from pympler.tracker import SummaryTracker
#from pympler.asizeof import asizeof



"""
my_thread:

Description: 
class that holds all of the data and methods to be used within each thread for 
processing and training the model. This class uses threading.Thread as the base class,
and extends on it's functionality to make it suitable for our use.

Memeber variables are included such as the feature object defined in feature_extract.py
Each thread will have a feature object as a member variable, and will use this member
to do the bulk of the processing. This class is defined purely to parellelize the workload
over multiple cores.


Some reference variables such as a queue, lock and exit flag are shared statically amongst all of 
the instances. Doing this just wraps everything a lot nicer.
"""


class shared(object):
    #some reference variables that will be shared by all of the individual thread objects
    #whenever any of these reference variables are accessed, the thread lock should be applied,
    #to ensure multiple threads arent accessing the same memory at the same time
    
    q = Queue(350000)             #queue that will contain the filenames of the scans
    q_cancer = Queue(350000)      #queue that will contain the cancer status of the scans
    q_laterality = Queue(350000)  #queue that will contain the laterality (view) of the scans
    q_exam = Queue(350000)        #queue that will contain the exam index of the scans
    q_subject_id = Queue(350000)  #queue that will contain the subject ID of the scans
    q_bc_history = Queue(350000)
    q_bc_first_degree_history = Queue(350000)
    q_bc_first_degree_history_50 = Queue(350000)
    q_anti_estrogen = Queue(350000)
    
    t_lock = Lock()
    exit_flag = False
    error_files = []   #list that will have all of the files that we failed to process
    scan_no = 0
    cancer_count = 0
    benign_count = 0
    feature_array = []
    class_array = []
    laterality_array = []
    exam_array = []
    subject_id_array = []
    bc_history_array = []
    bc_first_degree_array = []
    bc_first_degree_50_array = []
    anti_estrogen_array = []
    save_timer = time.time()
    save_time = 3600      #save once an hour or every 3600 seconds
    
    
    ##########################################
    #
    #List of helper functions that are used to
    #access the data in the manager
    #
    ##########################################
    
    
    def q_get(self):
        return self.q.get()
    
    def q_cancer_get(self):
        return self.q_cancer.get()
    
    def q_laterality_get(self):
        return self.q_laterality.get()
    
    def q_exam_get(self):
        return self.q_exam.get()
    
    def q_subject_id_get(self):
        return self.q_subject_id.get()    
    
    def q_bc_history_get(self):
        return self.q_bc_history.get()
        
    def q_bc_first_degree_history_get(self):
        return self.q_bc_first_degree_history.get()
        
    def q_bc_first_degree_history_50_get(self):
        return self.q_bc_first_degree_history_50.get()
        
    def q_anti_estrogen_get(self):
        return self.q_anti_estrogen.get()
        
    def q_put(self, arg):
        self.q.put(arg)
        
    def q_cancer_put(self, arg):
        self.q_cancer.put(arg)
        
    def q_laterality_put(self, arg):
        self.q_laterality.put(arg)
        
    def q_exam_put(self, arg):
        self.q_exam.put(arg)
        
    def q_subject_id_put(self, arg):
        return self.q_subject_id.put(arg)    
    
    def q_bc_history_put(self, arg):
        self.q_bc_history.put(arg)
        
    def q_bc_first_degree_history_put(self, arg):
        self.q_bc_first_degree_history.put(arg)
        
    def q_bc_first_degree_history_50_put(self, arg):
        self.q_bc_first_degree_history_50.put(arg)
        
    def q_anti_estrogen_put(self, arg):
        self.q_anti_estrogen.put(arg)
        
    def q_empty(self):
        return self.q.empty()
    
    def q_size(self):
        return self.q.qsize()
    
    def t_lock_acquire(self):
        self.t_lock.acquire()
        
    def t_lock_release(self):
        self.t_lock.release()
        
    def set_scan_no(self, scan_no):
        self.scan_no = scan_no
        
    def get_scan_no(self, scan_no):
        return self.scan_no
    
    def error_files_put(self, erf):
        self.error_files.append(erf)
        
    def get_exit_status(self):
        return self.exit_flag
    
    def set_exit_status(self, status):
        self.exit_flag = status
        
    def inc_cancer_count(self):
        self.cancer_count += 1
        
    def inc_benign_count(self):
        self.benign_count += 1
        
    def get_cancer_count(self):
        return self.cancer_count
    
    def get_benign_count(self):
        return self.benign_count
    
    def inc_scan_count(self):
        self.scan_no += 1
        
    def get_scan_count(self):
        return self.cancer_count
    
    def get_error_files(self):
        return self.error_files
    
    def add_error_file(self, f):
        self.error_files.append(f)
        
    def first_features_added(self):
        return (len(self.feature_array) == 0)
       
    def set_feature_array(self, feature_array):
        self.feature_array = feature_array
        
    def set_class_array(self, class_array):
        self.class_array = class_array
        
    def append_feature_array(self, feature_array):
        self.feature_array.append(feature_array)
        
    def append_class_array(self, class_array):
        self.class_array.append(class_array)
        
    def get_feature_array(self):
        return self.feature_array
        
    def get_class_array(self):
        return self.class_array
    
    def get_laterality_array(self):
        return self.laterality_array
    
    def get_exam_array(self):
        return self.exam_array
    
    def get_subject_id_array(self):
        return self.subject_id_array
    
    def get_bc_history_array(self):
        return self.bc_history_array
    
    def get_bc_first_degree_history_array(self):
        return self.bc_first_degree_history_array
    
    def get_bc_first_degree_history_50_array(self):
        return self.bc_first_degree_history_50_array
    
    def get_anti_estrogen_array(self):
        return self.anti_estrogen_array
    
    
    
    
    
    
    
    ###################################
    #
    # Functions to handle timing of saving data
    #
    ###################################
    
    """
    init_timer()
    
    Description:
    Initialise timer
    
    
    """
    def init_timer(self):
        self.save_timer = time.time()
        
        
    """
    reset_timer()
    
    Description: 
    Will just reinitialise both timers
    This function is just here as a convenience
    
    """
    
    def reset_timer(self):
        self.init_timer()
        
        
    def save_time_elapsed(self):
        return ( (time.time() - self.save_timer) >= self.save_time)
    
    
    
    def periodic_save_features(self, save_path):
        self.t_lock_acquire()
        X = np.array(self.get_feature_array())
        Y = np.array(self.get_class_array())
        lateralities = np.array(self.get_lateralities_array(), dtype='S1')
        exams = np.array(self.get_exams(), dtype=np.int)
        subject_ids = np.array(self.get_subject_ids(), dtype='S12')
        
        np.save(save_path + '/model_data/X_temp', X)
        np.save(save_path + '/model_data/Y_temp', Y)
        np.save(command_line_args.save_path + '/model_data/lateralities_temp', lateralities)
        np.save(command_line_args.save_path + '/model_data/exams_temp', exams)
        np.save(command_line_args.save_path + '/model_data/subject_ids_temp', subject_ids)
        
        print('Saved Temporary Data')
        print(X.shape)
        #reset the timer
        self.reset_timer()
        self.t_lock_release()
        
        
        
        
    """    
    add_features()
    
    Description: 
    This function will add all of the features from scans processed
    in a single thread/process and add them to the list in the manager
    
    @params
    X = N x M array containing the features from the scans processed in the thread
        N = number of scans processed
        M = Number of features per scan
    
    Y = 1D array containing status of each of training scans
    laterality = list containing the laterality of each scan (either 'L' or 'R')
    exam = list containing the examination number of each scan
    subject_id = list containing the subject ID of all the scans
    
    """
    
    def add_features(self,X, Y, laterality, exam, subject_id, bc_histories, bc_first_degree_histories, bc_first_degree_histories_50, anti_estrogens):
        #request lock for this process
        self.t_lock_acquire()
        #if this is the first set of features to be added, we should just initially
        #set the features in the manager to equal X and Y.
        #Otherwise, use extend on the list. Just a nicer way of doing it so we
        #don't have to keep track of indicies and sizes
        if(self.first_features_added()):
            self.feature_array = X
            self.class_array = Y
            self.laterality_array = laterality
            self.exam_array = exam
            self.subject_id_array = subject_id
            self.bc_history_array = bc_histories
            self.bc_first_degree_history_array = bc_first_degree_histories
            self.bc_first_degree_history_50_array = bc_first_degree_histories_50
            self.anti_estrogen_array = anti_estrogens
            
        else:
            print(np.shape(self.feature_array))
            self.feature_array.extend(X)
            self.class_array.extend(Y)
            self.laterality_array.extend(laterality)
            self.exam_array.extend(exam)
            self.subject_id_array.extend(subject_id)
            self.bc_history_array.extend(bc_histories)
            self.bc_first_degree_history_array.extend(bc_first_degree_histories)
            self.bc_first_degree_history_50_array.extend(bc_first_degree_histories_50)
            self.anti_estrogen_array.extend(anti_estrogens)
            
        #we are done so release the lock
        self.t_lock_release()
        
        
class my_manager(BaseManager):
    pass

my_manager.register('shared', shared)





class my_thread(Process):    
    
    """
    __init__()
    
    Description:
    Will initialise the thread and the feature member variable.
    
    @param thread_id = integer to identify individual thread s
    @param num_images_total = the number of images about to be processed in TOTAL
    @param command_line_args = arguments object that holds the arguments parsed from 
           the command line. These areguments decide whether we are training, preprocessing etc.
    
    """
    
    def __init__(self, thread_id, no_images_total, manager, command_line_args):
        Process.__init__(self)
        print('Initialising thread %d' %thread_id)
        self.manager = manager
        self.t_id = thread_id
        self.scan_data = feature(levels = 3, wavelet_type = 'haar', no_images = no_images_total ) #the object that will contain all of the data
        self.scan_data.current_image_no = 0    #initialise to the zeroth mammogram
        self.data_path = command_line_args.input_path
        self.time_process = []   #a list containing the time required to perform preprocessing
                                 #on each scan, will use this to find average of all times
        self.scans_processed = 0
        self.cancer_status = []        
        self.subject_ids = []
        self.lateralities = []
        self.bc_histories = []
        self.bc_first_degree_histories = []
        self.bc_first_degree_histories_50 = []
        self.anti_estrogens = []
        self.exam_nos = []
        self.training = command_line_args.training
        self.preprocessing = command_line_args.preprocessing
        self.validation = command_line_args.validation
        self.save_path = command_line_args.save_path
        self.challenge_submission = command_line_args.challenge_submission
        #some variables that are used to time the system for adding files to the shared manager
        self.add_timer = time.time()
        #add features every 13 minutes
        #made it not a factor of 60 minutes
        self.add_time = 780
        
        
        
        
        
        
    """
    run()
    
    Description:
    Overloaded version of the Thread modules run function, which is called whenever the thread
    is started. This will essentially just call the processing member function , which handles
    the sequence for the running of the thread
    """
    
    def run(self):
        #just run the process function
        print('Begin process in thread %d' %self.t_id)
        self.process()
        
        
        
    """
    process()
    
    Description:
    The bulk of the running of the thread is handeled in this function. This function handles
    the sequence of processing required, ie. loaading scans, preprocessing etc.
    
    Also handles synchronization of the threads, by locking the threads when accessing the 
    class reference variables, such as the queues
    
    The process function will continue to loop until the queues are empty. When the queues
    are empty, the reference variable (exit_flag) will be set to True to tell us that the processing
    is about ready to finish.
    
    If there is an error with any file, the function will add the name of the file where the error
    occurred so it can be looked at later on, to see what exactly caused the error. 
    This will raise a generic exception, and will just mean we skip to the next file if we 
    encounter an error.
    
    Will also time each of the preprocessing steps to see how long the preprocessing of each
    scan actually takes, just to give us an idea
    
    Finally, will make sure we will look at parts of the feature array that are actually valid
    
    """
    
    
    def process(self):
        print('Running thread %d' %self.t_id)
        #while there is still names on the list, continue to loop through
        #while the queue is not empty
        while( not self.manager.get_exit_status()):
            #variable that is used to see if we want to process this scan
            #or skip it
            valid = True
            #lock the threads so only one is accessing the queue at a time
            #start the preprocessing timer
            start_time = timeit.default_timer()
            self.manager.t_lock_acquire()
            if(not (self.manager.q_empty()) ):
                file_path = self.data_path + self.manager.q_get()
                #get the cancer status.
                #if we are validating on the synapse servers, the cancer status is not
                #given, so I have set them all to false in read_files
                self.cancer_status.append(self.manager.q_cancer_get())
                #add the rest of the metadata to the list
                self.subject_ids.append(self.manager.q_subject_id_get())
                self.lateralities.append(self.manager.q_laterality_get())
                self.exam_nos.append(self.manager.q_exam_get())
                self.bc_histories.append(self.manager.q_bc_history_get())
                self.bc_first_degree_histories.append(self.manager.q_bc_first_degree_history_get())
                self.bc_first_degree_histories_50.append(self.manager.q_bc_first_degree_history_50_get())
                self.anti_estrogens.append(self.manager.q_anti_estrogen_get())
                
                #if this scan is cancerous
                if(self.cancer_status[-1]):
                    self.manager.inc_cancer_count()
                    print('cancer count = %d' %self.manager.get_cancer_count())
                    
                #otherwise it must be a benign scan
                else:
                    self.manager.inc_benign_count()
                    print('benign count = %d' %self.manager.get_benign_count())
                    
                    
                #if we have enough benign scans, lets not worry about it this scan
                #as we have enough for training purposes
                #if we are validating, the benign count will most likely be more,
                #but we want to keep going to try and classify benign scans
                #if it is a cancerous file though we should keep going
                if(self.manager.get_benign_count() > 30000) & (not self.validation) & (not self.cancer_status[-1]):
                    self.remove_most_recent_metadata_entries()
                    print('Skipping %s since we have enough benign scans :)' %(file_path))
                    #now we can just continue with this loop and go on about our business
                    #this will go to next iteration of the while loop
                    valid = False
                    
                    
                #if the queue is now empty, we should wrap up and
                #get ready to exit
                if(self.manager.q_empty()):
                    print(' Queue is Empty')
                    self.manager.set_exit_status(True)                
                    
                    
                #this is just here whilst debugging to shorten the script
                #if(self.manager.get_cancer_count() > 2):
                #    print('Cancer Count is greater than 2 so lets exit')
                #    self.manager.set_exit_status(True)                
                    
                #are done getting metadata and filenames from the queue, so unlock processes
                self.manager.t_lock_release()
                print('Queue size = %d' %self.manager.q_size())
                
                #if we have made it here, and it isn't a scan we wish to skip
                #we should get busy processing some datas :)
                if(valid):
                    try:
                        #now we have the right filename, lets do the processing of it
                        #if we are doing the preprocessing, we will need to read the file in correctly
                        self.scan_data.initialise(file_path, self.preprocessing)
                        
                        #begin preprocessing steps
                        if(self.preprocessing):
                            self.scan_data.preprocessing()
                            self.save_preprocessed()
                            
                        self.scan_data.get_features()
                        
                        #now that we have the features, we want to append them to the list of features
                        #The list of features is shared amongst all of the class instances, so
                        #before we add anything to there, we should lock other threads from adding
                        #to it
                        self.scan_data.current_image_no += 1   #increment the image index
                        lock_time = timeit.default_timer()
                        #now increment the total scan count as well
                        self.inc_total_scan_count()
                            
                            
                    except Exception as e:
                        print e 
                        print('Error with current file %s' %(file_path))
                        self.manager.add_error_file(file_path)
                        #get rid of the last cancer_status flag we saved, as it is no longer
                        #valid since we didn't save the features from the scan
                        #
                        #NOTE: we won't want to get rid of these features whilst validating
                        #on the synapse server. Just find features from what we have and run with it
                        if(not self.validation) & (not self.challenge_submission):
                            print('removing this scan')
                            self.remove_most_recent_metadata_entries()
                        #if we are validating for the challenge, lets try our best on what we have
                        
                        else:
                            
                            #should be safe to have this outside a try and catch, as it
                            #should never fail
                            self.scan_data.get_features()
                            self.scan_data.current_image_no += 1   #increment the image index
                            self.inc_total_scan_count()
                            
                            
                            
                sys.stdout.flush()
                self.scan_data.cleanup()
                gc.collect()
                
            else:
                #we should just release the lock on the processes
                self.manager.t_lock_release()                    
                
                
        #there is nothing left on the queue, so we are ready to exit
        #before we exit though, we just need to crop the feature arrays to include
        #only the number of images we looked at in this individual thread
        self.add_features()
        
        #now just print a message to say that we are done
        print('Thread %d is out' %self.t_id)
        sys.stdout.flush()
        
        
        
    """
    remove_most_recent_metadata_entries():
    
    Description:
    Function will remove the data entries from the last scan if there was
    and error, or if there was too many scans collected etc.
    """ 
        
    def remove_most_recent_metadata_entries(self):
        
        del self.cancer_status[-1]
        del self.lateralities[-1]
        del self.exam_nos[-1]
        del self.subject_ids[-1]        
        
        
        
        
    def add_time_elapsed(self):
        return (time.time() - self.add_timer) > self.add_time
    
    
    
    
    def flatten(self, x):
        "Flatten one level of nesting"
        return chain.from_iterable(x)
    
    
    
    """
    save_preprocessed()
    
    Description:
    After a file has been preprocessed, will write it to file so we don't have to do 
    it again. 
    Before saving to file, we will convert all of the Nans to -1. This allows us to save the data
    as a 16-bit integer instead of 32-64 bit float.
    
    Using half the amount of disk space == Awesomeness
    
    Will just have to convert it when loading before performing feature extraction
    
    """
    
    def save_preprocessed(self):
        
        file_path = self.save_path +  self.scan_data.file_path[-10:-4]
        print(file_path)
        #copy the scan data
        temp = np.copy(self.scan_data.data)
        #set all Nan's to -1
        temp[np.isnan(temp)] = -1
        temp = temp.astype(np.int16)
        
        #now save the data
        np.save(file_path, temp)
        
        
    """
    inc_scan_count()
    
    Description:
    Will just increment the scans processed
    
    """
    def inc_total_scan_count(self):
        self.manager.t_lock_acquire()
        self.manager.inc_scan_count()
        self.manager.t_lock_release()                    
        
        
        
        
        
        
    """
    add_features()
    
    Description:
    Am done with selection and aquisition of features. 
    Will add the features from this thread to the manager
    
    """
    
    
    
    def add_features(self):
        
        self.scan_data._crop_features(self.scan_data.current_image_no)
        
        X = []
        Y = []
        for t_ii in range(0, self.scan_data.current_image_no):
            X.append([])
            for lvl in range(self.scan_data.levels):                
                homogeneity = self.scan_data.homogeneity[t_ii][lvl]
                entropy = self.scan_data.entropy[t_ii][lvl]
                energy = self.scan_data.energy[t_ii][lvl]
                contrast = self.scan_data.contrast[t_ii][lvl]
                dissimilarity = self.scan_data.dissimilarity[t_ii][lvl]
                correlation = self.scan_data.correlation[t_ii][lvl]
                
                #wave_energy = self.scan_data.wave_energy[t_ii][lvl]
                #wave_entropy = self.scan_data.wave_entropy[t_ii][lvl]
                #wave_kurtosis = self.scan_data.wave_kurtosis[t_ii][lvl]
                
                X[t_ii].extend(self.flatten(homogeneity))
                X[t_ii].extend(self.flatten(entropy))
                X[t_ii].extend(self.flatten(energy))
                X[t_ii].extend(self.flatten(contrast))
                X[t_ii].extend(self.flatten(dissimilarity))
                X[t_ii].extend(self.flatten(correlation))
                #X[t_ii].extend(flatten(wave_energy))
                #X[t_ii].extend(flatten(wave_entropy))
                #X[t_ii].extend(flatten(wave_kurtosis))
                
                #X[ii, lvl + lvl*jj*no_features] = homogeneity[0,jj]
                #X[ii, jj*no_features + 1 + kk] = entropy[0,jj]
                #X[ii, jj*no_features + 2*feat_len] = energy[0,jj]
                #X[ii, jj*no_features + 3*feat_len] = contrast[0,jj]
                #X[ii, jj*no_features + 4*feat_len] = dissimilarity[0,jj]
                #X[ii, jj*no_features + 5*feat_len] = correlation[0,jj]
                
            
            #add the density measure for this scan
            X[t_ii].append(self.scan_data.density[t_ii])
            #set the cancer statues for this scan            
            Y.append(self.cancer_status[t_ii])    
        #now add the features from this list to to conplete list in the manager
        self.manager.add_features(X, Y, self.lateralities, self.exam_nos, self.subject_ids, self.bc_histories, self.bc_first_degree_histories, self.bc_first_degree_histories_50, self.anti_estrogens)
        #reinitialise the feature list in the scan_data member
        self.scan_data.reinitialise_feature_lists()
        #reinitialise the metadata
        self.cancer_status = []        
        self.subject_ids = []
        self.lateralities = []
        self.exam_nos = []
        
        #set the number of scans processed back to zero
        self.scan_data.current_image_no = 0
        #reset the timer
        self.add_timer = time.time()
        
        
        
        
