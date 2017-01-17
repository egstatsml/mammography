#!/usr/bin/python
import threading
import timeit
import Queue
import gc

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

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet


#for tracking memory
from pympler.tracker import SummaryTracker
from pympler.asizeof import asizeof



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


class my_thread(threading.Thread):    
    
    #some reference variables that will be shared by all of the individual thread objects
    #whenever any of these reference variables are accessed, the thread lock should be applied,
    #to ensure multiple threads arent accessing the same memory at the same time
    
    q = Queue.Queue(1000)        #queue that will contain the filenames of the scans
    q_cancer = Queue.Queue(1000) #queue that will contain the cancer status of the scans
    t_lock = threading.Lock()
    exit_flag = False
    error_files = []   #list that will have all of the files that we failed to process
    scan_no = 0
    cancer_count = 0
    
    """
    __init__()
    
    Description:
    Will initialise the thread and the feature member variable.
    
    @param thread_id = integer to identify individual thread s
    @param num_images_total = the number of images about to be processed in TOTAL
    @param data_path = the path to the scans, will vary depending on which machine we are on
    @param training = boolean to say if we are training or if we are doing the validation process
    
    """
    
    def __init__(self, thread_id, no_images_total, data_path = './pilot_images/', training = True):
        threading.Thread.__init__(self)
        
        self.t_id = thread_id
        self.scan_data = feature(levels = 3, wavelet_type = 'haar', no_images = no_images_total ) #the object that will contain all of the data
        self.scan_data.current_image_no = 0    #initialise to the zeroth mammogram
        self.data_path = data_path
        self.time_process = []   #a list containing the time required to perform preprocessing
                                 #on each scan, will use this to find average of all times
        self.scans_processed = 0
        self.cancer_status = []
        self.training = training
        
    """
    run()
    
    Description:
    Overloaded version of the Thread modules run function, which is called whenever the thread
    is started. This will essentially just call the processing member function , which handles
    the sequence for the running of the thread
    """
    
    def run(self):
        #just run the process function
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
        
        #while there is still names on the list, continue to loop through
        #while the queue is not empty        
        while not my_thread.exit_flag:
            #lock the threads so only one is accessing the queue at a time
            #start the preprocessing timer
            start_time = timeit.default_timer()
            my_thread.t_lock.acquire()
            if(not (self.q.empty()) ):
                file_path = self.data_path + my_thread.q.get()
                self.cancer_status.append(my_thread.q_cancer.get())
                if(self.cancer_status[-1]):
                    my_thread.cancer_count += 1
                    print('cancer count = %d' %my_thread.cancer_count)
                        
                #file_path = 'pilot_images/502860.dcm'
                my_thread.t_lock.release()

                #try:
                #now we have the right filename, lets do the processing of it
                self.scan_data.initialise(file_path)
                self.scan_data.preprocessing()
                self.scan_data.get_features()
                
                
                #now that we have the features, we want to append them to the list of features
                #The list of features is shared amongst all of the class instances, so
                #before we add anything to there, we should lock other threads from adding
                #to it
                self.time_process.append(timeit.default_timer() - start_time)
                #print('time = %d s' %self.time_process[-1])
                self.scan_data.current_image_no += 1   #increment the image index
                self.scans_processed += 1              #increment the image counter
                lock_time = timeit.default_timer()
                my_thread.t_lock.acquire()
                my_thread.scan_no += 1
                print(my_thread.q.qsize())
                my_thread.t_lock.release()                    
                print(timeit.default_timer() - lock_time)                
                
                if(self.training & (my_thread.cancer_count > 18)):
                    my_thread.exit_flag = True
                    
                #if we aren't training, but have run out of scans for validation
                #we should also exit
                elif(my_thread.q.empty()):
                    my_thread.exit_flag = True
                    
                    #except:
                #    print('Error with current file %s' %(file_path))
                #    my_thread.error_files.append(file_path)
                #    #get rid of the last cancer_status flag we saved, as it is no longer valid since we didn't save the
                #    #features from the scan
                #    del self.cancer_status[-1]
                    
                self.scan_data.cleanup()
                gc.collect()
            else:
                my_thread.t_lock.release()
                
                
        #there is nothing left on the queue, so we are ready to exit
        #before we exit though, we just need to crop the feature arrays to include
        #only the number of images we looked at in this individual thread
        self.scan_data._crop_features(self.scans_processed)
        print('Thread %d is out' %self.t_id)





    
    
    """
    reinitialise()
    
    Description:
    Reinitialise the reference variables
    Need to do when we are moving from training to classifying
    
    """
    
    def reinitialise_ref(my_thread):
        
        my_thread.exit_flag = False
        my_thread.error_files = []   #list that will have all of the files that we failed to process
        my_thread.cancer_status = []
        my_thread.scan_no = 0
        my_thread.cancer_count = 0
        
    
    
    
