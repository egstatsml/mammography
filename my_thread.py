#!/usr/bin/python
import threading
import timeit
import Queue
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


class shared(object):
    #some reference variables that will be shared by all of the individual thread objects
    #whenever any of these reference variables are accessed, the thread lock should be applied,
    #to ensure multiple threads arent accessing the same memory at the same time
    
    q = Queue(1000)        #queue that will contain the filenames of the scans
    q_cancer = Queue(1000) #queue that will contain the cancer status of the scans
    t_lock = Lock()
    exit_flag = False
    error_files = []   #list that will have all of the files that we failed to process
    scan_no = 0
    cancer_count = 0
    
    def q_get(self):
        return self.q.get()
    
    def q_cancer_get(self):
        return self.q_cancer.get()
    
    def q_put(self, arg):
        self.q.put(arg)
        
    def q_cancer_put(self, arg):
        self.q_cancer.put(arg)
        
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
        
    def get_scan_count(self):
        return self.scan_no
    
    def error_files_put(self, erf):
        self.error_files.append(erf)
        
    def get_exit_status(self):
        return self.exit_flag
    
    def set_exit_status(self, status):
        self.exit_flag = status
        
    def inc_cancer_count(self):
        self.cancer_count += 1
        
    def set_cancer_count(self, cancer_count):
        self.cancer_count = cancer_count
        
        
    def get_cancer_count(self):
        return self.cancer_count
    
    def inc_scan_count(self):
        self.scan_no += 1
        
    def get_error_files(self):
        return self.error_files
    
    def add_error_file(self, f):
        self.error_files.append(f)
    
class my_manager(BaseManager):
    pass

my_manager.register('shared', shared)



class feature_manager(BaseManager):
    pass





class feature_data(object):
    
    def __init__(self):
        
        #number of scans this process went through
        self.scans_processed = 0
        self.cancer_status = []
        
        
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
        
        
    def set_scans_processed(self, no_scans):
        self.scans_processed = no_scans        
        
    def set_cancer_status(self, cancer_status):
        self.cancer_status = cancer_status        
        
        
    def set_homogeneity(self, homogeneity):
        self.homogeneity = homogeneity

    def set_entropy(self, entropy):
        self.entropy = entropy
        
    def set_energy(self, energy):
        self.energy = energy
        
    def set_contrast(self, contrast):
        self.contrast = contrast
        
    def set_dissimilarity(self, dissimilarity):
        self.dissimilarity = dissimilarity
        
    def set_correlation(self, correlation):
        self.correlation = correlation
        
    def set_density(self, density):
        self.density = density            
        
        
    def set_wave_entropy(self, entropy):
        self.wave_entropy = entropy
        
    def set_wave_energy(self, energy):
        self.energy = energy
        
        
        
    def get_scans_processed(self):
        return self.scans_processed
    
    def get_cancer_status_individual(self, index):
        return self.cancer_status[index]
    
    def get_cancer_status(self):
        return self.cancer_status
    
    
    def get_homogeneity(self):
        return self.homogeneity
    
    def get_entropy(self):
        return self.entropy
    
    def get_energy(self):
        return self.energy
    
    def get_contrast(self):
        return self.contrast
    
    def get_dissimilarity(self):
        return self.dissimilarity

    def get_correlation(self):
        return self.correlation

    def get_density(self):
        return self.density
    
    def get_wave_entropy(self):
        return self.wave_entropy
        
    def get_wave_energy(self):
        return self.wave_energy
        
        
feature_manager.register('feature_data', feature_data)        
        
        
        
        
        
        
        
class my_thread(Process):    
    
    #some reference variables that will be shared by all of the individual thread objects
    #whenever any of these reference variables are accessed, the thread lock should be applied,
    #to ensure multiple threads arent accessing the same memory at the same time
    """
    q = Queue(1000)        #queue that will contain the filenames of the scans
    q_cancer = Queue(1000) #queue that will contain the cancer status of the scans
    t_lock = Lock()
    exit_flag = False
    error_files = []   #list that will have all of the files that we failed to process
    scan_no = 0
    cancer_count = 0
    """
    """
    __init__()
    
    Description:
    Will initialise the thread and the feature member variable.
    
    @param thread_id = integer to identify individual thread s
    @param num_images_total = the number of images about to be processed in TOTAL
    @param data_path = the path to the scans, will vary depending on which machine we are on
    @param training = boolean to say if we are training or if we are doing the validation process
    
    """
    
    def __init__(self, thread_id, no_images_total, manager, feature_manager, data_path = './pilot_images/', training = True):
        Process.__init__(self)
        print thread_id
        self.manager = manager
        self.feature_manager = feature_manager
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
        while( not self.manager.get_exit_status()):
            print 'in'
            #lock the threads so only one is accessing the queue at a time
            #start the preprocessing timer
            start_time = timeit.default_timer()
            self.manager.t_lock_acquire()
            if(not (self.manager.q_empty()) ):
                file_path = self.data_path + self.manager.q_get()
                self.cancer_status.append(self.manager.q_cancer_get())
                if(self.cancer_status[-1]):
                    self.manager.inc_cancer_count()
                    print('cancer count = %d' %self.manager.get_cancer_count())
                    
                #if the queue is now empty, we should wrap up and
                #get ready to exit
                if(self.manager.q_empty()):
                    print 'here'
                    self.manager.set_exit_status(True)
                    
                if(self.manager.get_cancer_count() > 10):
                    self.manager.set_exit_status(True)
                    
                    
                #file_path = 'pilot_images/502860.dcm'
                self.manager.t_lock_release()
                #print('Queue size = %d' %self.manager.q_size())

                try:
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
                    self.manager.t_lock_acquire()
                    self.manager.inc_scan_count()
                    self.manager.t_lock_release()                    
                    print self.manager.q_size()


                    #get ready to exit
                    #if we aren't training, but have run out of scans for validation
                    #we should also exit

                except:
                    print('Error with current file %s' %(file_path))
                    self.manager.add_error_file(file_path)
                    #get rid of the last cancer_status flag we saved, as it is no longer valid since we didn't save the
                    #features from the scan
                    del self.cancer_status[-1]

                self.scan_data.cleanup()
                gc.collect()
            else:
                self.manager.t_lock_release()                    
                
                
        #there is nothing left on the queue, so we are ready to exit
        #before we exit though, we just need to crop the feature arrays to include
        #only the number of images we looked at in this individual thread
        self.scan_data._crop_features(self.scans_processed)
        #now will copy the features we found into the memory manager (feature_data)
        #so they can be used by the main process
        self._copy_features_manager()
        
        print('Thread %d is out' %self.t_id)
        
        
        
    def _copy_features_manager(self):
        self.feature_manager.set_scans_processed(self.scans_processed)
        self.feature_manager.set_cancer_status(self.cancer_status)
        self.feature_manager.set_correlation((self.scan_data.correlation))
        self.feature_manager.set_homogeneity((self.scan_data.homogeneity))
        self.feature_manager.set_energy((self.scan_data.energy))
        self.feature_manager.set_entropy((self.scan_data.entropy))
        self.feature_manager.set_contrast((self.scan_data.contrast))
        self.feature_manager.set_dissimilarity((self.scan_data.dissimilarity))
        
        
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
        
    
    
    
