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
from metric import *
from kl_divergence import *
from metric import *





if __name__ == "__main__":
    
    
    #lets load in the spreadsheet describing all of our scans
    args = arguments(sys.argv[1:])    
    
    for snapshot in xrange(1000,20000,1000):
        
        
        args.rcnn = './res/%d' %snapshot
        #load in the results from this snapshot
        descriptor = spreadsheet(args)
        
        tp = 0.0;
        fp = 0.0;
        tn = 0.0;
        fn = 0.0;
        
        for ii in range(len(descriptor.filenames)):
            
            #check if it is a true positive
            if(descriptor.cancer_list[ii]) & (descriptor.regions != [0]):
                tp += 1.0
            #false positive
            elif(not descriptor.cancer_list[ii]) & (descriptor.regions != [0]):
                fp += 1.0
            #true negative
            elif(not descriptor.cancer_list[ii]) & (descriptor.regions == [0]):
                tn += 1.0
            #otherwise is a false negative :(
            else:
                fn += 1.0
                
                
        accuracy = (tp + tn)/(tp +tn +fp +fn)
        sensitivity = tp/(tp + fn)
        specificity = tn/(tn + fp)
        #now lets display the results
        print('~~~~~~~~~ Results from Snapshot %d ~~~~~~~~~' %snapshot)
        print("TP: %d" %tp)
        print("FP: %d" %fp)        
        print("TN: %d" %tn)
        print("FN: %d" %fn)
        print("Accuracy: {0:15}".format(accuracy))
        print("Sensitivity: {0:15}".format(sensitivity))
        print("Specificity: {0:15}".format(specificity))
        print('\n\n')
        
        
        
        
