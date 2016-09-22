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

from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi
#import the breast class
from breast import breast
from feature_extract import feature
from read_files import spreadsheet
import pywt
from sklearn.cluster import KMeans
from skimage import measure

from scipy.cluster import vq as vq
import skfuzzy as fuzz



descriptor = spreadsheet()

#creating a list that will store the scan filename if any of them create an error
#will then save this list and have a look to see where they failed
error_files = []

while(descriptor.patient_pos < descriptor.no_patients):
    #load in a file
    #file_path = '/home/ethan/DREAM/pilot_images/111359.dcm' #image without pectoral muscle
    #file_path = '/home/ethan/DREAM/pilot_images/134060.dcm' #image with pectoral muscle
    #file_path = '/home/ethan/DREAM/pilot_images/502860.dcm' #malignant case
    #file_path = '/home/ethan/DREAM/pilot_images/151849.dcm' #malignant case with pectoral muscle
    #file_path = '/home/ethan/DREAM/pilot_images/485343.dcm' #scan that broke my initial boundary scan
    
    
    file_path = descriptor.next_scan()

    print(file_path)

    try:
        current_scan = feature(file_path)
        current_scan.get_features()
    except:
        error_files.append(file_path)




#will be here after have run through everything
#lets save the features we found, and the file ID's of any that
#created any errors so I can have a look later
#will just convert the list of error files to a pandas database to look at later
error_database = pd.DataFrame(error_files)

    

