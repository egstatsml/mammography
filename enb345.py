"""
main.py

Author: Ethan Goan
Queensland University of Technology
DREAM Mammography Challenge
2016

Description:

source file to run program for training SVM classifier for 
breast cancer detection. Uses breast, spreadsheet and feature classes
defined in read_files.py, breast.py and feature_extract.py, which contain
main functionality for reading in the files



"""


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
import pywt
from skimage import measure
from sklearn import svm
from sklearn.externals import joblib
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi

import pyqtgraph as pg
from PyQt4 import QtGui 

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet
from gui import view_scan, window



#initialising everything            
## Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])
w = window()
descriptor = spreadsheet(benign_files=None)
#creating a list that will store the scan filename if any of them create an error
#will then save this list and have a look to see where they failed
error_files = []
scan = feature(levels = 3, wavelet_type = 'haar', no_images = descriptor.benign_count + descriptor.malignant_count)

w.show()
app.exec_()
        

