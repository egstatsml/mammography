import dicom
import os
import numpy as np
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
import pywt

from scipy.cluster.hierarchy import  linkage as linkage



#load in a file
#file_path = '/home/ethan/DREAM/pilot_images/111359.dcm' #image without pectoral muscle
#file_path = '/home/ethan/DREAM/pilot_images/134060.dcm' #image wwith pectoral muscle
file_path = '/home/ethan/DREAM/pilot_images/502860.dcm' #malignant case
current_scan = breast(file_path)
current_scan.remove_artifacts()

boundary = current_scan.breast_boundary()




plt.figure()
titles = ['Horizontal', 'Vertical', 'Diagonal']

decomp = pywt.wavedec2(np.log(1 + current_scan.data), 'haar', mode='symmetric')
wav_i = []

for ii in range(6, len(decomp)):
    filter_bank = decomp[ii]
    print(np.shape(filter_bank))
    for jj in range(0, 3):
        print(jj)
        filter_temp = filter_bank[jj]
        cluster = linkage(filter_bank)
        plt.imshow(cluster )
        plt.title(titles[jj])
        plt.colorbar()
        plt.show()
        plt.clf()
            
    


    

