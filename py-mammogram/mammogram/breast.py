#!/bin/env python
"""
breast.py

Author: Ethan Goan

Queensland University of Technology
DREAM Mammography Challenge
2016

Description:

Breast object is defined for preprocessing of mammograms
Class contains methods for removing artifacts from the scan such as labels and
any other marks that may be included. Unwanted tissues such as pectoral muscle and
excess skin are removed so the image data that will be used for the feature extraction and
training purposes will contain only relevant breast tissue


"""







import dicom
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
from scipy import signal
from scipy.ndimage import filters
from scipy.ndimage import measurements as measurements
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import gc
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi

import pywt
######
#for when I port this to Cython
from mammogramCython.breast_cython import cy_search_prev
#from boundary import trace_boundary, edge_boundary




"""
Defining the breast class

Will contain all information regarding the mammogram scan of the breast
and some functions to process the data

"""


class breast(object):
    def __init__(self, file_path = None):
        
        
        self.data = []                           #the mammogram
        self.original_scan = []
        self.pectoral_mask = []                  #binary map of pectoral muscle
        self.breast_mask = []                    #binary map of breast
        self.fibroglandular_mask = []            #binary map of fibroglandular tissue
        self.microcalcs_mask = []       #binary map of microcalcifications
        self.pectoral_present = False
        self.pectoral_muscle_removed = False
        self.label_removed = False
        self.width = 0
        self.height = 0
        self.im_width = 0
        self.im_height = 0
        self.area = 0
        self.boundary = []
        self.boundary_y= []
        self.x_pec = []
        self.y_pec = []
        self.nipple_x = 0
        self.nipple_y = 0
        self.threshold = 0                #threshold for segmenting fibroglandular disk
        self.file_path = []
        self.plot = False                  #boolean for plotting figures when debugging
        self.file_path = file_path
        if(file_path != None):
            self.initialise(file_path)
            
            
            
            
    """
    breast.initialise()
    
    Desccription:
    
    Will take a file path and load the data and store it in a numpy array
    Also set the width and height of the image, and check the orientation of the breast
    in the scan is correct (ie. breast is on the left)
    
    @param file_path = string containing the location of the file you want to read in
    @param preprocessing = boolean to say if we are preprocessing the input data
                           If we are, we will be reading in an image in dicom format
                           if not will be a numpy array
    
    """
    def initialise(self, file_path, preprocessing = True):
        self.file_path = file_path
        #if we are preprocessing, read in the DICOM formatted image
        if(preprocessing):
            print file_path
            file = dicom.read_file(file_path)
            self.data = np.fromstring(file.PixelData,dtype=np.int16).reshape((file.Rows,file.Columns))
            self.original_scan = np.copy(self.data)
            #convert the data to floating point now
            self.data = self.data.astype(float)
            self.check_orientation()
        else:
            #we have already done the preprocessing, so can just load in the numpy array
            self.data = np.load(self.file_path).astype(float)
            #when saving the data, we set all Nan's to -1.
            #now we are going to use the preprocessed data, lets set them back Nan
            self.data[self.data == -1] = np.nan
            
            
            
            
    def cleanup(self):
        self.data = []                           #the mammogram
        self.original_scan = []
        self.pectoral_mask = []                  #binary map of pectoral muscle
        self.breast_mask = []                    #binary map of breast
        self.fibroglandular_mask = []            #binary map of fibroglandular tissue
        self.microcalcs_mask = []                #binary map of fibroglandular tissue        
        self.boundary = []
        self.boundary_y= []
        
        
    """
    preprocessing()
    
    Description:
    
    Wrapper function to call methods to apply preprocessing to data
    such as removing artifacts, aligning images and finding the brest boundaries
    
    This function should only be called after the breast.initialise() function has
    been called
    
    
    """
    
    
    def preprocessing(self):
        self.im_width = np.shape(self.data)[1]
        self.im_height = np.shape(self.data)[0]
        #check correct orientation of the image
        self.remove_label()
        self.remove_artifacts()
        self.breast_boundary()
        
        
        
    """
    check_orientation()
    
    Description:
    
    Will check if the image is the correct orientation,
    Eg. breast is on the left hand side of the image.
    
    If it isn't, just fliplr
    
    @param im = input image
    @retval image of correct orientation
    """
    
    
    def check_orientation(self):
        
        #finding the sum of pixel intensities on the left column
        left_edge = np.sum(self.data[:,0])
        #finding the sum of pixel intensities on the right column
        right_edge = np.sum(self.data[:,-1])
        
        if( right_edge > left_edge):
            self.data = np.fliplr(self.data)
            self.original_scan = np.fliplr(self.original_scan)
            
            
            
            
    """
    remove_artifacts()
    
    Description:
    
    Wrapper function that will call other methods to remove artifacts such as labels,
    pectoral muscles
    
    
    """
    
    
    def remove_artifacts(self):
        
        self.remove_label()
        #if there is pectoral muscle here, lets get rid of it
        if(self.pectoral_muscle_present()):
            self.remove_pectoral_muscle()
            
            
            
            
            
            
            
            
            
            
    """    
    breast.remove_label()
    
    Description:
    
    will use local Probability Density Function (pdf)  and Cumulative Density Function (CDF) to find the label and remove it
    search 50 by 50 area and find local pdf
    if local pdf has very few elements below a threshold and more elements above a threshold, then is probably the label 
    will adaptaviely set threshold based on image parameters
    """
    
    
    def remove_label(self):
        
        pdf_axis = np.arange(0, np.max(self.data))
        #first find the threshold value, which will be 95% of maximum value
        threshold_pos = np.round(np.max(self.data) * 0.8)
        threshold_val = 0.5
        threshold_val_high = 0.99
        #search from right to left, as that is where the label will be
        done = False
        #start at 200, the label is never going to be that far to the left
        x = 200
        while (x < (self.im_width - 200) ):
            y = 0
            while (y < (self.im_height- 200) ):
                
                #find pdf and cdf of local region
                pdf = np.histogram( self.data[y:y+200, x:x+200], bins=np.arange(0,np.max(self.data)+1), density=True)
                cdf = np.cumsum(pdf[0])
                #see if most of the pixels are background, and if there has been no change over mid part of cdf
                if( (cdf[0] > threshold_val) &( np.abs(cdf[1000] - cdf[100]) < 0.001 )):
                    
                    #looks like we may have found the label, but to be sure we will move to the left a little bit to see if the count increases.
                    #if it does, then wear really near the breast. If it doesn't, then we really have found the breast
                    #To check if the intensity becomes greater, we will compare median values
                    
                    #finding the median value of PDF in current location
                    med_current = np.median(pdf[0])
                    #now finding median value of space to left, doing it all in one line, same procedure as done above so I will allow this part
                    #of my code to be more terse :)
                    med_left = np.median( np.cumsum( np.histogram(self.data[y:y+200, x-10:x+190], bins=np.arange(0,np.max(self.data)+1), density=True)[0]))
                    #now lets compare
                    if(med_current < med_left):
                        """
                        if((not done) & (np.sum(self.data[y:y+200,x:x+200]) > 500)):
                            done = True
                            #plot the CDF of this scan
                            fig = plt.figure()
                            ax2 = fig.add_subplot(1,1,1)
                            ax2.plot(np.linspace(0,np.max(self.data), cdf.size), cdf)
                            plt.title('CDF of Region With Label')
                            plt.xlabel('Pixel Intensity')
                            plt.ylabel('CDF')
                            plt.ylim([0, 1.2])
                            fig.savefig(os.getcwd() + '/figs/lab_' + self.file_path[-10:-4] + '.eps', bbox_inches='tight')
                            
                        """
                        self.data[y:y+200,x:x+200] = 0                    
                    
                y = y+200
            x = x+200
            
                
                
                
    """
    pectoral_muscle_present()
    
    Description:
    Will see if there is pectoral muscle captured in the scan.
    Most of the time there isnt, but if there is we need to get rid of it.
    
    If the pixel intensities near the top left corner are significantly higher
    than the average intensity, will say that there is pectoral muscle present
    
    @retval boolean to say if there is or isnt pectoral muscle present
    
    """
    
    def pectoral_muscle_present(self):
        
        #find the mean value for just the pixels that arent background
        mean_val = 1.2*np.mean(self.data[self.data > 0])
        #now will search the top left corner
        count = 0
        for y in range(0, 200):
            for x in range(0, 50):
                if(self.data[y,x] > mean_val):
                    count = count+1
                    
        #if the majority of these pixels read greater than the average, then
        #pectoral muscle is said to be present and we should find it
        if(count >= (200.0*50.0*0.60)):
            self.pectoral_present = True
            return True
        else:
            self.pectoral_present = False
            return False
        
        
        
        
        
    """
    remove_pectoral_muscle()
    
    Description: 
    Will check to see if there is pectoral muscle in the image
    If there is, will set all the pectoral muscle values to zero
    
    Pectoral muscle is found via the Hough Transform
    
    First a global threshold is applied. Anything above a certain value will be replaced with zero
    (as the pectoral muscle will have intensity higher than global average)
    Then a Gaussian blur is applied with large variance followed by a sobel filter
    Another threshold is applied to remove the more high detail components
    The Hough Transform is applied to this image, and the most prominent line will correspond to the 
    pectoral muscle.
    
    
    """
    
    def remove_pectoral_muscle(self):
        
        
        self.pectoral_mask = np.zeros(np.shape(self.data), dtype=bool)        #copy the image and apply the first threshold
        #will remove the components of the pectoral muscle
        thresh = np.copy(self.data[0:self.im_height/2, 0:self.im_width/2])
        #will crop the image first, as the pectoral muscle will never be in the right hand side of the image
        #after the orientation has been corrected
        thresh = thresh[:,0:self.im_width]
        
        #now finding the mean value for all pixels that arent the background
        mean_val = np.mean(thresh[thresh > 0])
        thresh[thresh < mean_val] = 0
        #now apply a blur before edge detection
        thresh = filters.gaussian_filter(thresh,30) 
        
        #apply sobel filter
        #replacing with canny filter might make it more suitable for Hough Transform
        edge = sobel(thresh)
        
        thresh_val = np.mean(edge[edge > 0]) * 1.5
        #now will remove the lower value components to focus on the
        #more prominent edges
        edge[edge < thresh_val] = 0
        
        #apply the Hough transform
        h, theta, d = hough_line(edge)
        #using this variable for plotting for vres report
        h_full = np.copy(h)
        #find the most prominent lines in the Hough Transform
        h, theta, d = hough_line_peaks(h, theta, d)
        #use the peak values to create polar form of line describing edge of pectoral muscle
        valid_h = h[(theta < np.pi/8) & (theta > 0 )]
        
        #sometimes there are more than one index found, as the accumulator has returned
        #from the Hough transform has returned the same maximum value. This is a small bug in the
        #hough_line_peaks function, as the point is repeated.  Therefore if we get repeated maximum
        #points, just use the first one, as they are all for the same spot
        #to make sure we use the first maximum found, just using the where function and taking
        #the first element found
        
        #a boolean variable that we will use to decide if we are going to remove the pectoral muscle
        remove_pectoral = []
        
        #if there are any valid peaks found
        
        if(np.sum(valid_h)):
            index = np.where(np.abs(h) == np.max(valid_h))[0][0]
            pectoral_rho = d[index] #will need to account for rho being negative, but shouldnt happen for pectoral muscle case
            pectoral_theta = theta[index]
            remove_pectoral = True
            
        #if there weren't any vali peaks found, then let
        else:
            remove_pectoral = False
            #just setting this to a value to it will fail the check for valid limits
            #not essential but paranoia is really kicking in :)
            pectoral_rho = self.data.shape[0]
            pectoral_theta = 1
            
        #creating a list of positional indicies where the pectoral muscle is
        x_pec = []
        y_pec = []
        #lets check that the region that we have found is correct for the pectoral muscle
        #if the vertical position at the left edge is greater than the height of the
        #image, then what we have found is invalid
        #if it is less than, lets roll and remove it
        if(np.int(pectoral_rho / np.sin(pectoral_theta)) < self.data.shape[0]) & (remove_pectoral):
            
            #now lets get rid of all information to the left of this line
            #as this will be the pectoral muscle
            for x in range(0, np.shape(edge)[1]):
                y = np.int((pectoral_rho - x * np.cos(pectoral_theta))/np.sin(pectoral_theta))
                if( (y >= 0) & (y < self.im_height)):
                    #are in range set all pixels to the left of this x value to zero
                    self.data[0:np.floor(y).astype(int), 0:x] = 0
                    #set these locations in the pectoral muscle binary map to true
                    self.pectoral_mask[0:np.floor(y).astype(int), 0:x] = True
                    #save the location indicies
                    x_pec.append(x)
                    y_pec.append(y)
                    
                    #save the positional indicies to the class member variables, but do it as arrays
            self.x_pec = np.array(x_pec)
            self.y_pec = np.array(y_pec)
            
            self.pectoral_removed = True
            
            
            #plotting figure for VRES report
            
            if(self.plot):
                fig = plt.figure(figsize = (14,7))
                ax2 = fig.add_subplot(1,2,1)
                ax2.imshow(self.original_scan, cmap='gray')
                plt.title('Original Scan')
                plt.axis('off')
                ax2 = fig.add_subplot(1,2,2)
                ax2.imshow(self.data, cmap='gray')
                plt.title('Pectoral Muscle Removed')
                plt.axis('off')
                fig.savefig(os.getcwd() + '/figs/edge_' + self.file_path[-10:-4] +'d.' + 'png',bbox_inches='tight')

                """
                fig = plt.figure()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(self.original_scan, cmap='gray')
                plt.axis('off')
                fig.savefig(os.getcwd() + '/figs/edge_' + self.file_path[-10:-4] +'a.'  + 'png', bbox_inches='tight')
                fig.clf()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(edge)
                plt.axis('off')
                fig.savefig(os.getcwd() + '/figs/edge_' + self.file_path[-10:-4] +'b.' + 'png',bbox_inches='tight')
                fig.clf()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(edge)
                fig.clf()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(h_full, aspect='auto')
                plt.axis('off')
                fig.savefig(os.getcwd() + '/figs/edge_' + self.file_path[-10:-4] +'c.' + 'png',bbox_inches='tight')
                fig.clf()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(edge)
                
                fig.clf()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(self.data, cmap='gray')
                plt.axis('off')
                fig.savefig(os.getcwd() + '/figs/edge_' + self.file_path[-10:-4] +'d.' + 'png',bbox_inches='tight')
                fig.clf()
                ax2 = fig.add_subplot(1,1,1)
                ax2.imshow(edge)
                
                """
                fig.clf()
                plt.close()
                
                
                
            
            
            
            
            
    """
    breast_boundary()
    
    Description:
    
    Will find the outer breast boundary and map it to an array
    To do this, will first enhance the colours, by applying a small blur and then
    taking the logarithm of the input image
    Then will create a binary mask by creating a global threshold
    Will perform edge detection on binary mask and that will give us a first
    approximation of the breast boundary
    
    Will also try to get rid of the other stuff in this image, based on stationary points in the boundary
    or places where the gradient changes very little and gradient has a value near zero for long time near the 
    top or bottom of the image
    
    #contrast enhancement method from GONZALEZ
    GONZALEZ, R. C., and WOODS, R. E. (1992): 'Digital image proces-
    sing' (Addison-Wesley, Reading, MA, 1992)
    KASS, M., WlTKIN, A., and TERZOPOULOS, D., (1988): 'Snakes: active
    contour models', Int. J. Comput. Vis., 1, pp. 321-331
    
    """
    
    def breast_boundary(self):
        
        temp = np.copy(self.data.astype(np.int))
        
        #now do a directional blur to the left
        
        blur_kernel = np.zeros((1,15), dtype=np.int)
        kernel_height = np.shape(blur_kernel)[0]
        kernel_width = np.shape(blur_kernel)[1]
        
        gauss = signal.gaussian(kernel_width * 2, 10, sym=True)
        gauss = gauss[kernel_width::].reshape(1,kernel_width)
        
        #for ii in range(0,kernel_height):
        #    blur_kernel[ii,kernel_width/2:-1] = np.arange(kernel_width/2,0,-1)
            
        #enhance = signal.fftconvolve(temp, gauss, 'same')
        enhance = filters.gaussian_filter(temp, 5)
        
        enhance[enhance < 10] = 0
        enhance[enhance > 10] = 1.0
        label_mask, num_labels = measurements.label(enhance)
        self.breast_mask = np.zeros(np.shape(self.data))
        
        #now will see how many labels there are
        #in an ideal scenario, there should only be two, the breast being scanned,
        #and the background
        #there may be more included, due to things like removing pectoral muscle
        #or checking if some parts of skin above and below the breast have been
        #included in the scan. Also sometimes some artifacts in the background that havent been removed.
        #Can mostly identify these by the original value, and their area
        
        #if the number of labels is greater than two, will get rid of the ones not needed
        
        
        if(num_labels >=  1):
            
            for ii in range(0, num_labels+1):
                
                component = np.zeros(np.shape(label_mask))
                component[label_mask == ii] = 1
                
                #check to see if it is the background
                #if it is the background, all of the places in the original image
                #will be very close/equal to zero
                if( (np.sum(np.multiply(component,self.data))) < (np.sum(component))):
                    #then it is backgoundnp.sum(np.multiply(component,im)                      
                    #TODO - maybe do something here?
                    pass
                
                #now we will see if we are somewhere far to the right, like some stray component
                #that got included for some reason
                #will do that by just seeing if any pixel is located in the first quarter
                #of the width of the screen. If it doesnt have any pixels near the left,
                #then it is some stray artifact
                
                elif(np.sum(component[:,0:np.round( self.im_width / 4)]) == 0):
                    #then this is a stray artifact, and should get rid of it in the label
                    #mask and the original image
                    label_mask[component == 1] = 0
                    self.data[component == 1] = 0
                    
                #now will check if the size is way to small. If it is, then it is most likely not
                #part of the breast, and may be part of the skin included in the scan either above
                #or below the breast
                elif(np.sum(component) < 80000):
                    #then we should get rid of it
                    label_mask[component == 1] = 0
                    self.data[component == 1] = 0
                    
                #if all of the components in the mask are in the bottom half of the image
                #or the top half, will be skin component and not part of the breast
                elif( (np.sum(component[0:self.im_height/2,:]) == 0) | (np.sum(component[self.im_height/2::,:]) == 0) ):
                    label_mask[component == 1] = 0
                    self.data[component == 1] = 0
                    
                    
                #if we have passed all the other tests, then we have found the breast
                else:
                    self.breast_mask = np.copy(component.astype('int8'))
                    
                    
        #if we have gone through all of this and still havent found the breast, we should throw an error
        #lets use this breast mask to set all the background elements to zero
        self.data[ self.breast_mask == 0 ] = np.nan
        
        #use the breast mask we have found to locate the true breast boundary
        self.edge_boundary()
        #now trace the boundary so we can create a parametric model of the breast boundary    
        self.trace_boundary()
        #for when I move the code to Cython
        #self.boundary_y, self.boundary = trace_boundary(im)
        
        #now lets remove any extra bits of skin if we find them
        #will only try and do it if we have found enough bits as well
        if(self.boundary.size > 2000):
            self.remove_skin()
        else:
            print('%s skin not removed' %(self.file_path))
            
        if(self.plot):
            im = np.zeros((np.shape(self.data)))
            im[self.boundary_y, self.boundary] = 1
            #saving a figure so can look at it
            fig = plt.figure(num=None, figsize=(20, 40), dpi=400)
            #ax1 = fig.add_subplot(1,1,1)
            #im1 = ax1.imshow((self.data))
            #fig.colorbar(im1)
            ax2 = fig.add_subplot(1,1,1)
            im2 = ax2.imshow(im)
            fig.savefig(os.getcwd() + '/figs/' + '1_grad_' + self.file_path[-10:-3] + 'png')
            fig.clf()
            plt.close() 
            
            
            
            
        
    """
    edge_boundary()
    
    Description:
    
    Will use the breast mask to find the boundary of the breast
    Will first apply a sobel filter, and search along all points where the image
    does not equal zero, as these will be very close to the true edge, also apply a small blur
    to slightly increase size of initial edge to make sure the true edge will be included
    These step just reduces the search area
    
    Will search along the initial edge, and look at immediate neighbours in the breast mask scan.
    If an immediate neighbour is part of the background, then we have found an edge pixel
    
    Immediate neighbours are those either left, right, down, or up of current pixel
    
    NOTE:
    This code is suuuper slow, so will move it to Cython soon.
    
    """
    
    def edge_boundary(self):
        
        edges = sobel(self.breast_mask)
        #apply small blur to edge
        edges = filters.gaussian_filter(edges,5) 
        
        if(self.pectoral_present):
            #remove edges where the pectoral muscle was
            edges[self.pectoral_mask] = 0
            
            
        if(self.plot):
            fig = plt.figure(num=None, figsize=(80, 50), dpi=300)
            ax1 = fig.add_subplot(1,2,1)
            im1 = ax1.imshow(self.breast_mask)
            ax1 = fig.add_subplot(1,2,2)
            im1 = ax1.imshow(edges)
            fig.savefig(os.getcwd() + '/figs/' + 'plot_' + self.file_path[-10:-3] + 'png')
            fig.clf()
            plt.close()
            
            
            
            
            
        #now find the boundary
        y_temp,boundary_temp = np.where(edges != 0)
        #y_temp = y_temp.astype('uint16')
        #boundary_temp = boundary_temp.astype('uint16')
        
        #creating arrays to store the found boundary
        y = []
        boundary = []
        
        im_height = self.breast_mask.shape[0] - 1
        im_width = self.breast_mask.shape[1] - 1
        
        #list for the search locations
        
        search_x = np.array([-1, 1, 0, 0])
        search_y = np.array([0, 0, -1, 1])
        
        y_lim = np.shape(self.data)[0]        
        x_lim = np.shape(self.data)[1]
        
        for ii in range(0, np.size(y_temp) -1):            
            
            #now search up, left, right, and down
            for jj in range(0, len(search_x)):
                #first just check that we are in a valid search range
                if((y_temp[ii] + search_y[jj]) >= 0) & ((y_temp[ii] + search_y[jj]) < y_lim) & ((boundary_temp[ii] + search_x[jj]) >= 0) &  ((boundary_temp[ii] + search_x[jj]) < x_lim):
                    #then we are in valid searching areas, so lets say hello to our neighbour
                    if(self.breast_mask[y_temp[ii], boundary_temp[ii]] == 1) & (self.breast_mask[y_temp[ii] + search_y[jj] , boundary_temp[ii] + search_x[jj]] == 0):
                        
                        #then this part is on the true boundary
                        y.append( y_temp[ii] )
                        boundary.append( boundary_temp[ii] )
                        break
                    
                    
        #now save the true boundary location
        self.boundary = np.array(boundary)
        self.boundary_y = np.array(y)
        
        
        
        
        
        
        
    def trace_boundary(self):
        
        im = np.zeros(self.data.shape, dtype=np.int)
        im[self.boundary_y,self.boundary] = 1
        
        #will remove pixels at the border, as these will sometimes give weird errors
        im[0:10,:] = 0
        im[-1:-11,:] = 0
        im[:,0:10] = 0
        
        
        #start at the top
        y,x = np.where(im != 0)
        y = y[0]
        x = x[0]
        
        
        #defining the initial boundary points as a large array of zeros
        #this will allow for much faster indexing since I am using a cython
        #function for checking if parts have already been found
        #Will allocate array much larger than the true boundaries will be, 
        #and will just have to truncate it at the end to use only the portions that we selected
        l_y = np.zeros(15000, dtype=np.int16)
        l_x = np.zeros(15000, dtype=np.int16)
        
        first = True
        
        x_s = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int16)
        y_s = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int16)
        
        count = 0
        
        while((x >= 0) & (y >= 0) & (y < im.shape[0]) & (count < 14999)):
            l_y[count] = y
            l_x[count] = x
            found = False
            for ii in range(0,y_s.size):
                #first check that the indicies we will be using are valid
                if((x + x_s[ii]) >= 0) & ((x + x_s[ii]) < im.shape[1]) & ((y + y_s[ii]) >= 0) & ((y + y_s[ii]) < im.shape[0]):
                    #if the pixel we have found is on the boundary
                    if(im[y + y_s[ii], x + x_s[ii]] == 1):
                        #if it is the first pass through, we will add this point
                        if(first):
                            x = x + x_s[ii]
                            y = y + y_s[ii]
                            
                            first = False
                            found = True
                            break
                        
                        #otherwise check we havent already found this point
                        if(not cy_search_prev(l_y,l_x,y + y_s[ii], x + x_s[ii], count)):
                            x = x + x_s[ii]
                            y = y + y_s[ii]
                            found = True
                            break
                        else:
                            pass
                        
            #if we haven'f found another part of the boundary, lets finish up
            if(found == False):
                break
            
            count += 1
            
        #Once we are here, we are done following the line, so will now truncate the
        #line arrays to include just the parts that we used
        
        l_y = l_y[0:count-1]
        l_x = l_x[0:count-1]
        if(self.plot):
            test = np.zeros(im.shape)
            test[l_y, l_x] = 1
            fig = plt.figure(num=None, figsize=(80, 50), dpi=800)
            ax1 = fig.add_subplot(1,2,1)
            im1 = ax1.imshow(test)
            test = np.zeros(im.shape)
            test[self.boundary_y, self.boundary] = 1
            ax1 = fig.add_subplot(1,2,2)
            im1 = ax1.imshow(im)
            #fig.colorbar(im1)
            fig.savefig(os.getcwd() + '/figs/' + 'test_' + self.file_path[-10:-3] + 'png')
            fig.clf()
            plt.close()
            
            
        #now re-write the breast boundary member variables
        if(l_y.size == l_x.size):
            self.boundary = l_x.astype(np.int)
            self.boundary_y = l_y.astype(np.int)
            
            
        
        
    def search_prev(self,l_y,l_x, y, x, count):
        recently_found = False
        if(count < 20):
            lim = count + 1
        else:
            lim = 20
        for jj in range(count,count-lim,-1):
            if(l_y[jj] == y) & (l_x[jj] == x):
                recently_found = True
                break
        
        return recently_found
        
        
        
        
    """
    remove_skin()
    
    Description:
    
    Function uses the breast boundary to locate any extra skin in the image.
    
    Extra skin is found through a point of inflection in the boundary
    Use these points of inflection to decide which parts of the breast to get rid of
    
    """
    
    
    def remove_skin(self):
        
        #first lets smooth the boundary
        x = signal.savgol_filter(self.boundary, 51, 3)
        y = signal.savgol_filter(self.boundary_y, 51, 3)
        #finding the derivatives
        dx = self.deriv(x)
        d2x = self.deriv(dx)
        dy = self.deriv(y)
        d2y = self.deriv(dy)
        
        
        curvature = (dx * d2y - dy * d2x)/((dx**2 + dy**2)**(3/2))
        curvature[np.abs(curvature) > 0.5] = 0
        #apply low pass filter
        curvature = self.hamming_lpf(curvature)
        
        t = np.arange(np.size(curvature))
        curv_x, curv_y = self.stationary_points(curvature)
        
        #use this to find the location of excess skin
        mask = (curvature <  -1.0 * np.abs(3.0 * np.median(curvature)))
        skin = np.where(mask)
        
        if(np.sum(mask[0: round(len(curvature) * 1.0/3.0)])):
            top = skin[0][skin[0] < len(curvature) * 1.0/3.0]
        else:
            top = [1]
            
        if(np.sum(mask[round(len(curvature) * 2.0/3.0)::])):
            bottom = skin[0][skin[0] > len(curvature) * 2.0/3.0]
        else:
            bottom = [len(curvature) -1]
            
            
        #will use these locations to find the top and bottom parts of skin
            
        peaks = np.array([top[-1], bottom[0]])
        
        
        
        
        #now want to find point of inflection in  breast boundary
        #which is where the second derivative will equal zero
        #use the stationary_points function to find any of these
        inflec_y, inflec_x = self.stationary_points(d2x)
        
        #finding peaks in second derivative
        #first filter out small  values
        #d2x[np.abs(d2x) < np.mean(np.abs(d2x)) * 5] = 0
        
        #peaks = np.array( signal.find_peaks_cwt(d2x, np.arange(25,40)))
        #now filter out peaks until we get the one pertaining to any extra skin
        
        #sometimes get peaks in the middle caused by the nipple in the scan
        #can get rid of these peaks by checking the value of the breast boundary
        #at these locations.
        #if the value of the breast boundary close to the maximim at this point,
        #then is most likely due to the nipple
        
        
        #if we actually found any peaks
        if(peaks.size > 0):
            #get rid of peaks near the nipple
            peaks = peaks[self.boundary[peaks] < np.max(self.boundary) * 0.85]
            #remove the extra bit of skin
            #see if there are twwo bits of skin we need to remove
            #if we do have parts to remove
            if(peaks.size):
                if(peaks.size >=2):
                    
                    #first lets check we are actually near the top of the image
                    #we will say it is at the top if it is in the top third of the
                    #image
                    if(self.boundary_y[peaks[0]] <= self.data.shape[0] * (1.0/3.0)):
                        #remove the top bit of skin
                        for ii in range(0, peaks[0]):
                            self.data[self.boundary_y[ii],0:self.boundary[ii] + 2] = np.nan
                        
                #now do the bottom section
                #similarly, we will check that this is in the bottom third of the image
                if(self.boundary_y[peaks[-1]] >= self.data.shape[0] * (2.0/3.0)):
                    for ii in range(peaks[-1], self.boundary.size):
                        self.data[self.boundary_y[ii],0:self.boundary[ii] + 2] = np.nan
                    
                
                
        
        #Code just for making pretty plots to check it is all working :)
        if(self.plot):
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            plt.axis('off')
            im1 = ax1.imshow(self.original_scan, cmap='gray')
            fig.savefig(os.getcwd() + '/figs/' + 'pre_a' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            plt.clf()
            
            ax1 = fig.add_subplot(1,1,1)
            plt.axis('on')
            im1 = ax1.plot(t,curvature)
            plt.title('Curvature of Breast Boundary')
            plt.xlabel('Boundary index')
            plt.ylabel('Curvature')
            fig.savefig(os.getcwd() + '/figs/' + 'pre_b' + self.file_path[-10:-3] + 'eps', bbox_inches='tight')
            plt.clf()

            
            
            ax1 = fig.add_subplot(1,1,1)
            plt.axis('off')
            im1 = ax1.imshow(self.data, cmap='gray')
            fig.savefig(os.getcwd() + '/figs/' + 'pre_c' + self.file_path[-10:-3] + 'png', bbox_inches='tight')



            """
            if(peaks.size > 0):
                im1 = ax1.scatter(self.boundary_y[peaks], d2x[peaks], label='Peaks')
            im2 = ax1.plot(self.boundary_y, np.divide(x.astype(float), np.max(self.boundary.astype(float))) * np.max(d2x), 'r', label='Scaled Breast Boundary')
            im3 = ax1.plot(self.boundary_y, d2x, 'm', label='Second Derivative')
            #fig.colorbar(im1)
            plt.title('Breast Boundary and Second Derivative')
            plt.legend(loc=3)
            fig.savefig(os.getcwd() + '/figs/' + 'deriv_' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            fig.clf()
            plt.close()
            
            
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            im1 = ax1.plot(t,curvature)
            plt.title('Curvature of Breast Boundary')
            plt.xlabel('Boundary index')
            plt.ylabel('Curvature')
            #im1 = ax1.scatter(curv_x, curv_y)
            fig.savefig(os.getcwd() + '/figs/' + 'curv_' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            fig.clf()
        
            
            
            fig = plt.figure()
            plt.axis('off')
            ax1 = fig.add_subplot(1,1,1)
            im1 = ax1.imshow(self.data, cmap='gray')
            fig.savefig(os.getcwd() + '/figs/' + 'pre_' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            fig.clf()
            
            fig = plt.figure()
            plt.axis('off')
            ax1 = fig.add_subplot(1,1,1)
            im1 = ax1.imshow(self.breast_mask)
            fig.savefig(os.getcwd() + '/figs/' + 'msk_' + self.file_path[-10:-3] + 'png', bbox_inches='tight')
            
            """
            
            fig.clf()
            plt.close()
            
            
            
            
            
            
            
        
    """
    stationary_points()
    
    Description:
    
    Function will find stationary points in a 1D array
    
    @param data = input signal
    
    @retval = two arrays containing the stationary point locations (stationary_x and stationary_y)
    
    """

    def stationary_points(self, dx):
        
        stationary_y = []
        stationary_x = []
        #now lets first apply a low pass filter/Gaussian blur to remove any high frequency
        #noise that would mess with the derivative
        #default high standard deviation (sigma = 10) on blur so it gets rid of high frequencies better
        
        
        #temp = filters.gaussian_filter1d(np.asarray(boundary).astype(float) , sigma)
        #dx = np.gradient(temp)
        
        #now will find the stationary points. A stationary point will occur when the derivative changes from pos to negative
        for ii in range(0,np.shape(dx)[0] - 1):
            if( np.sign(dx[ii]) != np.sign(dx[ii+1]) ):
                stationary_y.append(ii)
                stationary_x.append(dx[ii])
                
        return np.asarray(stationary_y), np.asarray(stationary_x)
            
        
        
        
    

    






    def search_microcalcifications(self):
        """
        #testing microcalcification detection
        a = np.copy(self.scan_data.data)
        a = pywt.dwt2(a, 'haar')
        #do it again to the approximation level
        b = pywt.dwt2(a[0], 'haar')
        #now set the approximation coefficients to zero
        b[0][b[0] > 1] =  0
        #now do the reconstruction of the second level
        c = pywt.idwt2(b, 'haar')
        
        a = list(a)
        a[0] = c
        a = tuple(a)
        sig = pywt.idwt2(a, 'haar')
        
        #find absolute value
        sig = np.abs(sig)
        
        #now lets find the mean of the signal
        sig[ sig < (np.nanmean(sig) * 1.5)] = 0
        #now use this on the original to get only spots with high intensity
        sig = sig * self.scan_data
        
        sig[ sig < (np.nanmean(sig) * 1.5)] = 0
        
        #now blur the image with large std to create mask that covers all of the microcalcifications
        sig = filters.gaussian_filter(sig,30) 
        
        #now use this to find a mask of the microcalcifications
        sig[(sig > 0) & (np.isfinite(sig))] = 1
        
        self.microccalcifications_mask = np.array((np.shape(sig)), dtype=bool)
        self.microcalcifications_mask[sig == 1] = True
        self.microcalcifications_mask[sig != 1] = False        
        """
        
        
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow(self.data)
        fig.savefig(os.getcwd() + '/figs/' + self.file_path[-10:-3] + 'png')
        fig.clf()
        plt.close()
        """
        a = []
        b = []
        c = []
        sig = []
        ax1 = []
        ax2 = []
        
        
        
        
        
    def deriv(self, input):
        
        #create the gaussian pdf
        x = np.linspace(-2,2,51)
        normal = np.exp(- (x**2/2) )/np.sqrt(2.0*np.pi)
        #take the derivative of it
        dx = np.gradient(normal)
        #convolve this with the input signal to approximate the derivative
        return np.convolve(input, dx, mode='same')
    
    
    
    
    def hamming_lpf(self, input):
        N = 51
        n = np.arange(N)
        alpha = (N-1)/2
        fc = 0.05
        h = np.zeros(N)
        h[(n-alpha) != 0] = 1.0/(np.pi * (n[(n-alpha) != 0] - alpha)) * np.sin(2.0 * np.pi * fc * (n[(n-alpha) != 0] - alpha))
        h[(n -alpha) == 0] = alpha
        w = signal.hamming(N)
        h_r = h * w
        return signal.convolve(input, h_r, mode='same')
    
    
