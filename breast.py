
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
from scipy import signal as signal
from scipy.ndimage import filters as filters
from scipy.ndimage import measurements as measurements
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)


from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy import ndimage as ndi

import pywt




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
        self.nipple_x = 0
        self.nipple_y = 0
        self.threshold = 0                #threshold for segmenting fibroglandular disk
        
        if(file_path != None):
            self.initialise(file_path)
            
            
            
            
    """
    breast.initialise()
    
    Desccription:
    
    Will take a file path and load the data and store it in a numpy array
    Also set the width and height of the image, and check the orientation of the breast
    in the scan is correct (ie. breast is on the left)
    
    @param file_path = string containing the location of the file you want to read in
    
    """
    def initialise(self, file_path):
        file = dicom.read_file(file_path)
        self.data = np.fromstring(file.PixelData,dtype=np.int16).reshape((file.Rows,file.Columns))
        #convert the data to floating point now
        self.data = self.data.astype(float)
        self.original_scan = np.copy(self.data)
        self.check_orientation()
        
        
        
    def cleanup(self):
        self.data = []                           #the mammogram
        self.original_scan = []
        self.pectoral_mask = []                  #binary map of pectoral muscle
        self.breast_mask = []                    #binary map of breast
        self.fibroglandular_mask = []            #binary map of fibroglandular tissue
        
        
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
        
        
        x = 0
        while (x < (self.im_width - 200) ):
            y = 0
            while (y < (self.im_height- 200) ):
                
                #find pdf and cdf of local region
                pdf = np.histogram( self.data[y:y+200, x:x+200], bins=np.arange(0,np.max(self.data)+1), density=True)
                cdf = np.cumsum(pdf[0])
                #see if most of the pixels are background, and if there has been no change over mid part of cdf
                if( (cdf[0] > threshold_val) &( np.abs(cdf[1000] - cdf[100]) < 0.001 )):
                    
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
        mean_val = np.mean(self.data[self.data > 0])
        #now will search the top left corner
        count = 0
        for y in range(0, 100):
            for x in range(0, 50):
                if(self.data[y,x] > mean_val):
                    count = count+1

        #if the majority of these pixels read greater than the average, then
        #pectoral muscle is said to be present and we should find it
        if(count >= (100.0*50.0*0.5)):
            self.pectoral_present = True
            return True
        else:
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


        self.pectoral = np.zeros(np.shape(self.data), dtype=bool)        #copy the image and apply the first threshold
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
        #find the most prominent lines in the Hough Transform
        h, theta, d = hough_line_peaks(h, theta, d)
        #use the peak values to create polar form of line describing edge of pectoral muscle
        valid_h = h[(theta < np.pi/8) & (theta > 0 )]
        
        pectoral_rho = d[h == max(valid_h.min(), valid_h.max(), key=abs)] #will need to account for rho being negative, but shouldnt happen for pectoral muscle case
        pectoral_theta = theta[d == pectoral_rho]
        
        #now lets get rid of all information to the left of this line
        #as this will be the pectoral muscle
        for x in range(0, np.shape(edge)[1]):
            y = np.int((pectoral_rho - x * np.cos(pectoral_theta))/np.sin(pectoral_theta))
            if( (y >= 0) & (y < self.im_height)):
                #are in range set all pixels to the left of this x value to zero
                self.data[0:np.floor(y).astype(int), x] = 0
                #set these locations in the pectoral muscle binary map to true
                self.pectoral[0:np.floor(y).astype(int), x] = True
                
                
                
        self.pectoral_removed = True
        





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

        blur_kernel = np.zeros((3,15), dtype=np.int)
        kernel_height = np.shape(blur_kernel)[0]
        kernel_width = np.shape(blur_kernel)[1]
        for ii in range(0,kernel_height):
            blur_kernel[ii,kernel_width/2:-1] = np.arange(kernel_width/2,0,-1)
            
        enhance = signal.fftconvolve(temp, blur_kernel, 'same')
        

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
                    #TODO - do something here
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
                    self.breast_mask = np.copy(component.astype('uint8'))
                    
        
        #if we have gone through all of this and still havent found the breast, we should throw an error
        #lets use this breast mask to set all the background elements to zero
        #MAY WANT TO SET THEM TO NAN AT A LATER STAGE, WILL SEE HOW THAT WOULD WORK
        self.data[ self.breast_mask == 0 ] = np.nan
        
        edges = sobel(self.breast_mask.astype(float))
        #get just a graph of the boundary
        #y is the y position of the image, and will use as independant variable here
        y,boundary = self.thin_boundary(edges)
        
        #remove any extra skin bits that have made it this far
        y, boundary = self.remove_skin(boundary, y)
        
        #now will check to see if we have a point of inflection, caused by some skin included in the scan
        #finding stationary points in the boundary
        stationary_y, stationary_x =  self.stationary_points(boundary)
        
        #stationary point in the where the boundary is maximum will likely be the nipple
        if(len(stationary_y) > 0):
            self.nipple_x = np.max(stationary_x)
            self.nipple_y = stationary_y[stationary_x == self.nipple_x] + y[0]
            
        self.boundary = boundary.astype(np.int)
        self.boundary_y = y
        
        
        #plt.figure()
        #plt.plot(self.boundary)
        #plt.show()
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    def thin_boundary(self, edges):
        y_temp, boundary_temp = np.where(np.abs(edges) > 0.01)
        
        #now will loop through to make sure there is only one boundary location for each boundary position
        #if there are multiple boundary points at the same location, will just take the outermost position
        
        
        #finding the range of our loop
        y_min = np.min(y_temp)
        y_max = np.max(y_temp)
        y = np.arange(y_min,y_max)
        boundary = np.zeros(np.shape(y))
        
        #want last value inclusive
        for ii in range(0, np.shape(y)[0]):
            boundary[ii] = np.max(boundary_temp[y_temp == ii+y_min])
            
            
        return y, boundary
    
    
    
    
    """
    stationary_points()
    
    Description:
    
    Function will find stationary points in a 1D array
    
    @param data = input signal
    
    @retval = two arrays containing the stationary point locations (stationary_x and stationary_y)
    
    """
    
    def stationary_points(self, boundary, sigma = 20):
        
        stationary_y = []
        stationary_x = []
        #now lets first apply a low pass filter/Gaussian blur to remove any high frequency
        #noise that would mess with the derivative
        #default high standard deviation (sigma = 10) on blur so it gets rid of high frequencies better
        
        
        temp = filters.gaussian_filter1d(np.asarray(boundary).astype(float) , sigma)
        dx = np.gradient(temp)
        
        #now will find the stationary points. A stationary point will occur when the derivative changes from pos to negative
        for ii in range(0,np.shape(dx)[0] - 1):
            if( np.sign(dx[ii]) != np.sign(dx[ii+1]) ):
                stationary_y.append(ii)
                stationary_x.append(boundary[ii])
        return np.asarray(stationary_y), np.asarray(stationary_x) 






    """
    
    
    TODO
    
    USE Y ORIG VALUE BEFORE CHANGING BOUNDARIES
    remove_skin()
    
    Description:
    Will remove any extra bits of skin at the bottom or top of the scan
    
    Will breast boundary to do this, by finding points that are near stationary near the
    top or bottom of the scan
    
    @param boundary = array containing the breast boundary locations
    
    """
    
    def remove_skin(self, boundary, y):
        
        
        #plt.figure()
        #plt.imshow(self.data)
        #plt.show()
        
        y_orig = y[0]
        
        temp = filters.gaussian_filter1d(np.asarray(boundary).astype(float) , 5)
        
        
        dx = np.gradient(temp)
        temp = filters.gaussian_filter1d(dx , 20)
        d2x = np.gradient(temp)
        temp = filters.gaussian_filter1d(d2x , 10)
        
        #find the stationary points
        stationary_y, stationary_x =  self.stationary_points(boundary)
        #now will see where the stationary points are
        #if they are near the end then probably where we go from skin to
        #breast
        
        
        #check near top of image
        if( (stationary_y[0] < self.im_height /4.0 ) ):
            #then it is skin and we should get rid of it
            boundary = boundary[y > stationary_y[0]]
            y = y[y > stationary_y[0]]
            self.data[0:y[0],:] = np.nan
            
            
            
        elif( np.max(np.abs(d2x[0:self.im_height/4])) > (np.max(np.abs(d2x)) * 0.6) ):
            lim_pos =  np.where( np.abs(d2x) == np.max(np.abs(d2x[0:self.im_height/4])))[0]
            boundary = boundary[lim_pos::]
            y = y[lim_pos::]
            
            
        #now lets check near the bottom of the image for any extra skin underneath
        if(stationary_y[-1] > self.im_height * (3.0/4.0) ):
            boundary = boundary[y < stationary_y[-1]]
            y = y[y < stationary_y[-1]]
            self.data[y[-1] + y_orig::,:] = np.nan
            
            
        elif( np.max(np.abs(d2x[np.round(self.im_height * (3/4))::])) > (np.max(np.abs(d2x)) * 0.6) ):
            lim_pos =np.int(np.where( np.abs(d2x) == np.max(np.abs(d2x[np.round(self.im_height * (3/4))::])))[0] + int(self.im_height*(3.0/4.0)))
            boundary = boundary[0:lim_pos+y[0]]
            y = y[0:lim_pos+y[0]]
            self.data[lim_pos + y_orig::,:] = np.nan
            
            
        return y, boundary






    
        
    """
    cross_entropy_threshold()
    
    Description:
    Creating adaptive threshold for segmenting fibroglandular disk
    and dense breast tissue.
    
    The function uses the minimum cross entropy between the two classes
    to find the threshold. Minimum cross entropy developed by Li, C. and Lee, C
    Is a wrapper function, the cross entropy (eta)  is calculated using a helper function
    

    Also creating a mask of the breast boundary as well. The breast skin has higher intensity, 
    and won't contain any useful information, so wont use the breast skin region for thresholding


    Reference
    
    @article{Li_1993,
            doi = {10.1016/0031-3203(93)90115-d},
            url = {http://dx.doi.org/10.1016/0031-3203(93)90115-d},
            year = 1993,
            month = {apr},
            publisher = {Elsevier {BV}},
            volume = {26},
            number = {4},
            pages = {617--625},
            author = {C.H. Li and C.K. Lee},
            title = {Minimum cross entropy thresholding},
            journal = {Pattern Recognition}
    }
    """
    
    def cross_entropy_threshold(self):
        
        #deep copy of image data
        temp = np.copy(self.data)
        #creating a mask of the breast boundary
        edge_mask = np.zeros(np.shape(self.data), dtype=np.uint8)
        

        for ii in range(0, len(self.boundary)):
            for jj in range(0, 20):
                edge_mask[self.boundary_y[ii], self.boundary[ii] - 20 + jj] = 1
                
                
        #now lets just use the points where it isnt the edge of the skin
        temp = temp[edge_mask == 0]
        temp = temp[np.isfinite(temp)]
        #lets get rid of zero value pixels
        temp = temp[temp > 0]        
        eta = 10e10        #large value of eta that will be overwritten
        
        hist = np.histogram(temp, bins = 4096)
        hist = hist[0]
        for t in range(300, 3000):
            current_eta = self.calc_eta(t, hist)
            if(current_eta < eta):
                eta = current_eta
                self.threshold = t
                
                
        
        #creating mask of breast boundary
        
        pdf = np.histogram(temp, 4096, density=True)
        cdf = np.cumsum(pdf[0])
        
        #find the 0.95 value in the cdf
        #the axis is still in the pdf, and use this to find the pixel value
        axis = pdf[1][0:-1]
        upper_limit = axis[cdf >= 0.95]
        upper_limit = upper_limit[0]
        
        #now will use this to create a binary mask of the fibroglandular disk/dense tissue
        self.fibroglandular_mask = (self.data > self.threshold) & (self.data < upper_limit)
        
        
        
    """
    calc_eta()
    
    Description:
    Helper function used to calculate the cross entropy (eta) for different threshold values
    Cross entropy calculations from paper referenced for cross_entropy_threshold()
    
    The cross entropy between the data above and below the threshod value is found
    
    @param t = threshold value
    @param hist = histogram of data
    
    @retval eta = cross entropy
    
    """
    
    def calc_eta(self, t, hist):
        
        #creating arrays for the two data sets
        mu_1_data = hist[1:t-1]
        mu_2_data = hist[t:-1]
        
        #creating index values
        mu_1_range = np.arange(1,t-1)
        mu_2_range = np.arange(t,4096-1)
        #numerator parts for cross entropy
        mu_1_num = mu_1_data*mu_1_range
        mu_2_num = np.multiply(mu_2_data, mu_2_range)
        
        mu_1 = (np.sum(mu_1_num)) / (np.sum(mu_1_data))
        mu_2 = (np.sum(mu_2_num)) / (np.sum(mu_2_data))
        
        #calculating cross entropy
        eta = np.sum( np.multiply(mu_1_num, np.divide(np.log(mu_1_range), mu_1))) + np.sum( np.multiply(mu_2_num, np.divide(np.log(mu_2_range), mu_2)))
        
        return eta                                  
