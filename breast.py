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
    def __init__(self, file_path = False):


        self.data = []                          #the mammogram
        self.pectoral_mask = []                  #binary map ofpectoral muscle
        self.breast_mask = []                    #binary map of breast
        self.pectoral_present = False
        self.pectoral_muscle_removed = False
        self.label_removed = False
        self.width = 0
        self.height = 0
        self.im_width = 0
        self.im_height = 0
        self.area = 0
        self.boundary = []
        self.nipple_x = 0
        self.nipple_y = 0
        
        if(file_path != False):
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
        self.im_width = np.shape(self.data)[1]
        self.im_height = np.shape(self.data)[0]
        #check correct orientation of the image
        self.check_orientation()
        






            
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
        if( sum(self.data[:,0]) < 100):
            self.data = np.fliplr(self.data)





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
                    #then this area contains text
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
        if(count > 100*50/2):
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
        self.pectoral = np.zeros(np.shape(self.data), dtype=bool)        #cpy the image and apply the first threshold
        #will remove the components of the pectoral muscle
        thresh = np.copy(self.data)
        #will crop the image first, as the pectoral muscle will never be in the right hand side of the image
        #after the orientation has been corrected
        thresh = thresh[:,0:self.im_width/2]

        #now finding the mean value for all pixels that arent the background
        mean_val = np.mean(thresh[thresh > 0])
        thresh[thresh > mean_val] = 0
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
        pectoral_rho = np.max(d) #will need to account for rho being negative, but shouldnt happen for pectoral muscle case
        pectoral_theta = theta[d == pectoral_rho]

        #now lets get rid of all information to the left of this line
        #as this will be the pectoral muscle
        for x in range(0, np.shape(edge)[1]):
            y = (pectoral_rho - x * np.cos(pectoral_theta))/np.sin(pectoral_theta)
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

        temp = np.copy(self.data)

        temp = filters.gaussian_filter(temp,0.5) 
        #do some contrast enhancement
        enhance = np.log10( 1 + temp)
        enhance[enhance > 0.1] = 1.0
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
                    print('its background yo')

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

                #if we have passed all the other tests, then we have found the breast
                else:
                    print('found the breast')
                    self.breast_mask = component
                    

        #lets use this breast mask to set all the background elements to zero
        #MAY WANT TO SET THEM TO NAN AT A LATER STAGE, WILL SEE HOW THAT WOULD WORK
        self.data[ self.breast_mask == 0 ] = 0
        
        print('finding the edges')
        edges = feature.canny(self.breast_mask)
        
        #get just a graph of the boundary
        #y is the y position of the image, and will use as independant variable here
        y,boundary = self.thin_boundary(edges)

        print(boundary)
        #remove any extra skin bits that have made it this far
        boundary,y = self.remove_skin(boundary, y)
        #now will check to see if we have a point of inflection, caused by some skin included in the scan
        print(boundary)
        #finding stationary points in the boundary
        stationary_y, stationary_x =  self.stationary_points(boundary)

        #stationary point in the where the boundary is maximum will likely be the nipple
        self.nipple_x = np.max(stationary_x)
        self.nipple_y = stationary_y[stationary_x == self.nipple_x] + y[0]
        
        self.boundary = boundary.astype('uint8')

        return edges









    def thin_boundary(self, edges):
        y_temp, boundary_temp = np.where(edges == 1)

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

        print('here')
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

        
        y_orig = y[0]
        
        temp = filters.gaussian_filter1d(np.asarray(boundary).astype(float) , 5)

        
        plt.figure()
        plt.subplot(311)
        plt.title('boundary')
        plt.plot(boundary)

        dx = np.gradient(temp)
        temp = filters.gaussian_filter1d(dx , 20)
        d2x = np.gradient(temp)
        temp = filters.gaussian_filter1d(d2x , 10)
        
        test = signal.find_peaks_cwt(temp, np.arange(100,500))
        print(test)
        plt.subplot(312)
        plt.title('first derivative')
        plt.plot(dx)

        plt.subplot(313)
        plt.plot(d2x)
        plt.title('second derivative')
        plt.show()


        #find the stationary points
        stationary_y, stationary_x =  self.stationary_points(boundary)
        #now will see where the stationary points are
        #if they are near the end then probably where we go from skin to
        #breast


        #check near top of image
        if( (stationary_y[0] < self.im_height /4.0 ) ):
            #then it is skin and we should get rid of it
            print('upper')
            boundary = boundary[y > stationary_y[0]]
            y = y[y > stationary_y[0]]
            self.data[0:y[0],:] = 0
            

        if( np.max(np.abs(d2x[0:self.im_height/4])) > (np.max(np.abs(d2x)) * 0.6) ):
            lim_pos =  np.where( np.abs(d2x) == np.max(np.abs(d2x[0:self.im_height/4])))[0]
            print lim_pos
            print np.shape(lim_pos)
            boundary = boundary[lim_pos::]
            y = y[lim_pos::]
            self.data[0:lim_pos,:] = 0

            
        
        if(stationary_y[-1] > self.im_height * (3.0/4.0) ):
            print('lower')
            boundary = boundary[y < stationary_y[-1]]
            y = y[y < stationary_y[-1]]
            self.data[y[-1]::,:] = 0


        if( np.max(np.abs(d2x[self.im_height * (3/4)::])) > (np.max(np.abs(d2x)) * 0.6) ):
            lim_pos = np.where( np.abs(d2x) == np.max(np.abs(d2x[self.im_height *(3/4)::])))[0]

            boundary = boundary[0:lim_pos+y[0]]
            y = y[0:lim_pos+y[0]]
            print(np.shape(lim_pos))
            print('pleeease')
            
            self.data[lim_pos + y_orig::,:] = 0


        return boundary, y
