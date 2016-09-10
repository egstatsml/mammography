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





#will use local Probability Density Function (pdf)  and Cumulative Density Function (CDF) to find the label and remove it
#search 50 by 50 area and find local pdf
#if local pdf has very few elements below a threshold and more elements above a threshold, then is probably the label 

#will adaptaviely set threshold based on image parameters
def remove_label(data):
    data_rm = np.copy(data)

    pdf_axis = np.arange(0, np.max(data))
    #first find the threshold value, which will be 95% of maximum value
    threshold_pos = np.round(np.max(data) * 0.8)
    threshold_val = 0.5
    threshold_val_high = 0.99
    #search from right to left, as that is where the label will be

    x = 0
    while (x < np.shape(data)[1] - 200):
        y = 0
        while (y < np.shape(data)[0] - 200):
            #find pdf of region

            pdf = np.histogram( data[y:y+200, x:x+200], bins=np.arange(0,np.max(data)+1), density=True)
            cdf = np.cumsum(pdf[0])
            #see if most of the pixels are background, and if there has been no change over mid part of cdf
            if( (cdf[0] > threshold_val) &( np.abs(cdf[1000] - cdf[100]) < 0.001 )):

                #then this area contains text
                data_rm[y:y+200,x:x+200] = 0

            y = y+200
        x = x+200
        
    return data_rm



"""
check_orientation()

Description:

Will check if the image is the correct orientation,
Eg. breast is on the left hand side of the image.

If it isn't, just fliplr

@param im = input image

@retval image of correct orientation

"""


def check_orientation(im):
    if( sum(im[:,0]) > 100):
        return im
    else:
        return np.fliplr(im)





"""
pectoral_muscle_present()

Description:
Will see if there is pectoral muscle captured in the scan.
Most of the time there isnt, but if there is we need to get rid of it.

If the pixel intensities near the top left corner are significantly higher
than the average intensity, will say that there is pectoral muscle present

@param im = input image

@retval boolean to say if there is or isnt pectoral muscle present


"""

def pectoral_muscle_present(im):

    #find the mean value for just the pixels that arent background
    mean_val = np.mean(im[im > 0])
    #now will search the top left corner
    count = 0
    for y in range(0, 100):
        for x in range(0, 50):
            if(im[y,x] > mean_val):
                count = count+1

    #if the majority of these pixels read greater than the average, then
    #pectoral muscle is said to be present and we should find it
    if(count > 100*50/2):
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



@param im = input image

@retval image with pectoral muscle removed

"""
def remove_pectoral_muscle(im):


    if( pectoral_muscle_present(im) ):

        #will copy the image and apply the first threshold
        #will remove the components of the pectoral muscle
        thresh = np.copy(im)
        #will crop the image first, as the pectoral muscle will never be in the right hand side of the image
        #after the orientation has been corrected
        thresh = thresh[:,0:np.shape(im)[1]/2]

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
            if( (y >= 0) & (y < np.shape(im)[0])):
                #are in range set all pixels to the left of this x value to zero
                im[0:np.floor(y).astype(int), x] = 0

    #return the original image with the pectoral muscle removed
    return im





"""
breast_boundary()

Description:

Will find the outer breast boundary and map it to an array
To do this, will first enhance the colours, by applying a small blur and then
taking the logarithm of the input image
Then will create a binary mask by creating a global threshold
Will perform edge detection on binary mask and that will give us a first
approximation of the breast boundary

Will also try to get rid of the other stuff in this image

#contrast enhancement method from GONZALEZ
GONZALEZ, R. C., and WOODS, R. E. (1992): 'Digital image proces-
sing' (Addison-Wesley, Reading, MA, 1992)
KASS, M., WlTKIN, A., and TERZOPOULOS, D., (1988): 'Snakes: active
contour models', Int. J. Comput. Vis., 1, pp. 321-331



@param im = input image
@retval array containing map of breast boundary


"""



def breast_boundary(im):

    temp = np.copy(im)
    
    temp = filters.gaussian_filter(temp,0.5) 
    #do some contrast enhancement
    enhance = np.log10( 1 + temp)
    enhance[enhance > 0] = 1.0
    label_mask, num_labels = measurements.label(enhance)
    breast = np.zeros(np.shape(im))
    
    #now will see how many labels there are
    #in an ideal scenario, there should only be two, the breast being scanned,
    #and the background
    #there may be more included, due to things like removing pectoral muscle
    #or checking if some parts of skin above and below the breast have been
    #included in the scan. Also sometimes some artifacts in the background that havent been removed.
    #Can mostly identify these by the original value, and their area

    #if the number of labels is greater than two, will get rid of the ones not needed

    if(num_labels > 2):
        for ii in range(0, num_labels):
            component = np.zeros(np.shape(label_mask))
            component[label_mask == ii] = 1

            #check to see if it is the background
            #if it is the background, all of the places in the original image
            #will be very close/equal to zero
            if( (np.sum(np.multiply(component,im))) < (np.sum(component))):
               #then it is backgoundnp.sum(np.multiply(component,im)                      
               #TODO - do something here
                print('its background yo')

            #now we will see if we are somewhere far to the right, like some stray component
            #that got included for some reason
            #will do that by just seeing if any pixel is located in the first quarter
            #of the width of the screen. If it doesnt have any pixels near the left,
            #then it is some stray artifact
               
            elif(np.sum(component[:,0:np.round( np.shape(im)[1] / 8)]) == 0):
                #then this is a stray artifact, and should get rid of it in the label
                #mask and the original image
                label_mask[component == 1] = 0
                im[component == 1] = 0

            #now will check if the size is way to small. If it is, then it is most likely not
            #part of the breast, and may be part of the skin included in the scan either above
            #or below the breast
            elif(np.sum(component) < 5000):
               #then we should get rid of it
               label_mask[component == 1] = 0
               im[component == 1] = 0


            #will check if any other small bits of skin above or below the breast made it through
            elif( (np.sum(component[0:np.round(np.shape(component)[0]/3), :]) == 0) | (np.sum(component[np.round(2 * np.shape(component)[0]/3):np.shape(component)[0], :])  == 0)):
               label_mask[component == 1] = 0
               im[component == 1] = 0


            #if we have passed all the other tests, then we have found the breast
            else:
                breast = component


                
    edges = feature.canny(breast)

    coords = corner_peaks(corner_harris(edges, k=0.01), min_distance=5)
    coords_subpix = corner_subpix(edges, coords, window_size=13)




    fig, ax = plt.subplots()
    ax.imshow(edges, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)

    plt.show()


    
    print(coords)
    print(coords_subpix)

    plt.figure()
    plt.subplot(121)
    plt.imshow(enhance)
    plt.subplot(122)
    
    plt.imshow(edges)
    plt.show()


    plt.figure()
    plt.subplot(111)
    plt.imshow(label_mask)
    plt.show()

    return edges



    


if(__name__ == "__main__"):

    #load in a file
    #file_path = '/home/ethan/DREAM/pilot_images/111359.dcm' #image without pectoral muscle
    file_path = '/home/ethan/DREAM/pilot_images/134060.dcm' #image wwith pectoral muscle
    file_path = '/home/ethan/DREAM/pilot_images/502860.dcm' #malignant case
    file = dicom.read_file(file_path)

    #convert to an array and reshape it to correct size
    im = np.fromstring(file.PixelData,dtype=np.int16).reshape((file.Rows,file.Columns))
    #see if we need to flip it
    im = check_orientation(im)
    
    test = remove_label(im)
    test = remove_pectoral_muscle(test)
    boundary = breast_boundary(test)
    #get a histogram as well
    hist = np.histogram(im, bins=2**12)
    print np.max(hist[0])
    plt.figure()
    plt.subplot(121)
    plt.imshow(im, cmap='Greys_r')
    plt.colorbar()
    plt.title('Original Mammogram')

    plt.subplot(122)
    plt.imshow(test)
    plt.title('Removed Label and Pectoral Muscle')
    plt.show()


    #just testing the wavelet transform from PyWavelet
    coeffs = pywt.dwt2(test, 'sym20')


    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    LL, (LH, HL, HH) = coeffs
    fig = plt.figure()
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=12)
    plt.show()


    plt.figure()
    titles = ['Horizontal', 'Vertical', 'Diagonal']
    
    decomp = pywt.wavedec2(test, 'bior4.4', mode='symmetric')
    wav_i = []
    """
    for ii in range(1, len(decomp)):
        filter_bank = decomp[ii]
        print(np.shape(filter_bank))
        for jj in range(0, 3):
            print(jj)
            filter_temp = filter_bank[jj]
            print(np.shape(filter_temp))
            #plt.imshow(signal.fftconvolve(test,filter_temp))
            plt.imshow(filter_temp)
            plt.title(titles[jj])
            plt.colorbar()
            plt.show()
            plt.clf()
                       
    


    """

    
