import dicom
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy



def sobel(mono):
    #creating sobel filter
    dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    dy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    im_dx = np.zeros(np.shape(mono))
    im_dy = np.zeros(np.shape(mono))

    for x in range(0,np.shape(mono)[1]-2):
        for y in range(0,np.shape(mono)[0]-2):
            
            im_dx[y,x] = np.divide( np.sum(np.multiply(mono[y:y+3,x:x+3], dx)), 6)
            im_dy[y,x] = np.divide( np.sum(np.multiply(mono[y:y+3,x:x+3], dy)), 6)
            

    #finding magnitude to get of dx and dy
    edges = np.sqrt( np.power(im_dx,2) + np.power(im_dy,2)) 
    theta = np.arctan( im_dy, im_dx )

    return (edges, theta, im_dx, im_dy)






if(__name__ == "__main__"):

    #load in a file
    file_path = '/home/ethan/DREAM/pilot_images/111359.dcm'
    file = dicom.read_file(file_path)

    #convert to an array and reshape it to correct size
    im = np.fromstring(file.PixelData,dtype=np.int16).reshape((file.Rows,file.Columns))
    #get a histogram as well
    hist = np.histogram(im, bins=2**12)
    print np.max(hist[0])
    plt.figure()
    plt.subplot(121)
    plt.imshow(im)
    plt.colorbar()
    plt.title('Original Mammogram')

    plt.subplot(122)
    plt.stem(hist[0])
    plt.axis([0,2**12,0, 150000])
    plt.title('Histogram of Original Mammogram')
    plt.show()
    plt.clf()
    


    #do some edge detection
    edge_detect = sobel(im)
    edge_im = edge_detect[0] #get just the edge image
    #a bit of thresholding
    edge_im[np.abs(edge_im) < 10] = 0
    #now set any edge value that isnt zero to the phase of the edge

    
    plt.figure()
    plt.imshow(edge_im)
    plt.colorbar()
    plt.title('Sobel Filter Applied')
    plt.show()
    
