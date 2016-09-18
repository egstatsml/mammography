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
from feature_extract import feature
import pywt
from sklearn.cluster import KMeans
from skimage import measure

from scipy.cluster import vq as vq
import skfuzzy as fuzz



a = spreadsheet()

#load in a file
#file_path = '/home/ethan/DREAM/pilot_images/111359.dcm' #image without pectoral muscle
file_path = '/home/ethan/DREAM/pilot_images/134060.dcm' #image with pectoral muscle
#file_path = '/home/ethan/DREAM/pilot_images/502860.dcm' #malignant case
current_scan = feature(file_path)


titles = ['Horizontal', 'Vertical', 'Diagonal']
data_mean = np.mean(current_scan.data[current_scan.data > 0])/2.0
test = np.copy(current_scan.data)

print(data_mean)
var = ( (data_mean - 100)/2)**2

test = np.multiply( 1/(np.sqrt(2.0 * np.pi*var) ), np.exp( -( (test - data_mean)**2.0)/(2.0 * var)))

enhanced = np.log(1 + current_scan.data)
enhanced[enhanced > np.max(enhanced) * (4000.0/2**12)] = 0



plt.figure()
plt.imshow(enhanced)
plt.title('Logarithmic Enhancement')
plt.colorbar()
plt.show()



plt.figure()
plt.imshow(current_scan.data)
plt.title('Current Scan')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(current_scan.breast_mask)
plt.show()
pdf = np.histogram( current_scan.data[current_scan.breast_mask == 1], bins=np.arange(0,500), density=True)
cdf = np.cumsum(pdf[0])


plt.figure()
plt.subplot(211)
plt.plot(np.arange(0, pdf[1][-1]),pdf[0])
plt.title('PDF of log Enhanced scan')
plt.subplot(212)
plt.plot(np.arange(0, pdf[1][-1]), cdf)
plt.title('CDF of log Enhanced scan')
plt.show()

current_scan.get_features()









"""
enhanced = np.copy(current_scan.data)
enhanced[enhanced > np.mean(enhanced)] = enhanced[enhanced > np.mean(enhanced)]*10
enhanced = filters.gaussian_filter((enhanced),30) 
enhanced[enhanced < np.mean(enhanced[enhanced > 0]*(4.0/3.0))] = 0
contour = measure.find_contours(enhanced, 1.0)

plt.figure()
plt.imshow(sobel(enhanced))
plt.show()


fig, ax = plt.subplots()
ax.imshow(enhanced, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contour):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


"""
"""
#lets do some clustering
centre,dist = vq.kmeans(current_scan.data, 3)
code, distance = vq.vq(current_scan.data, centre)
res = centre[code]
res2 = res.reshape(np.shape(current_scan.data))
plt.figure()
plt.imshow(res2)
plt.show()
"""
"""

im_size = np.shape(current_scan.data)

test = np.copy(filters.gaussian_filter(current_scan.data,10)).reshape(im_size[0] * im_size[1],1)
print(np.shape(test))
kmeans = KMeans(3,init='k-means++')
z = kmeans.fit_predict(test)
print(np.shape(z))
z = z.reshape(im_size)

plt.figure()
plt.imshow(z)
plt.show()
"""

