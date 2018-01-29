import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
import os
import sys
from PIL import Image



def usage():
    print('First argument should be the directory with the preprocessed scans you want to visualise. These prepocessed scans should be in .npy format')
    
    exit()
    

if __name__ == "__main__":
    
    dir_path = sys.argv[1]
    if(not os.path.isdir(dir_path)):
        print('Supplied path is incorrect')
        usage()
        
    for f in os.listdir(dir_path):
        if f.endswith('.npy'):
            array = np.load(os.path.join(dir_path, f))
            #correct the array to make sure it is in integers and no nan values
            array[np.isnan(array)] = 0
            array[np.isinf(array)] = 0
            #convert it to float, scale it then back to uint8
            array = (np.divide(array.astype(np.float), np.max(array)) * 255.0).astype(np.uint8)
            
            im = Image.fromarray(array)
            #now save the image
            sv_path = os.path.join(dir_path, 'vis', f[0:-3] + 'jpg')
            print sv_path
            im.save(sv_path)
            
            #testing the images
            """
            fig = plt.figure()
            ax2 = fig.add_subplot(1,1,1)
            ax2.imshow(array, cmap='gray')
            fig.savefig('./test.png')
            """
            
            
            
            

