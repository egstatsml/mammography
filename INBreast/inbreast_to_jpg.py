#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.ndimage.measurements import label
import xml.etree.cElementTree as ET
from PIL import Image
import dicom












    
if __name__ == "__main__":
    
    root = "/media/dperrin/Data/INBreast/AllDICOMs/" #../data/"
    out_path = "/media/dperrin/Data/INBreast/Images/" #../data/"
    
    #for this test will just loop through all the files in the data folder
    for file_name in os.listdir(root):
        try:
            file_path = os.path.join(root, file_name)
            file = dicom.read_file(file_path)
            data = np.fromstring(file.PixelData,dtype=np.int16).reshape((file.Rows,file.Columns))
            
            #rescale it to the correct pixel levels
            data = np.divide(data, np.max(data)) * 255
            data = data.astype(np.uint8)
            sv = os.path.join(out_path, file_name[0:8]+'.jpg')
            print sv
            im_jpg = Image.fromarray(data)
            im_jpg.save(sv)
            
        except:
            raise
            pass
