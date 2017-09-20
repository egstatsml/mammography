import numpy as np
from PIL import Image
import dicom
import os
import xml.etree.cElementTree as ET









def create_xml(fname, im, train_test):
    
    #train test will be a string saying pilot or val
    annotation_dir = "/media/dperrin/Data/DDSM/data/" + train_test + "_annotations"
    im_dir = "/media/dperrin/pilot_images"
    fname = os.path.basename(fname)[0:-3] + "jpg"
    if('RIGHT' in fname):
        pose = "right"
    else:
        pose = "left"
            
    #reading the size using the indecies back the front because it is
    #an Imgae object and not an arrow
    #
    #Why do they do this to me :(
    height = im.size[1] 
    width = im.size[0]
    
    top_y = 100
    top_x = 100    
    bottom_y = 500
    bottom_x = 500
    #make sure all the bounding boxes are within reasonable ranges
    assert(bottom_y < height)
    assert(bottom_x < width)
    
    
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = fname
    #size data for the image
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz,"height").text = str(height) 
    ET.SubElement(sz,"width").text = str(width)
    ET.SubElement(sz,"depth").text = "1"
    
    #object information (which is the lesion found)
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, 'name').text = 'lesion'
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(top_x)
    ET.SubElement(bndbox, "ymin").text = str(top_y)
    ET.SubElement(bndbox, "xmax").text = str(bottom_x)
    ET.SubElement(bndbox, "ymax").text = str(bottom_y)
    
    #now write the XML file
    sv_path = os.path.join(annotation_dir, fname[0:-3] + 'txt')
    tree = ET.ElementTree(root)
    tree.write( sv_path )
    print sv_path













if __name__ == "__main__":
    
    #want to load in all the images in our dataset to locate lesions, and
    #want to convert them to a format that is suitable for the faster rcnn network
    train_test = "val"
    
    in_path = "/media/dperrin/" + train_test + "_images"
    out_path = "/media/dperrin/Data/DDSM/data/val_images"
    
    for filenames in os.listdir(in_path):
        print filenames
        file = dicom.read_file(os.path.join(in_path,filenames))
        data = np.fromstring(file.PixelData,dtype=np.int16).reshape((file.Rows,file.Columns))
        #scale it
        im = np.divide(data, np.max(data)/255).astype(np.uint8)
        #convert it to a jpeg
        im_jpg = Image.fromarray(im)
        
        #create an XML file for this scan
        create_xml(filenames, im_jpg, train_test)
        
        
        print filenames[0:-4]
        #now want to save it
        im_jpg.save(os.path.join(out_path, filenames[0:-3]+'jpg'))
        
