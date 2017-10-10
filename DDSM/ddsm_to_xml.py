#!/usr/bin/python


import os
import sys
import numpy as np
import xml.etree.cElementTree as ET
from PIL import Image


"""
ddsm_to_xml.py

Description:
 Converts the annotated files from the DDSM database into an XML file that can be more easily handled by current RCNN examples 


import xml.etree.cElementTree as ET

root = ET.Element("root")
doc = ET.SubElement(root, "doc")

ET.SubElement(doc, "field1", name="blah").text = "some value1"
ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

tree = ET.ElementTree(root)
tree.write("filename.xml")

will generate something like

<root>
 <doc>
     <field1 name="blah">some value1</field1>
     <field2 name="asdfasd">some vlaue2</field2>
 </doc>

</root>

"""






def get_lesion_type(f):

    lesion_type = None
    for line in f:
        #find the type of lesion here
        if("LESION_TYPE" in line):
            #if microcalcifications in the split, lets set the class to microcalcifications
            if('CALCIFICATION' in line.split()[1]):
                lesion_type = 'microcalcification'
                print lesion_type
            else:
                lesion_type = 'mass'
            break
        
    return lesion_type







def find_boundary(f, fpath):
    
    #dictionary for the chain code
    chain_x = {0:0,1:1,2:1,3:1,4:0,5:-1,6:-1,7:-1}
    chain_y = {0:-1,1:-1,2:0,3:1,4:1,5:1,6:0,7:-1}
    
    #read all of the lines until get to the one describing the boundary
    lesion_type = []
    line = []
    found = 0
    
    lesion_type = get_lesion_type(f)
    for line in f:
        if("BOUNDARY" in line):
            found += 1
            #the boundary data is on the line after where it says BOUNDARY
            #in the file, this next bit of code with the continue just
            #goes to the next line for getting the data
            continue
        if(found > 1):
            break
        
        
    #if we didn't find the boundary, then we probably shouldn't make a file for this one 
    if(not found):
        print("No Boundary or lesion information found for this example")
        
    else:
        #convert the line into a list, then get rid of the first and last element
        trace = line.split()
        #del trace[0]
        del trace[-1]
        #should now be a list of integers in string format
        
        #first to elements give the starting position of the trace
        boundary_x = [int(trace[0])]
        boundary_y = [int(trace[1])]
        #now loop over the boundary
        
        #the minus two is because the first two elements are contain the initial x and y position
        #the rest contains the trace of the boundary
        for jj in range(2, len(trace)):
            boundary_x.append(boundary_x[jj-2] + chain_x[int(trace[jj])])
            boundary_y.append(boundary_y[jj-2] + chain_y[int(trace[jj])])        
            
        #now find the limits of this for the bounding box
        top_x = np.min(boundary_x)
        top_y = np.min(boundary_y)
        bottom_x = np.max(boundary_x)
        bottom_y = np.max(boundary_y)
        
        #use this information to create the XML file needed
        
        create_xml(fpath, lesion_type, top_x, top_y, bottom_x, bottom_y)
        
        
        
        
        
        
        
        
        
def create_xml(fpath, lesion_type, top_x, top_y, bottom_x, bottom_y):
    
    annotation_dir = "/media/dperrin/Data/DDSM/data/Annotations"
    im_dir = "/media/dperrin/Data/DDSM/data/Images"
    fname = os.path.basename(fpath)[0:-7] + "jpg"
    if('RIGHT' in fname):
        pose = "right"
    else:
        pose = "left"
        
    #load in the file so we can get it's dimensions
    im = Image.open(os.path.join(im_dir, fname))
    
    
    
    
    
    #reading the size using the indecies back the front because it is
    #an Imgae object and not an arrow
    #
    #Why do they do this to me :(
    height = im.size[1] 
    width = im.size[0]
    
    #make sure all the bounding boxes are within reasonable ranges
    #make sure these values are ok and not outside the dimensions of the image
    if(top_x < 0):
        top_x = 0
    if(top_y < 0):
        top_y = 0
    if(bottom_x >= width):
        bottom_x = width - 2
    if(bottom_y >= height):
        bottom_y = height - 2

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
    ET.SubElement(obj, 'name').text = lesion_type
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
    
    #for now just have an example file
    path_to_ddsm = "/media/dperrin/Data/DDSM/figment.csee.usf.edu/pub/DDSM/cases/" #../data/"
    #read in the overlay file
    
    #for this test will just loop through all the files in the data folder
    for root, subFolders, file_names in os.walk(path_to_ddsm):
        for file_name in file_names:
            try:
                #if overlay is in the file name, then it is an overlay file we
                #are interested in for generating the metadata
                if('OVERLAY' in file_name):
                    fpath = os.path.join(root, file_name)
                    f = open(fpath, 'r')
                    find_boundary(f, fpath)
                    
            except:
                #raise
                pass
