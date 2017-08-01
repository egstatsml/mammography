#!/usr/bin/python


import os
import sys
import numpy as np
import xml.etree.cElementTree as ET


"""
ddsm_to_xml.py

Description:
 Converts the annotated files from the DDSM database into an XML file that can be more easily handled by current RCNN examples 

"""




if __name__ == "__main__":
    
    #for now just have an example file
    
    #read in the overlay file
    f = open('test.OVERLAY', 'r')
    #dictionary for the chain code
    chain_x = {0:0,1:1,2:1,3:1,4:0,5:-1,6:-1,7:-1}
    chain_y = {0:-1,1:-1,2:0,3:1,4:1,5:1,6:0,7:-1}
    
    
    
    #read all of the lines until get to the one describing the boundary
    
    line = []
    found = False
    for line in f:
        if("BOUNDARY" in line):
            found = True
            break
        
    #if we didn't find the boundary, then we probably shouldn't make a file for this one 
    if(not found):
        print("No Boundary information found for this example")
        
    else:
        #convert the line into a list, then get rid of the first and last element
        trace = line.split()
        del trace[0]
        del trace[-1]
        #should now be a list of integers in string format
        
        #first to elements give the starting position of the trace
        boundary_x = [int(trace[0])]
        boundary_y = [int(trace[1])]
        #now loop over the boundary
        for jj in range(0, len(trace)):
            boundary_x.append(boundary_x[jj] + chain_x[trace[jj]])
            boundary_y.append(boundary_y[jj] + chain_y[trace[jj]])        
            
            
            
            
            
            






            

