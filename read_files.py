#!/bin/env python
"""
read_files.py

Author: Ethan Goan
Queensland University of Technology
DREAM Mammography Challenge
2016

Description:

spreadsheet object is defined to handle the reading of image file names
and the associated metadata
spreadsheet object is capable of reading malignant or benign scans individually
for training purposes


"""
import sys
import dicom
import os
import numpy as np
import pandas as pd





class spreadsheet(object):

    """
    __init__()
    
    Description:
    
    
    @param benign_files = boolean to tell us if we want benign or malignant scans
    @param run_synapse = boolean to let us know is we are running on synapse server or
                         just testing on my computer. Will change the paths for accessing the data
                         to suit synapse servers if we are
    
    """
    def __init__(self, directory = './', training = True, run_synapse = False):
        
        #if we are running on synapse, change the file paths a bit
        if(run_synapse == True):
            print(run_synapse)
            self.metadata = pd.read_csv('/exams_metadata_pilot.tsv', sep='\t')
            self.crosswalk = pd.read_csv('/images_crosswalk_pilot.tsv', sep='\t')        
            self.training_path = '/trainingData/'
            #else everything is in the source directory so dont worry
        else:
            self.metadata = pd.read_excel( directory + 'exams_metadata_pilot.xlsx')
            self.crosswalk = pd.read_excel( directory + 'images_crosswalk_pilot.xlsx')        
            self.training_path = './pilot_images/'
            
        #now setting the member variables
        self.run_synapse = run_synapse  #save whether we are running on the synapse servers
        self.total_no_exams = self.metadata.shape[0] - 1
        self.no_images = self.crosswalk.shape[0] - 1
        self.run_synapse = run_synapse
        self.cancer = False     #cancer status of the current scan
        self.cancer_list = []   #list of cancer status for all of the scans
        self.filenames = [] #list that contains all of the filenames
        self.file_pos = 0   #the location of the current file we are looking for
        
        #lets load in all of the files
        if(training):
            self.get_training_scans()
        else:
            print('Have only implemented training so far')
            
        self.no_scans = len(self.filenames)
        
    """
    get_training_data()
    
    Description:
    Will call the get_all_scans() function with the training parameter set, and will then 
    load in all of the files
    
    """
    
    def get_training_scans(self):
        
        #call the get_all_scans function and tell it to look in the training data
        self.get_all_scans('training')
        
        
        
    """
    get_all_scans()
    
    Description:
    Function will load in all of the filenames available for the scans and store them in a list.
    Will use member variable self.run_synapse to see if we are running on local machine or on synapse server
    
    
    @param directory = string to say where we are looking for the data
                 'training' = look in the training directory
                 'classifying' = look in the classifying directory
    """
    
    def get_all_scans(self, directory):
        
        
        if(directory == 'training'):
            #now lets load in all of the filenames
            for (data_dir, dirnames, filenames) in os.walk(self.training_path):
                self.filenames.extend(filenames)
                break
            
            #now will add the cancer status of these files
            for ii in range(0, len(self.filenames)):
                self.next_scan()
                self.cancer_list.append(self.cancer)
                
            #after done adding the cancer status, will set reset the file position back to the start (0)
            self.file_pos = 0
            
        else:
            print('have only implemented training this far')
                
                
                
                
                
    """
    next_scan()
    
    Description:
    Function will just get the next available scan, doesn't matter what type it is
    will just get the next scan
    
    Will use the filename of the scan and backtrack to the metadata spredsheet to see if it is
    a cancerous scan or not
    
    """
    
    def next_scan(self):
        
        #find the patient number of the current scan, the exam number ant the breast we are
        #looking at
        
        current_file = self.filenames[self.file_pos]
        #if we arent on the synapse server, will need to add the .gz suffix, which is used on
        #the metadata and crosswalks spreadsheets
        if(self.run_synapse == False):
            current_file = current_file + '.gz'
            
        temp = (self.crosswalk['filename'] == current_file)
        crosswalk_data = self.crosswalk.loc[temp,:]
        
        #now lets check if this file has cancer
        self.check_cancer(crosswalk_data)
        
        #increment the scan position
        self.file_pos = self.file_pos + 1
        
        #now return the filename with the path
        #am including a minus one for the file position because we just incremented it
        return self.training_path + self.filenames[self.file_pos -1]
    
    
    
    """
    check_cancer()
    
    Description:
    Will just read the spreadsheet to see if this current scan is a malignant case

    """
    
    def check_cancer(self, crosswalk_data):

        #get just the metadata of this current exam
        #by finding the row with this patient id and exam number
        #finding these individually, it's not the most elegant way to do it,
        #but it is the clearest
        patient_mask = (self.metadata['patientId'] == crosswalk_data.iloc[0,0])
        exam_mask = (self.metadata['examIndex'] == crosswalk_data.iloc[0,1])
        mask = (patient_mask & exam_mask)
        
        scan_metadata = self.metadata.loc[mask,:]
        #the spreadsheet will read a one if there is cancer
        if(crosswalk_data.iloc[0,2] == 'L') & (scan_metadata.iloc[0,3] == 1):
            self.cancer = True
        elif(crosswalk_data.iloc[0,2] == 'R') & (scan_metadata.iloc[0,4] == 1):
            self.cancer = True
        else:
            self.cancer = False
            




    #just creating a mask that will tell me which rows to look at in the crosswalk spreadsheet
    #to get the filenames of the scans
    



    
    
    def get_filenames(self):
        
        left = (self.crosswalk['patientId'] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('L')) 
        
        left_filenames = (self.crosswalk.loc[left, 'filename'])
        
        self.no_scans_left = np.sum(left)
        for ii in range(0, np.sum(left)):
            left_name = left_filenames.iloc[ii]
            self.filename_l.append(left_name)
            
            
        right = (self.crosswalk['patientId'] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('R')) 
        
        right_filenames = (self.crosswalk.loc[right, 'filename'])
        self.no_scans_right = np.sum(right)
        for ii in range(0, np.sum(right)):
            right_name = right_filenames.iloc[ii]
            self.filename_r.append(right_name)
        
        
        
        
        
    """
    return_filename()
    
    Description:
    Function will return the filename of the current scan we are looking at
   
    @param breast = string containg left or right
    @param file_index = index to say which fielename in the list we are looking at
    
    @retval string containing current path to scan we want to look at
    
    """
    def _return_filename(self, breast):
        
        if(self.run_synapse == True):
            base_dir = '/trainingData'
        else:
            base_dir = './pilot_images/'
            
        if(breast == 'left'):
            return base_dir + str(self.filename_l[self.left_index][0:-3])
        else:
            return base_dir + str(self.filename_r[self.right_index][0:-3])
        
        
        
        
    def filenames(self):
        base_dir = './pilot_images/'
        return  base_dir + str(self.filename_l[self.left_index][0:-3]), base_dir + str(self.filename_r[self.left_index][0:-3])
    
    
    
    
    
    
    
    
    #####################################################################
    # Functions that were used by the gui for ENB345
    #
    #####################################################################    
    
    """
    get_most_recent()
    
    Description:
    Function will load in the most recent scan from a patient
    
    @param patientId = patient number
    
    """
    
    def get_most_recent(self, patientId):
        #create a truncated database that just includes the information for the
        #current patient
        
        #just clearing the filename variables in case they have something already in them
        self.filename_l = []
        self.filename_r = []
        self.patient_id = patientId
        mask = (self.metadata['patientId'] == patientId)
        patient_data = self.metadata.loc[mask,:]
        #now lets find the most recent exam and get those files
        self.exam_index = np.max(patient_data['examIndex'])
        
        #setting the number of exams
        self.no_exams_patient = self.exam_index
        
        #now get the filenames for this exam
        self.get_filenames()
        
        
        
        
              
        
    
    """
    get_exam()
    
    Description:
    Function will load in the specific scans from a signle examination
    
    @param patientId = patient number
    @param exam = exam index
    
    """
    
    def get_exam(self, patientId, exam):
        #create a truncated database that just includes the information for the
        #current patient
        
        #just clearing the filename variables in case they have something already in them
        self.filename_l = []
        self.filename_r = []
        self.patient_id = patientId
        mask = (self.metadata['patientId'] == patientId)
        patient_data = self.metadata.loc[mask,:]
        #now lets find the most recent exam and get those files
        self.exam_index = exam
        
        #now get the filenames for this exam
        self.get_filenames()
        
        
        
    """
    get_patient_data()
    
    Description:
    Function that will return a Pandas database of the current patients
    information listed in the metadata spreadsheet
    Will return all of the information form all examinations
    
    The current patient ID is read from the current_patient_id member variable
    
    Can be used then for viewing in the GUI
    
    @retval Pandas Database of the current patient we are interested in
    
    """
    
    def get_patient_info(self):
        #creating a mask for crop the data to information for just our patient of interest
        mask = (self.metadata['patientId'] == self.current_patient_id)
        #get the header
        header = self.metadata.columns.values
        patient_info = pd.DataFrame(header) 
        patient_info = patient_info.T #transpose so it is row not a column
        data = self.metadata.loc[mask,:]
        data_values = pd.DataFrame(data.values)
        #patient_info.append(data_values, ignore_index=True)
        patient_info = pd.concat([patient_info,data_values], ignore_index=True)
        #now return the cropped database
        return patient_info
    


