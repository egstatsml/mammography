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
    
    
    @param command_line_args = arguments object that holds the path and boolean values determined
                               from the command line 
    
    
    """
    
    def __init__(self, command_line_args):
        
        self.metadata = pd.read_csv(command_line_args.metadata_path + '/exams_metadata.tsv', sep='\t')
        self.crosswalk = pd.read_csv(command_line_args.metadata_path + '/images_crosswalk.tsv', sep='\t')    
        self.training_path = command_line_args.input_path
        
        
        #now setting the member variables
        self.total_no_exams = self.metadata.shape[0] - 1
        self.cancer = False     #cancer status of the current scan
        self.cancer_list = []   #list of cancer status for all of the scans
        self.filenames = [] #list that contains all of the filenames
        self.file_pos = 0   #the location of the current file we are looking for
        self.patient_subject = 'subjectId'
        
        #lets load in all of the files
        if(command_line_args.training):
            self.get_training_scans(command_line_args.input_path)
        elif(command_line_args.validation):
            self.get_validation_scans(command_line_args.input_path)
        self.no_scans = len(self.filenames)
        
        
    """
    get_training_data()
    
    Description:
    Will call the get_all_scans() function with the training parameter set, and will then 
    load in all of the files
    
    """
    
    def get_training_scans(self, directory):
        
        #call the get_all_scans function and tell it to look in the training data
        self.get_all_scans('training', directory)
        
        
    def get_validation_scans(self, directory):
        #call the get_all_scans function and tell it to look in the validation data
        self.get_all_scans('validation', directory)
        
        
    """
    get_all_scans()
    
    Description:
    Function will load in all of the filenames available for the scans and store them in a list.
    Will use member variable self.run_synapse to see if we are running on local machine or on synapse server
    
    
    @param data_type = string to say where we are looking for the data
                 'training' = look in the training directory
                 'validation' = look in the classifying directory
    """
    
    def get_all_scans(self, data_type, directory):
        
        #now lets load in all of the filenames
        for (data_dir, dirnames, filenames) in os.walk(directory):
            self.filenames.extend(filenames)
            break
        
        if(data_type == 'training'):
            #now will add the cancer status of these files
            for ii in range(0, len(self.filenames)):
                self.next_scan()
                self.cancer_list.append(self.cancer)
                
            #after done adding the cancer status, will set reset the file position back to the start (0)
            self.file_pos = 0
            
            
            
            
            
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
        #now lets check if this file has cancer
        self.check_cancer(current_file)
        
        #increment the scan position
        self.file_pos = self.file_pos + 1
        
        #now return the filename with the path
        #am including a minus one for the file position because we just incremented it
        return self.training_path + self.filenames[self.file_pos -1]
    
    
    
    def check_cancer(self, filename):
        
        #will get rid of any file extentsion suffixies
        #will be either .npy or .dcm, wither way both are 4 chars long
        filename = filename[0:-4]
        list_all_files = list(self.crosswalk['filename'])
        file_loc = []
        for ii in range(0,len(list_all_files)): 
            file_loc.append(filename in list_all_files[ii])
            
        #temp = (filename in list(self.crosswalk['filename']))
        #print 'here'
        #print temp
        crosswalk_data = self.crosswalk.loc[file_loc,:]
        #self.cancer = int(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('cancer')]) == 1
        
        #now use the helper function to actually check for cancer
        self._check_cancer(crosswalk_data)
        
        
        
        
    """
    _check_cancer()
    
    Description:
    Will just read the spreadsheet to see if this current scan is a malignant case
    
    """
    
    def _check_cancer(self, crosswalk_data):
        
        #get just the metadata of this current exam
        #by finding the row with this patient id and exam number
        #finding these individually, it's not the most elegant way to do it,
        #but it is the clearest
        patient_mask = (self.metadata[self.patient_subject] == crosswalk_data.iloc[0,0])
        exam_mask = (self.metadata['examIndex'] == crosswalk_data.iloc[0,1])
        mask = (patient_mask & exam_mask)
        scan_metadata = self.metadata.loc[mask,:]
        
        #the spreadsheet will read a one if there is cancer
        #print crosswalk_data
        #print scan_metadata
        if(str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')]) == 'L') & ((scan_metadata.iloc[0,scan_metadata.columns.get_loc('cancerL')]) == 1):
            self.cancer = True
            print('cancer')
        elif(str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')]) == 'R') & ((scan_metadata.iloc[0,scan_metadata.columns.get_loc('cancerR')]) == 1):
            self.cancer = True
            print('cancer')
        else:
            self.cancer = False
            
            
            
            
    #just creating a mask that will tell me which rows to look at in the crosswalk spreadsheet
    #to get the filenames of the scans
    
    
    
    
    def get_filenames(self):
        
        left = (self.crosswalk[self.patient_subject] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('L')) 
        
        left_filenames = (self.crosswalk.loc[left, 'filename'])
        
        self.no_scans_left = np.sum(left)
        for ii in range(0, np.sum(left)):
            left_name = left_filenames.iloc[ii]
            self.filename_l.append(left_name)
            
            
        right = (self.crosswalk[self.patient_subject] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('R')) 
        
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
        mask = (self.metadata[self.patient_subject] == patientId)
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
        mask = (self.metadata[self.patient_subject] == patientId)
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
        mask = (self.metadata[self.patient_subject] == self.current_patient_id)
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
    
    
    
    
