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
        
        #now setting the member variables
        self.cancer = False        #cancer status of the current scan
        self.cancer_list = []      #list of cancer status for all of the scans DONE IN THIS PROCESS!!!!!
        self.laterality = []       #laterality of current scan
        self.laterality_list = []  #list of laterality of all scans
        self.exam = []             #exam index of current scan 
        self.exam_list = []        #list of exam indicies for all scans
        self.subject_id = []       #ID of current scan
        self.subject_id_list = []  #list of subject ID's for all scans
        self.bc_history = []
        self.bc_history_list = []
        self.anti_estrogen = []
        self.anti_estrogen_list = []
        self.bc_first_degree_history = []
        self.bc_first_degree_history_list = []
        self.bc_first_degree_history_50 = []
        self.bc_first_degree_history_50_list = []
        
        
        self.filenames = []        #list that contains all of the filenames
        self.file_pos = 0          #the location of the current file we are looking for
        self.sub_challenge = command_line_args.sub_challenge  #save which sub challenge we are doing
        #this is so we know whether to store metadata or not
        
        #this variable is here just because there was a minor discrepancy between the pilot metadata
        #given out early in the challenge
        self.patient_subject = 'subjectId'
        
        #load in data from the spreadsheets
        self.crosswalk = pd.read_csv(command_line_args.metadata_path + '/images_crosswalk.tsv', sep='\t')    
        self.training_path = command_line_args.input_path
        #will have access to the metadata if we are validating on the synapse servers
        #though if we are validating for ourselves we will use the validation data
        print command_line_args.validation
        print command_line_args.challenge_submission
        if( not(command_line_args.validation & command_line_args.challenge_submission) ):
            print('am getting the metadata')
            self.metadata = pd.read_csv(command_line_args.metadata_path + '/exams_metadata.tsv', sep='\t')
            print(self.metadata.columns.values)
            
        #just printing the database headers to make sure they are correct
        print(self.crosswalk.columns.values)        
        
        #lets load in the files
        #
        #If we are running the validation model on the server, we wont have access to any
        #metadata about the cancer status of this file. 
        #If we are running validation (-v) and running on synapse server for the challenge (-c)
        #we wont try and load in the cancer status, we will just set them all to 0
        #
        #
        #For every other instance (including validation on pilot data, as I wan't to get
        #performance metrics) we will load the cancer status.
        
        #if we are validatinging and on the synapse servers for the challenge
        if(command_line_args.validation & command_line_args.challenge_submission):
            self.get_validation_challenge_scans(command_line_args.input_path)        
                
        #otherwise just load all of the scans and get their cancer status
        else:
            self.get_all_scans(command_line_args.input_path)
            
        #now lets set the number of scans we have available
        self.no_scans = len(self.filenames)
        
        
        
        
        
        
    """
    get_validation_challenge_scans()
    
    Description:
    Will call the get_all_scans() function with the validation argument set, and will then 
    load in all of the files. This is done so we don't try and get the cancer status
    
    """
        
    def get_validation_challenge_scans(self, directory):
        #call the get_all_scans function and tell it to look in the validation data
        self.get_all_scans(directory, 'validation_challenge')
        
        
        
        
    """
    get_all_scans()
    
    Description:
    Function will load in all of the filenames available for the scans and store them in a list.
    
    
    @param data_type = string to say where we are looking for the data
                 'all' = default value, and we will get the cancer status as well
                 'validation_challenge' = just get the files, don't try and get the cancer status
    
    """
    
    def get_all_scans(self,directory, data_type = 'all'):
        
        #now lets load in all of the filenames
        for (data_dir, dirnames, filenames) in os.walk(directory):
            self.filenames.extend(filenames)
            break   
        
        if(data_type == 'all'):
            #now will add the cancer status of these files
            for ii in range(0, len(self.filenames)):
                self.next_scan(data_type)
                self.cancer_list.append(self.cancer)
                self.laterality_list.append(self.laterality)
                self.exam_list.append(self.exam)
                self.subject_id_list.append(self.subject_id)
                
                #add the medata features, though these will only be used
                #if we have specified we are doing sub challenge 2
                #otherwise everything is just set to zero
                self.bc_history_list.append(self.bc_history)
                self.bc_first_degree_history_list.append(self.bc_first_degree_history)
                self.bc_first_degree_history_50_list.append(self.bc_first_degree_history_50)
                self.anti_estrogen_list.append(self.anti_estrogen)
                    
                    
        #after done adding the cancer status, will set reset the file position back to the start (0)
        self.file_pos = 0
        
        
        
        
        
    """
    next_scan()
    
    Description:
    Function will just get the next available scan, doesn't matter what type it is
    will just get the next scan
    
    Will use the filename of the scan and backtrack to the metadata spredsheet to see if it is
    a cancerous scan or not
    
    @param data_type = string to tell us if we are validating or not
                     == 'all' when not validating
                     == 'validation_challenge' when validating on the servers
    
    """
    
    def next_scan(self, data_type = 'all'):
        
        #find the patient number of the current scan, the exam number ant the breast we are
        #looking at
        
        current_file = self.filenames[self.file_pos]
        #now lets get info from this scan, such as the cancer status,
        #view and exam number
        self.get_info(current_file, data_type)
        
        #increment the scan position
        self.file_pos = self.file_pos + 1
        
        #now return the filename with the path
        #am including a minus one for the file position because we just incremented it
        return self.training_path + self.filenames[self.file_pos -1]
    
    
    """
    get_info()
    
    Description:
    Will get all of the relevant data from the current scan
    
    @param filename = string containing just the code of the current image
    @param data_type = string saying whether we are training or validating
                     == 'all' when not validating
                     == 'validation_challenge' when validating on the servers
                      
    """
    
    
    
    
    def get_info(self, filename, data_type):
        
        #will get rid of any file extentsion suffixies
        #will be either .npy or .dcm, wither way both are 4 chars long
        filename = filename[0:-4]
        list_all_files = list(self.crosswalk['filename'])
        file_loc = []
        for ii in range(0,len(list_all_files)): 
            file_loc.append(filename in list_all_files[ii])
            
            
        crosswalk_data = self.crosswalk.loc[file_loc,:]
        #get the view and exam index of this scan
        self.laterality = crosswalk_data.iloc[0, crosswalk_data.columns.get_loc('laterality')]
        self.exam = crosswalk_data.iloc[0, crosswalk_data.columns.get_loc('examIndex')]
        self.subject_id = crosswalk_data.iloc[0, crosswalk_data.columns.get_loc(str(self.patient_subject))]
        
        #if we are doing sub challenge 2, lets add some metadata as features
        #if we aren't it will just set all the metadata to zero
        self.metadata_sub_challenge_2(crosswalk_data)
            
        #now use the helper function to actually check for cancer
        #but only do this if we aren't validating on the synapse servers
        if(data_type != 'validation_challenge'):
            self._check_cancer(crosswalk_data)
        #otherwise it doesn't really matter what the cancer status is,
        #but just set it to false
        else:
            self.cancer = False
            
            
            
            
            
            
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
        
        patient_mask = (self.metadata.iloc[:, self.metadata.columns.get_loc(self.patient_subject)] == crosswalk_data.iloc[0,crosswalk_data.columns.get_loc(self.patient_subject)])
        exam_mask = (self.metadata.iloc[:,self.metadata.columns.get_loc('examIndex')] == crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('examIndex')])
        mask = (patient_mask & exam_mask)
        scan_metadata = self.metadata.loc[mask,:]
        #print scan_metadata
        #print crosswalk_data
        
        #the spreadsheet will read a one if there is cancer
        #print crosswalk_data
        #print scan_metadata
        #( 'L' in str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')])) &
        #print str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')])
        #print str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')]) == 'L'
        #print scan_metadata
        #print (scan_metadata.iloc[0,scan_metadata.columns.get_loc('cancerL')])
        #print(str(scan_metadata.iloc[0,scan_metadata.columns.get_loc('cancerL')]) == '1')
        #print(scan_metadata.columns.get_loc('cancerL'))
        
        if(str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')]) == 'L') & (str(scan_metadata.iloc[0,scan_metadata.columns.get_loc('cancerL')]) == '1'):
            self.cancer = True
            print('cancer left')
            
        elif(str(crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('laterality')]) == 'R') & (str(scan_metadata.iloc[0,scan_metadata.columns.get_loc('cancerR')]) == '1'):
            self.cancer = True
            print('cancer right')
        else:
            self.cancer = False
            
            
            
            
            
            
    """
    metatdata_sub_challenge_2()
    
    Description:
    If we are taking part in sub challene 2, will get the metadata as well
    
    I copied most of the code from the _check_cancer function above ;)
    which is sloppy but I will let it slide this time
    
    """
    
    def metadata_sub_challenge_2(self, crosswalk_data):
        
        if(self.sub_challenge == 2):
            patient_mask = (self.metadata.iloc[:, self.metadata.columns.get_loc(self.patient_subject)] == crosswalk_data.iloc[0,crosswalk_data.columns.get_loc(self.patient_subject)])
            exam_mask = (self.metadata.iloc[:,self.metadata.columns.get_loc('examIndex')] == crosswalk_data.iloc[0,crosswalk_data.columns.get_loc('examIndex')])
            mask = (patient_mask & exam_mask)
            scan_metadata = self.metadata.loc[mask,:]
            self.bc_history = int(scan_metadata.iloc[0,scan_metadata.columns.get_loc('bcHistory')])
            self.anti_estrogen = int(scan_metadata.iloc[0,scan_metadata.columns.get_loc('antiestrogen')])
            self.bc_first_degree_history = int(scan_metadata.iloc[0,scan_metadata.columns.get_loc('firstDegreeWithBc')])
            self.bc_first_degree_history_50 = int(scan_metadata.iloc[0,scan_metadata.columns.get_loc('firstDegreeWithBc50')])
            
            
        else:
            self.bc_history = 0
            self.anti_estrogen = 0
            self.bc_first_degree_history = 0
            self.bc_first_degree_history_50 = 0
            
            
            
            
            
            
            
            
            
            
            
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
    
    
    
    
