"""
read_files.py

Author: Ethan Goan
QUT
2016


Description:

Declare object to handle reading all of the data files and the excel sheets



"""


import dicom
import os
import numpy as np
import pandas as pd





class spreadsheet(object):

    def __init__(self, directory):
    
        self.metadata = pd.read_excel( './' + directory + '/exams_metadata_pilot.xlsx')
        self.crosswalk = pd.read_excel( './' + directory + '/images_crosswalk_pilot.xlsx')        

        self.no_patients = self.metadata.shape[0] - 1
        self.no_images = self.crosswalk.shape[0] - 1

        
        self.patient_pos = 1          #patient position in the metadata spreadsheet
        self.current_patient_id = 0
        self.exam_index = 0
        self.no_exams = 1
        self.image_index = 0
        self.no_left_scans = 1 #number of scan for each breast
        self.no_right_scans = 1 #should be the same but I am double checking
        self.images_in_exam = 0
        self.image_view = []
        self.cancer_l = False
        self.cancer_r = False
        self.breast = 'left'
        self.left_index = 1
        self.right_index = 1
        self.no_scans_left = 0
        self.no_scans_right = 0
        self.filename_l = []  #list containing the filenames of the left and right breast scans
        self.filename_r = []


        
    """
    get_benign()

    Description:
    function will load the next benign image from the current exam.
    If have gone through all the scans in the exam
    
    """

    
    def get_benign(self):
        
        
        #check if the current exam is finished
        #if we have, will just increment the counter to the next exam
        if( (self.image_index > self.images_in_exam) | (self.right_index > self.no_scans_right) ):
            self.patient_pos = self.patient_pos +1
            print('patient pos = %d' %(self.patient_pos))
            print(self.metadata)
            self.patient_id = self.metadata.iloc[self.patient_pos,0]
            #print('patient id = %d' %(self.patient_id))
            self.image_index = 1
            self.exam_index = 1
            self.breast = 'left'

            #see how many exams there are
            #this will create a mask to help me access just the elements with the patient I am interested in
            print(self.patient_id)
            temp = self.crosswalk['patientId'] == self.patient_id
            self.no_exams = np.max(self.crosswalk[temp])
            
            
        #now make sure this scan doesnt have cancer
        self.cancer_l, self.cancer_right = self.check_cancer()
        #now lets load all of the file names for this examination
            
        self.get_filenames()
        
            
        #get number of scans etc. for current breast
        


        


    """
    check_cancer()

    Description:
    Will just read the spreadsheet to see if this current scan is a malignant case

    @retval boolean value for left and right breast.
            True if malignant, False if benign

    """
    
    def check_cancer(self):

        #get just the metadata of this current exam
        scan_metadata = self.metadata.iloc[self.patient_pos,:]
        
        #the spreadsheet will read a one if there is cancer
        return (scan_metadata['cancerL'] == 1), (scan_metadata['cancerR'] == 1)




    #just creating a mask that will tell me which rows to look at in the crosswalk spreadsheet
    #to get the filenames of the scans


    
    def get_filenames(self):

        if(self.cancer_l == False):
            left = (self.crosswalk['patientId'] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('L')) 
            
            left_filenames = (self.crosswalk.loc[left, 'filename'])
            for ii in range(0, np.sum(left)):
                left_name = left_filenames.iloc[ii]
                self.filename_l.append(left_name)

        
        if(self.cancer_r == False):
            right = (self.crosswalk['patientId'] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('R')) 
            
            right_filenames = (self.crosswalk.loc[left, 'filename'])
            for ii in range(0, np.sum(left)):
                right_name = right_filenames.iloc[ii]
                self.filename_r.append(right_name)
        
