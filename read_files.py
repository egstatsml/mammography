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

        
        self.patient_pos = 1          #patient position in the spreadsheet
        self.current_patient_id = 0
        self.exam_index = 0
        self.image_index = 0
        self.images_in_exam = 0
        self.image_view = []
        self.cancer_l = False
        self.cancer_r = False
        self.breast = 'left'
        self.no_scans_left = 0
        self.no_scans_right = 0



    """
    get_benign()

    Description:
    function will load the next benign image from the current exam.
    If have gone through all the scans in the exam
    
    """

    def get_benign(self):

        #check if the current exam is finished
        #if we have, will just increment the counter to the next exam
        if( (self.image_index >= self.images_in_exam) | (self.image):
            self.patient_pos = self.patient_pos +1
            self.image_index = 1
            self.exam_index = 1
            self.breast = 'left'
            


        
        #now make sure this scan doesnt have cancer
        self.cancer_l, self.cancer_right = self.check_cancer()

        
            

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
        scan_metadata = self.metadata[self.patient_pos]
        
        return (scan_metadata['cancerL'] == 0), (scan_metadata['cancerR'] == 0)
