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

    """
    __init__()
    
    Description:
    

    @param benign_files = boolean to tell us if we want benign or malignant scans

    """
    def __init__(self, directory = './', benign_files = True):
        
        self.metadata = pd.read_excel( directory + '/exams_metadata_pilot.xlsx')
        self.crosswalk = pd.read_excel( directory + '/images_crosswalk_pilot.xlsx')        

        self.no_patients = self.metadata.shape[0] - 1
        self.no_images = self.crosswalk.shape[0] - 1

        
        self.patient_pos = -1          #patient position in the metadata spreadsheet
        self.current_patient_id = 0
        self.exam_index = 0
        self.benign_files = benign_files
        self.no_exams = 1
        self.image_index = 0
        self.no_left_scans = 0 #number of scan for each breast
        self.no_right_scans = 0 #should be the same but I am double checking
        self.images_in_exam = 0
        self.image_view = []
        self.cancer_l = False
        self.cancer_r = False
        self.breast = 'left'
        self.left_index = 0
        self.right_index = 0
        self.no_scans_left = 0
        self.no_scans_right = 0
        self.filename_l = []  #list containing the filenames of the left and right breast scans
        self.filename_r = []


        #if(benign_files):
        #   self.get_benign()



    """
    next_scan()

    Description:
    Will load in the parameters for the next mammogram scan
    If we have reached the end of this examination (eg. done all of the left and right scans in this examination),
    will call the get benign function to get the next patients metadata


    """
    def next_scan(self):

        #see if we have gone through all of the left images_in_exam
        #if yes, then we just need to get the next scan
        #subtracting 1 to make it zero based
        print('here')
        if(self.left_index >= (self.no_scans_left - 1)) & ( self.right_index >= (self.no_scans_right -1)):
            #if we want the next benign file

            if(self.benign_files):

                #just use a loop that will keep searching until it finds a benign scan
                #meaning that both breasts arent cancerous
                #basically a do while loop
                while(True):
                    print('here')
                    self.get_benign()
                    if( (self.cancer_l == False) |  (self.cancer_r == False)):
                        print('breaking the loop')
                        #will break the loop
                        break
                    
            #if we are getting the malignant scans
            else:
                while(True):
                    self.get_malignant()
                    if(self.cancer_l |  self.cancer_r):
                        #will break the loop
                        break

        print(self.left_index)
        print(self.no_scans_left)
        
        if(self.left_index < (self.no_scans_left)):
            print('left')
            file_name = self._return_filename('left')
            self.left_index = self.left_index + 1
            
        elif( self.right_index < (self.no_scans_right) ):
            print self.right_index
            print self.filename_r
            file_name = self._return_filename('right')
            self.right_index = self.right_index + 1

        return file_name






    """
    get_benign()

    Description:
    function will load the next benign examination
    If have gone through all the scans in the exam
    
    """



        
    
    def get_benign(self):

        #lets increment our patient position counter
        self.patient_pos = self.patient_pos +1
        print('patient pos = %d' %(self.patient_pos))
        self.patient_id = int(self.metadata.iloc[self.patient_pos,0])
        self.filename_l = []
        self.filename_r = []
        self.left_index = 0
        self.right_index = 0
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
        print('cancer left = %r, cancer right = %r' %(self.cancer_l, self.cancer_right))
        #will also get number of scans etc. for current patient, current exam and each breast
        self.get_filenames()
        
        #since in this instance we want benign scans only, if the scans are malignant, will just set the number of scans for that
        #breast to zero so we dont look at it
        if(self.cancer_l):
            self.no_scans_left = 0
            #change to the right breast
            self.breast = 'right'

        if(self.cancer_r):
            self.no_scans_right = 0
        
        
        





    def get_malignant(self):
        print('getting malignant scan')








        

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

        left = (self.crosswalk['patientId'] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('L')) 

        left_filenames = (self.crosswalk.loc[left, 'filename'])
        #print(left_filenames)
        self.no_scans_left = np.sum(left)
        for ii in range(0, np.sum(left)):
            left_name = left_filenames.iloc[ii]
            self.filename_l.append(left_name)


        right = (self.crosswalk['patientId'] == self.patient_id) & (self.crosswalk['examIndex'] == self.exam_index) & (self.crosswalk["imageView"].str.contains('R')) 

        right_filenames = (self.crosswalk.loc[right, 'filename'])
        self.no_scans_right = np.sum(right)
        for ii in range(0, np.sum(right)):
            print(ii)
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
        base_dir = './pilot_images/'
        if(breast == 'left'):
            return base_dir + str(self.filename_l[self.left_index][0:-3])
        else:
            return base_dir + str(self.filename_r[self.right_index][0:-3])


        
