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
    def __init__(self, directory = './', benign_files = True, run_synapse = False):

        #if we are running on synapse, change the file paths a bit
        if(run_synapse == True):
            print(run_synapse)
            self.metadata = pd.read_csv('/exams_metadata_pilot.tsv', sep='\t')
            self.crosswalk = pd.read_csv('/images_crosswalk_pilot.tsv', sep='\t')        

        #else everything is in the source directory so dont worry
        else:
            self.metadata = pd.read_excel( directory + 'exams_metadata_pilot.xlsx')
            self.crosswalk = pd.read_excel( directory + 'images_crosswalk_pilot.xlsx')        

        self.run_synapse = run_synapse
        self.total_no_exams = self.metadata.shape[0] - 1
        self.no_images = self.crosswalk.shape[0] - 1

        
        self.exam_pos = -1          #patient position in the metadata spreadsheet
        self.current_patient_id = 0
        self.exam_index = 0
        self.benign_files = benign_files
        self.no_exams_patient = 1
        self.no_benign_scans = 0       #number of cancer free image scans we have in the sample
        self.no_malignant_scans = 0    #number of cancerous image scans
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
        
        
        self.malignant_count = 0
        self.benign_count = 0
        self._count_files()   #function that will count the number of benign and cancerous files we have in the sample

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

        if(self.left_index >= (self.no_scans_left)) & ( self.right_index >= (self.no_scans_right)):
            #if we want the next benign file

            if(self.benign_files == True):

                #just use a loop that will keep searching until it finds a benign scan
                #meaning that both breasts arent cancerous
                #basically a do while loop
                while(True):

                    self.get_benign()
                    if( (self.cancer_l == False) |  (self.cancer_r == False)):
                        #will break the loop
                        break
                    
            #if we are getting the malignant scans
            elif(self.benign_scans == False):
                while(True):
                    self.get_malignant()
                    if( (self.cancer_l == True) |  (self.cancer_r == True) ):
                        #will break the loop
                        break
                    
            #otherwise we will be loading both in
            else:
                print('Error: If you want to load any scan in, use get_either function')
                sys.exit()
                
                
                
        if(self.left_index < (self.no_scans_left)):
            file_name = self._return_filename('left')
            self.left_index = self.left_index + 1
            
        elif( self.right_index < (self.no_scans_right) ):
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
        self.exam_pos = self.exam_pos +1
        self.patient_id = int(self.metadata.iloc[self.exam_pos,0])
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
        
        ########################################
        # Check this it might not be right     #
        ########################################
        temp = self.crosswalk['patientId'] == self.patient_id
        #the maximum value from the exam index will tell us how many exams were done per patient_id
        self.no_exams_patient = np.max(self.crosswalk[temp])
        
        
        #now make sure this scan doesnt have cancer
        self.cancer_l, self.cancer_right = self.check_cancer()
        #now lets load all of the file names for this examination
        #will also get number of scans etc. for current patient, current exam and each breast
        self.get_filenames()
        print(self.filename_l)
        print(self.filename_r)
        
        
        
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
        #will keep looping through and searching until we find a scan that contains cancerous cells
        self.cancer_l = False
        self.cancer_r = False
        while( (self.cancer_l == False) & (self.cancer_r == False)):
            
            #lets increment our patient position counter
            self.exam_pos = self.exam_pos +1
            self.patient_id = int(self.metadata.iloc[self.exam_pos,0])
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

            ########################################
            # Check this it might not be right     #
            ########################################
            temp = self.crosswalk['patientId'] == self.patient_id
            #the maximum value from the exam index will tell us how many exams were done per patient_id
            self.no_exams_patient = np.max(self.crosswalk[temp])


            #now make sure this scan doesnt have cancer
            self.cancer_l, self.cancer_right = self.check_cancer()
        #now lets load all of the file names for this examination
        #will also get number of scans etc. for current patient, current exam and each breast
        self.get_filenames()
        print(self.filename_l)
        print(self.filename_r)

        #since in this instance we want benign scans only, if the scans are malignant, will just set the number of scans for that
        #breast to zero so we dont look at it
        if(self.cancer_l == False):
            self.no_scans_left = 0
            #change to the right breast
            self.breast = 'right'

        if(self.cancer_r == False):
            self.no_scans_right = 0





    """
    get_either()

    Description:
    Function will just get the next available scan, doesn't matter what type it is
    will just get the next scan
    
    """
    def get_either(self):
        #lets increment our patient position counter
        self.exam_pos = self.exam_pos +1
        self.patient_id = int(self.metadata.iloc[self.exam_pos,0])
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
        
        ########################################
        # Check this it might not be right     #
        ########################################
        temp = self.crosswalk['patientId'] == self.patient_id
        #the maximum value from the exam index will tell us how many exams were done per patient_id
        self.no_exams_patient = np.max(self.crosswalk[temp])
        
        #now make sure this scan doesnt have cancer
        self.cancer_l, self.cancer_right = self.check_cancer()
        #now lets load all of the file names for this examination
        #will also get number of scans etc. for current patient, current exam and each breast
        self.get_filenames()
        print(self.filename_l)
        print(self.filename_r)





        

    """
    check_cancer()

    Description:
    Will just read the spreadsheet to see if this current scan is a malignant case

    @param exam_pos = location of patient position in the spreadsheet
                         default to the current patient but may want to look at any patient_id

    @retval boolean value for left and right breast.
            True if malignant, False if benign

    """
    
    def check_cancer(self, exam_pos = np.nan):

        #if there wasnt a specific patient position input to check, just use the current one from the scanning procedure
        if(np.isnan(exam_pos)):
            exam_pos = self.exam_pos

        #get just the metadata of this current exam
        scan_metadata = self.metadata.iloc[exam_pos,:]
        
        #the spreadsheet will read a one if there is cancer
        return (scan_metadata['cancerL'] == 1), (scan_metadata['cancerR'] == 1)




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
    count_files()

    Description:
    Function will go through the metadata spreadsheets and count the number of cancerous and malignant scans we have
    


    """

    def _count_files(self):
        
        for n in range(1,self.total_no_exams):
            #will go through and find all of the scans per examination
            #first lets see which exam number we are looking at
            patient_id = int(self.metadata.iloc[n, 0])
            exam_index = int(self.metadata.iloc[n,1])
            
            #now will check cancer for this exam
            left, right = self.check_cancer(n)

            num_left = np.sum( (self.crosswalk['patientId'] == patient_id) & (self.crosswalk['examIndex'] == exam_index) & (self.crosswalk['imageView'].str.contains('L')) )
            num_right = np.sum( (self.crosswalk['patientId'] == patient_id) & (self.crosswalk['examIndex'] == exam_index) & (self.crosswalk['imageView'].str.contains('R')) )

            if(left == True):
                self.malignant_count = self.malignant_count + num_left
            else:
                self.benign_count = self.benign_count + num_left
            
            if(right == True):
                self.malignant_count = self.malignant_count + num_right
            else:
                self.benign_count = self.benign_count + num_right

        print('benign_count = %d' %(self.benign_count))
        print('malignant_count = %d' %(self.malignant_count))                

        #find the number of unique patients as well
        self.no_patients = self.metadata.patientId.nunique()
        print('number of unique patients = %d' %(self.no_patients))
        

        
    

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

        
        
