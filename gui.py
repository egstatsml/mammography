"""
widget class to handle the gui stuff

 
TODO LIST:

put left and right arrow figures on the buttons to jump left and right


days since previous scans

make all of the displed indecies start from 1, radiologists will like this more

"""
import dicom
import os
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
import time

import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QTimer
from PyQt4.QtGui import *
import sip

#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet
from information_window import *





#will be a member class for the window 
class view_scan(QtGui.QWidget):

    def __init__(self):

        #initialise the GUI Widget
        QtGui.QWidget.__init__(self)

        #setting all the member variables
        #member buttons for the viewing screens
        self.information_btn = QtGui.QPushButton('Information')
        self.information_btn.clicked.connect(self.display_information)
        self.second_btn = QtGui.QPushButton('Second')
        self.second_btn.clicked.connect(self.load_second)
        self.features_btn = QtGui.QPushButton('Features')
        self.features_btn.clicked.connect(self.show_features)
        self.notes_btn = QtGui.QPushButton('Notes')
        self.notes_btn.clicked.connect(self.display_notes)
        self.asymmetry_btn = QtGui.QPushButton('Asymmetry')
        self.asymmetry_btn.clicked.connect(self.load_asymmetry)
        #buttons that allow  rotating the scans
        #initialising text box and list Widgets
        self.text = QtGui.QLineEdit('enter text')
        self.listw = QtGui.QListWidget()
        self.track_btn =QtGui.QPushButton('Track')
        self.track_btn.clicked.connect(self.track_image)

        #Window that will pop up and display information of the patient 
        self.information_window = information()
        #window for saving notes 
        self.notes_window = notes()
        #classifier window
        self.classifier_window = classifier_info()
        
        
        #image view widgets
        self.im_l = mammogram_view()
        self.im_r = mammogram_view()

        #boolean variables to say if we have initialised and displayed the image views yet
        self._displayed_l = False
        self._displayed_r = False

        #dictionary that is used to convert index in the breast dropdown menu to a string
        self._breast_dict = {0:'left', 1:'right'}
        
        
        #overlay for the waiting indicator
        self.loading = QtGui.QProgressBar(self)
        self.loading.hide()
        
        #boolean variable that will say if the right image is being used or not
        #might change the way this is done later on though
        self._right_used = False
        #hiding the histogram and other stuff
        self._hide_imageview_extras()
        
                
        #member variables
        self._data = []
        #create member variable that will hold the features of a patients scan
        self._patient_data_l = []#feature(levels = 3, wavelet_type = 'haar', no_images = 1) for ii in range(6)]
        self._patient_data_r = []#feature(levels = 3, wavelet_type = 'haar', no_images = 1) for ii in range(6)]
        self._descriptor = spreadsheet(benign_files=None, run_synapse=False)
        #dictonary that will hold a descriptor and layout id number
        self._load_scan_dict = {'dimensions_btn' : 0, 'second_btn' : 1,
                                'previous_btn' : 2, 'artifacts_btn': 3, 'close_btn' : 4,
                                'im_l': 5, 'im_r': 6}
         
        


        


        



    ######################################################################
    #
    #                        Button Functions
    #
    ######################################################################

    """
    menu_buttons()

    Description:
    Function will setup menu button locations


    """

    
    def menu_buttons(self):
        
        self.layout.addWidget(self.information_btn, 0, 0,1,2) 
        self.layout.addWidget(self.second_btn, 0, 2,1,2)  
        self.layout.addWidget(self.asymmetry_btn, 0, 4,1,2) 
        self.layout.addWidget(self.features_btn, 0, 6,1,2)    
        self.layout.addWidget(self.notes_btn, 0, 8,1,2)
        
        
    

    """
    begin_viewing()

    Description:
    Function first called when we start viewing

    """

    def begin_viewing(self):
        
        #will load in the most recent scan
        self.load_most_recent_scan()

        #make sure the screen is maximised
        self.showMaximized()
        #now setup the signals for the mammogram_view objects to activate
        #reloading when the image view is changed
        self.setup_signals(self.im_l, 'left')
        self.setup_signals(self.im_r, 'right')


        
    """
    setup_signals()

    Description:
    Function will connect the signals for when a drop down menu changes to the functions'
    to reload the image data

    @param mammogram = the mammogram_view object we are connecting the signals from
    @param location = string saying if it is the left or right hand view

    """ 

    def setup_signals(self, mammogram, location):
        
        if(location == 'left'):
            mammogram.breast_btn.currentIndexChanged.connect(self.change_view_l)
            mammogram.exam_index_btn.currentIndexChanged.connect(self.change_view_l)
        elif(location == 'right'):
            mammogram.breast_btn.currentIndexChanged.connect(self.change_view_r)
            mammogram.exam_index_btn.currentIndexChanged.connect(self.change_view_r)
        else:
            print('Invalid location in setup_signals(). Must be either left or right')
            sys.exit()
            

        

    """
    load_most_recent_scan()

    Description:
    Function will get the filenames of the most recent scan and send them to the load scan function to be
    loaded in and displayed

    Default Parameters

    @param im_location = string saying whether or not is the right or left view
    @param breast = string having left or right to say scans from which breast 
                    to view
    

    """
    def load_most_recent_scan(self, im_location = 'left', breast = 'left'):
        self._descriptor.get_most_recent(self._descriptor.current_patient_id)
        #now load the data in
        self.load_scan(im_location, breast)
        self.im_l.exam_index_btn.setCurrentIndex(self.im_l.exam_index_btn.count()-1)


        


        
    """
    load_scan()
    
    Description:
    Function is called when we are redy to show some mammogram scans
    Will load up the data and then get it ready to ready to show
    
    Also initialise the layout for the GUI design
    Patient ID has been saved in the member class _descriptor


    Default parameters

    @param im_location = string saying whether or not is the right or left view
    @param breast = string having left or right to say scans from which breast 
                    to view

    """

    
    def load_scan(self, im_location = 'left', breast = 'left'):

        self.begin_loading()

        #saving the files that we want to view into the filenames variable
        if(breast == 'left'):
            filenames = self._descriptor.filename_l
        elif(breast == 'right'):
            filenames = self._descriptor.filename_r
        #if breast is equal to anything else, we should throw an error
        else:
            print('Invalid breast for mammogram view. Must be either left or right')
            sys.exit()
            
        #now we begin to load in the scan for either the left or right view
        #initialise the _patient data class for each image
        #if we arer setting up the left image view        
        if(im_location == 'left'):
            print(filenames)
            self.im_l.initialise(filenames, im_location)
            print(np.shape(self.im_l._features[0].original_scan))
            self.im_l._breast_loc = breast
            self.im_l.set_im()
            self.im_buttons(self.im_l)
            #now will add the mammogram widget if we need to
            if(self._displayed_l == False):
                self.add_mammogram(self.im_l)
                self._displayed_l = True
        

        #if we are setting the right image
        elif(im_location == 'right'):
            print(filenames)
            self.im_r.initialise(filenames, im_location)
            self.im_r._breast_loc = breast
            self.im_r.set_im()
            self.im_buttons(self.im_r)
            #now will add the mammogram widget if we need to
            if(self._displayed_r == False):
                self.add_mammogram(self.im_r)
                self._displayed_r = True
            
            
        #if location doesn't equal left or right, will print a message and throw an error
        else:
            print('Error in Load Scan: Invalid location for mammogram view. Must be either left or right')
            sys.exit()

        #now am done with the loading bar, so lets hide it
        self.hide_loading()
        #make sure all of the widgets are stretched evenly
        self.set_stretch()

            


    """
    add_mammogram()

    Description:
    function to add a mammogram view to the window
    will call to get the location of all of the buttons associated with
    a mammogram view

    """

    def add_mammogram(self, mammogram):

        #getting the y position, x pos, height and width of the widgets
        
        im_y, im_x, im_h, im_w = mammogram.get_pos(mammogram._im_layout)
        l_y, l_x, l_h, l_w = mammogram.get_pos(mammogram._left_btn_layout)
        r_y, r_x, r_h, r_w = mammogram.get_pos(mammogram._right_btn_layout)
        in_y, in_x, in_h, in_w = mammogram.get_pos(mammogram._index_disp_layout)
        ex_d_y, ex_d_x, ex_d_h, ex_d_w = mammogram.get_pos(mammogram._exam_index_disp_layout)
        ex_y, ex_x, ex_h, ex_w = mammogram.get_pos(mammogram._exam_index_layout)
        b_y, b_x, b_h, b_w = mammogram.get_pos(mammogram._breast_dropbox_layout)
        b_d_y, b_d_x, b_d_h, b_d_w =  mammogram.get_pos(mammogram._breast_disp_layout)
        rot_y, rot_x, rot_h, rot_w = mammogram.get_pos(mammogram._rotate_btn_layout)
        

        #now add all of these widgets
        self.layout.addWidget(mammogram, im_y, im_x, im_h, im_w)
        self.layout.addWidget(mammogram.jump_left_btn, l_y, l_x, l_h, l_w)
        self.layout.addWidget(mammogram.jump_right_btn, r_y, r_x, r_h, r_w)
        self.layout.addWidget(mammogram.index_disp, in_y, in_x, in_h, in_w)
        self.layout.addWidget(mammogram.rotate_btn, rot_y, rot_x, rot_h, rot_w)
        self.layout.addWidget(mammogram.exam_index_disp, ex_d_y, ex_d_x, ex_d_h, ex_d_w)
        self.layout.addWidget(mammogram.exam_index_btn, ex_y, ex_x, ex_h, ex_w)
        self.layout.addWidget(mammogram.exam_index_disp, ex_d_y, ex_d_x, ex_d_h, ex_d_w)
        self.layout.addWidget(mammogram.breast_btn, b_y, b_x, b_h, b_w)
        self.layout.addWidget(mammogram.breast_disp, b_d_y, b_d_x, b_d_h, b_d_w)




    """
    im_buttons()

    Description:
    Function is called to set the data for all of the drop down menus and everything
    Will first clear them to makse sure there isn't any old data in there

    Then will find the number of exams and set the breast dropdown menu

    Will also set the additional labels


    @param mammogram = the mammogram_view object whose details we are setting
    
    """
    def im_buttons(self, mammogram):
        
        mammogram.index_disp.setText('Current Scan: %d' %(mammogram.currentIndex + 1))
        mammogram.exam_index_disp.setText('Exam Number')
        mammogram.breast_disp.setText('Breast')
        
        
        #now make sure the drop boxes are cleared
        mammogram.exam_index_btn.clear()
        mammogram.breast_btn.clear()
        #add all of the exam numbers        
        for ii in range(1, self._descriptor.no_exams_patient +1):  
            mammogram.exam_index_btn.addItem( ('%d' %ii) )
        #adding the labels for the breast scan as well
        print('printing breast views')
        mammogram.breast_btn.addItems(['left', 'right'])

        self._breast_disp_layout = []
        self._breast_dropbox_layout = []
        



    """
    begin_loading()

    Description:
    Will show the loading screen bar in the correct position
    on the current layout


    """

    def begin_loading(self):
        #will want to add the loading bar to the correct position, which is the bottom left corner
        #will get the dimensions of the grid layout to do this
        n_cols = self.layout.columnCount()
        n_rows = self.layout.rowCount()
        print('n_cols = %d' %n_cols)
        print('n_rows = %d' %n_rows)
        self.layout.addWidget(self.loading,12,n_cols-3,1,2)
        self.loading_widget_index = self.layout.count() #should be the last widget
        
        #now need to show it
        self.loading.show()

        
    """
    hide_loading()

    Description:
    Done loading whatever we were loading, so lets hide the loading bar

    """


    def hide_loading(self):
        #first lets remove the loading bar from the layout
        self.layout.removeWidget(self.loading)
        #lets hide it as well
        self.loading.hide()
        #set it back to zero
        self.loading.reset()
            
        
        


        

    """
    clear_mammogram_x()

    Description:

    Wrapper function that will clear the mammography view, which is handy when we need to load in new mammograms
    when the user selects a different view/scan/exam

    Must remove all of the widgets to successfully clear everything
    clear_mammogram function is called to handle this

    @param mammogram = a mammogram_view object that we want to clear
    @location = the location of the mammogram_view object, either left or right


    """

    
    def clear_mammogram_l(self):
        
        self.im_l._features[:] = []
    

        
    def clear_mammogram_r(self):
        
        self.im_r._features[:] = []
        

        

    def clear_mammogram(self, mamm):

        #remove all of the widgets from the layout
        self.layout.removeWidget(mamm)
        self.layout.removeWidget(mamm.breast_btn)
        self.layout.removeWidget(mamm.rotate_btn)
        self.layout.removeWidget(mamm.exam_index_btn)
        self.layout.removeWidget(mamm.exam_index_disp)
        self.layout.removeWidget(mamm.breast_disp)
        self.layout.removeWidget(mamm.jump_left_btn)
        self.layout.removeWidget(mamm.jump_right_btn)        

        #now delete them all
        sip.delete(mamm.breast_btn)
        sip.delete(mamm.rotate_btn)
        sip.delete(mamm.exam_index_btn)
        sip.delete(mamm.exam_index_disp)
        sip.delete(mamm.breast_disp)
        sip.delete(mamm.jump_left_btn)
        sip.delete(mamm.jump_right_btn)   
        sip.delete(mamm)

        
        
        #set them all equal to non is the final step
        mamm.breast_btn = None
        mamm.rotate_btn = None
        mamm.exam_index_btn = None
        mamm.exam_index_disp = None
        mamm.breast_disp = None
        mamm.jump_left_btn = None
        mamm.jump_right_btn = None
        mamm = None
        
        



    
    """
    change_view_x()

    Description:
    Function is connected to currentIndexChanged signal in the mammogram breast dropdown menu.
    Also copnnected to view currentIndexChanged in the mammogram exam index dropdown menu.
    When the breast view is changed, this function is called to load in the filenames for the new scans
    and then load in the new scans
    
    The x above will be either l for left or r for right

    There is one wrapper function for each scan in the view, ie. left and right view

    """
    
    def change_view_l(self):

        self.block('left')
        exam_index = self.im_l.exam_index_btn.currentIndex()
        breast_index = self.im_l.breast_btn.currentIndex()
        self._descriptor.get_exam(self._descriptor.current_patient_id, self.im_l.exam_index_btn.currentIndex() + 1)
        #now load in that scan
        self.im_l._breast_loc = self._breast_dict[self.im_l.breast_btn.currentIndex()]

        #Now we need to clear the current image features stored for this view,
        #so we dont append too many images
        self.clear_mammogram_l()
        self.load_scan(im_location = 'left', breast = self.im_l._breast_loc)

        #set the indicies to what the were
        self.im_l.exam_index_btn.setCurrentIndex(exam_index)
        self.im_l.breast_btn.setCurrentIndex(breast_index)

        
        self.unblock('left')


        ### Same as above function, just for right view


        
    def change_view_r(self):

        #saving the exam index and the breast view index
        self.block('right')
        exam_index = self.im_r.exam_index_btn.currentIndex()
        breast_index = self.im_r.breast_btn.currentIndex()

        #will get the specific exams from this index
        self._descriptor.get_exam(self._descriptor.current_patient_id, self.im_r.exam_index_btn.currentIndex() + 1)
        #now load in that scan
        self.im_r._breast_loc = self._breast_dict[self.im_r.breast_btn.currentIndex()]
        
        #Now we need to clear the current image features stored for this view,
        #so we dont append too many images
        self.clear_mammogram_r()
        self.load_scan(im_location = 'right', breast = self.im_r._breast_loc)

        #set the indicies to what the were
        self.im_r.exam_index_btn.setCurrentIndex(exam_index)
        self.im_r.breast_btn.setCurrentIndex(breast_index)
        #now connect the signals for the changing the view again
        self.unblock('right')






    """
    block()

    Description:
    Function that will block an mammogram view from outputting any
    event signals
    
    @param side = string saying left or right image

    
    """

    def block(self, side):
        if(side == 'left'):
            self.im_l.breast_btn.blockSignals(True)
            self.im_l.exam_index_btn.blockSignals(True)        
        #otherwise will be right side
        else:
            self.im_r.breast_btn.blockSignals(True)
            self.im_r.exam_index_btn.blockSignals(True)        



    """
    unblock()

    Description:
    Function that will block an mammogram view from outputting any
    event signals
    
    @param side = string saying left or right image

    """

    def unblock(self, side):
        if(side == 'left'):
            self.im_l.breast_btn.blockSignals(False)
            self.im_l.exam_index_btn.blockSignals(False)        
        #otherwise will be right side
        else:
            self.im_r.breast_btn.blockSignals(False)
            self.im_r.exam_index_btn.blockSignals(False)        






    """
    set_stretch()

    Description:
    Function will make sure all of the widgets are set to zero stretch

    """
    def set_stretch(self):
        #the default stretch value
        stretch = 0
        #iterate over all columns
        for ii in range(0, self.layout.columnCount()):
            self.layout.setColumnStretch(ii, 0)
    
        for ii in range(0, self.layout.rowCount()):
            self.layout.setRowStretch(ii, 0)



    ################################################################
    #
    #        Functions connected to PYQT signals
    #
    ################################################################        

    ##############
    #
    # Menu Button Signal Functions
    #
    ##############



    """
    load_second()
    
    Description:
    Function will load in the data from the right image and place it next to the other scan of
    the breast
    
    Will immediately jump to the next frame available

    """
    def load_second(self):

        #temporarily block signals
        self.block('right')
        print(self.im_l._breast_loc)
        self.load_right(self.im_l._breast_loc)

        #make sure we arent looking at the same frame twice
        #will only happen if the current index is zero
        if(self.im_l.currentIndex == 0):
            self.im_r.jump_right()
        else:
            self.im_r.jump_left()

        
        self.im_r.breast_btn.setCurrentIndex(1)
            
        #make sure the colourmap is set to default of grey
        self.im_r.ui.histogram.gradient.loadPreset('grey')
        #make sure the signals are unblocked again
        self.unblock('right')
        
        
        
        
        
    """
    load_asymmetry()
        
    Description:
    Function is connected to the Asymmetry menu button
    Will load in the data for the asymmetry breast.
    
    
    """
    
    def load_asymmetry(self):
        
        #temporarily block signals
        self.block('right')
        #if the left image view is currently displaying the left breast, lets show the right
        #breast in the right screen
        print(self.im_l._breast_loc)
        if(self.im_l._breast_loc == 'left'):
            self.load_right(breast_loc = 'right')
            self.im_r.breast_btn.setCurrentIndex(1)
        else:
            self.load_right(breast_loc = 'left')
            self.im_r.breast_btn.setCurrentIndex(0)
        
        #make sure the colourmap is set to default of grey
        self.im_r.ui.histogram.gradient.loadPreset('grey')
        #make sure the signals are unblocked again
        self.unblock('right')




        


    """
    show_features()
    
    Description:
    Connected to the clicked signal for the features button in the GUI
    
    Function will perform the feature enhancement done in the feature class
    and will show the enhanced image on the image display

    The features that will be shown will be a multiplication of the detail components in the
    first level of wavelet decomposition.

    This will enhance the visibility of microcalcifications

    #set a custom colour gradient to make them as visible as posible

    """
    
    def show_features(self):

        #temporarily block signals
        self.block('right')

        #now lets load the right image
        self.load_right()

        
        for ii in range(0, len(self.im_l._features)):
            #first need to perform preprocessing to remove artifacts
            
            self.im_r._features[ii].preprocessing()
            #now extract the features by performing the wavelet packet decomposition
            self.im_r._features[ii].get_features()

        #now lets set the image data to the multiplication of the horizontal with the vertical
        self.im_r.set_im('wave')

        #lets set the colormap
        self.im_r.ui.histogram.gradient.restoreState({'ticks': [(0.005, (0, 10, 185, 50)), (0.05, (0, 50, 10, 255)), (0.08, (0, 100, 10, 255)), (0.09, (20, 100, 10, 255)), (0.15, (20, 80, 0, 255)), (0.2, (100, 0, 0, 255)),(0.3, (150, 0, 0, 255)), (0.00, (0, 0, 0, 255)), (np.nan, (0, 0, 0, 0))], 'mode': 'rgb'})

        
        #now lets unblock the signals
        self.unblock('right')
        
        #lets open the classifier information window
        self.classifier_window.show()
        



        

    """
    load_right()

    Description:
    Helper function that will load in data for the right image

    Before calling this function, would most likely want to block the signals
    for changing image parameters as to not cause an unwanted event


    """
    def load_right(self, breast_loc = None):

        self.im_r._features = []
        #if breast location isnt defined, lets just set it to the same as the left
        if(breast_loc == None):
            breast_loc = self.im_l._breast_loc
            
        print(self.im_l._breast_loc)
        #set the breast button to the same as the left view
        self.im_r._breast_loc = breast_loc
        self.load_scan(im_location = 'right', breast = breast_loc)

        self.im_r.breast_btn.setCurrentIndex(self.im_l.breast_btn.currentIndex() )
        self.im_r.exam_index_btn.setCurrentIndex(self.im_l.exam_index_btn.currentIndex() )




    
    def display_information(self):
        #lets get the data of the patient we are looking for
        patient_info = self._descriptor.get_patient_info()
        self.information_window.table.set_data(patient_info)
        self.information_window.view.setModel(self.information_window.table)
        self.information_window.view.show()
        self.information_window.view.setWindowTitle('Mammography-E - Patient Information')
        self.information_window.view.resize(1600,300)

        





    def display_notes(self):
        self.notes_window.show()


        

    def track_image(self):
        print('TODO: add functionality')


                

        #code to get the viewing area of an image
        #a = self.im.getView()
        #print(a.viewRange())
        
        
        
        
    def rotate_r(self):
        self._im_r_data = np.rot90(self._im_data)
        self.im_r.setImage(self._im_r_data)
        
    def _hide_imageview_extras(self):
        
        self.im_l.ui.roiBtn.hide()
        self.im_l.ui.menuBtn.hide()
        self.im_l.ui.histogram.hide()
        
        
        self.im_r.ui.roiBtn.hide()
        self.im_r.ui.menuBtn.hide()
        self.im_r.ui.histogram.hide()    
        

        
    def clear_window(self):
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)





"""
mammogram_view

will build on the ImageView class to add buttons for scanning and rotating and stuff


Default Parameters for init

@param filenames = None
@param location = None
@param breast = string to say if this is the left or right breast

"""

class mammogram_view(pg.ImageView):

    def __init__(self, filenames = None, location = None, breast = 'left'):
        
        pg.ImageView.__init__(self, view = pg.PlotItem())
        self.rotate_btn = QtGui.QPushButton('Rotate')
        self.rotate_btn.clicked.connect(self._rotate)
        self.jump_left_btn = QtGui.QPushButton('Previous')
        self.jump_left_btn.clicked.connect(self.jump_left)
        self.jump_right_btn = QtGui.QPushButton('Next')
        self.jump_right_btn.clicked.connect(self.jump_right)
        self.index_disp = QtGui.QLabel()
        self.exam_index_btn = QtGui.QComboBox(self)
        self.exam_index_disp = QtGui.QLabel()
        self.breast_btn = QtGui.QComboBox(self)
        self.breast_disp = QtGui.QLabel()

        #want to set the size policy to ignore the size hint, as it messes with my stuff a bit
        self.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        
        self._im_data = []            #array that will have the image data stored in here
        self._features = []           #will be list of features class that will hold the features of individual scans
        self._filenames = []          #list of filenames for this breast on this scan
        self._breast_loc = breast     #location of the breast, if it is left or right
        self._im_data = []            #array that will hold the data that is being displayed
        self._im_index = 1            #image view index 

        #member variables that will hold the data for the location of images and buttons for the layout
        self._location = []   #will be a string saying left or right, to decide where to display the im in the WINDOW
        self._im_pos = []
        self._rotate_pos = []
        self._jump_left_pos = []
        self._jump_right_pos = []

        #layout member variables that will hold the position of all of the buttons and images
        self._im_layout = []
        self._left_btn_layout = []
        self._right_btn_layout = []
        self._rotate_layout = []
        self._index_disp_layout = []
        self._exam_index_layout = []
        self._exam_index_disp_layout = []
        self._breast_disp_layout = []
        self._breast_dropbox_layout = []
        
        
        #if filenames and location have been supplied, will run the initialise function
        #to load in the data and set everything up
        if( (filenames != None) & (location != None) ):
            self.initialise(filenames, location)
          
        #now just some error checking

        #if the only the filename or the location has been supplied, throw an error
        #to remind me to supply both
        elif(( (filenames == None) | (location == None) ) & ( (filenames != None) | (location != None) )):
            #throw an error
            print('Need to supply both filenames and location to the init function if you want to initialise from constructor')
            sys.exit()








            
    """
    initialise()

    Description:
    Function will initialise the scan by loading in the data to view from the filenames
    and record the location of the breast
    
    @param filenames = list of strings that hold the filenames of interest
    @param location = location of the image to be displayed in the WINDOW.
                      a string saying 'left' or 'right'

    """
    def initialise(self, filenames, location):
        #load in the files into the feature class
        for ii in range(0, len(filenames)):
            self._features.append(feature(levels = 3, wavelet_type = 'haar') )
            file_path = './pilot_images/' + str(filenames[ii])[:-3]   #minus 3 is to get rid of the .gz suffix
            self._features[ii].initialise(file_path)

        self._location = location
        #now will set all of the positions of the images and buttons in the frame
        self.set_location(self._location)


        

        

    """
    set_location()

    Description:
    will set the location of all of the buttons and everything else
    Will set the location for the image, and will offset all of the buttons from the image
    Makes the code usable for both sides and automates it all


    @param location = string saying left or right, tells us which image view we are setting locations for

    """

    
    def set_location(self, location):

        if(location == 'left'):
            self._im_layout = [1,0,10,10]
        else:
            self._im_layout = [1,10,10,10]

        #makes for neater code so I am going to be slightly evel and add            
        #a small amount of hardcoding for the dimensions
        self._rotate_btn_layout = [self._im_layout[2] + self._im_layout[0] + 1, self._im_layout[1], 1,1]
        self._left_btn_layout = [self._im_layout[2] + self._im_layout[0] + 1, self._im_layout[1] + 1, 1,1]
        self._index_disp_layout = [self._im_layout[2] + self._im_layout[0] + 1, self._im_layout[1] + 2, 1,1]
        self._right_btn_layout = [self._im_layout[2] + self._im_layout[0] + 1, self._im_layout[1] + 3, 1,1]

        self._exam_index_disp_layout = [self._im_layout[2] + self._im_layout[0] + 2, self._im_layout[1], 1,1]
        self._exam_index_layout = [self._im_layout[2] + self._im_layout[0] + 2, self._im_layout[1]+1, 1,1]

        self._breast_disp_layout = [self._im_layout[2] + self._im_layout[0] + 2, self._im_layout[1]+2, 1,1]
        self._breast_dropbox_layout = [self._im_layout[2] + self._im_layout[0] + 2, self._im_layout[1]+3, 1,1]
        



        
        

    def get_pos(self, pos_data):
        return pos_data[0], pos_data[1], pos_data[2], pos_data[3]



    
    """
    set_im()

    Decription:
    function will set the mammogram data from the feature class to the ImageView

    @param view = string that will say what type of data we want to view
           valid options:
                  'orig' = default to original mammogram data
                  'pre'  = preprocessed data with artifacts removed
                  'wave' = view of wavelet decomposition
    
    """
    
    def set_im(self, view = 'orig'):

        #align scan function is called to turn the data to the correct orientation for viewing
        self._im_data = self.align_scan(view)

        if(view == 'wave'):
            mask = np.isnan(self._im_data)
            #now lets invert the mask
            mask = np.logical_not(mask)
            #if the view is the wavelet data, will normalise it a bit so it displays nicer
            self._im_data[mask] = np.abs(self._im_data[mask])
            self._im_data[mask] = np.multiply(self._im_data[mask], 4096/(np.max(self._im_data[mask])))
        
    
        self.setImage(self._im_data)
        


        
        
    """
    rotate()

    Description:
    will rotate the image view data when the 
    rotate button in the gui is pressed

    """
    def _rotate(self):

        temp = np.zeros((np.shape(self._im_data)[0], np.shape(self._im_data)[2], np.shape(self._im_data)[1]))
        for ii in range(0, np.shape(temp)[0]):
            temp[ii,:,:] = np.rot90(self._im_data[ii,:,:])

        self._im_data = temp
        self.setImage(self._im_data, autoLevels=False)

            



        


    """
    align_scan()
    
    Description:
    In order to stay consistent, am making the images show as they are supposed to
    For some reason the ImageView widget decides to flip and rotate them, so I am
    just undoing that.
    
    will do a flip up down and rotate 90 degrees clockwise (or 270 degrees counter clockwise)
    
    Scans within an examination aren't always the same size either.
    To prevent errors from this, 

    @param view = string that will say what type of data we want to view
           valid options:
                  'orig' = default to original mammogram data
                  'pre'  = preprocessed data with artifacts removed
                  'wave' = view of wavelet decomposition

    """    

    def align_scan(self, view = 'orig'):    

        num_frames = len(self._features)
        #now will get the maximum size of the scans in this examination
        max_size = self.get_max_size(view)

        #create a temporary array to store the frames that has is the  size of the maximum dimensions
        temp = np.zeros((num_frames, max_size[0], max_size[1]))
        
        #create array that has dimensions of [num_frames, width, height]
        #switch the width and height so image shows properly
        temp_flipped = np.zeros((num_frames, np.shape(temp)[2], np.shape(temp)[1]))
        #now set each frame into the temp array
        for ii in range(0,num_frames):
            #to account for changes in size of scans between scans from the same examination
            #am including some array slicing in the temp variable.
            #if a scan is smaller than the maximum, it will be zero padded to the max size
            
            if(view == 'orig'):
                #get the size, minus one is to account for zero indexing
                size = (np.shape(self._features[ii].original_scan))
                temp[ii,0:size[0],0:size[1]] = np.copy(self._features[ii].original_scan)
            elif(view == 'wave'):
                size = (np.shape(self._features[ii].packets[self._features[ii].indicies[2][0,0]].data))
                temp[ii,0:size[0],0:size[1]] = np.copy(self._features[ii].packets[self._features[ii].indicies[2][0,1]].data) * self._features[ii].packets[self._features[ii].indicies[2][1,0]].data
                
            else:
                print('TODO: Add functionality for viewing other type of images')

            temp[ii,:,:] = np.flipud(temp[ii,:,:])
            #now flip it, k == 3 will flip 270 degrees
            temp_flipped[ii,:,:] = np.rot90(temp[ii,:,:], k=3)

        return temp_flipped



    """
    jump_left() and jump_right()

    Description:
    Will change the image frame we are currently looking at
    
    """
    def jump_left(self):
        self.jumpFrames(-1)
        self.index_disp.setText('Current Scan: %d' %(self.currentIndex + 1))
        
    def jump_right(self):
        self.jumpFrames(1)
        self.index_disp.setText('Current Scan: %d' %(self.currentIndex + 1))



        


    """
    get_max_size()

    Description:
    function to return the maximum size of the images in the current scan
    Is called when we want to update the scan information being displayed

    

    @param view = string to say what view we are looking at.
           valid options:
                  'orig' = default to original mammogram data
                  'pre'  = preprocessed data with artifacts removed
                  'wave' = view of wavelet decomposition
    
    @retval = maximum size of scans in the array. If the maximum size height and width dont occur int
              the same scan, the return value will be the maximum height and width in any scan

    """

    def get_max_size(self, view):

        max_size = [0,0]
        #variable that will point to the data we are finding the size of
        data = []

        #want to loop over all of the scans
        for ii in range(0, len(self._features)):

            if(view == 'orig'):
                data = self._features[ii].original_scan
            elif(view == 'wave'):
                data = self._features[ii].packets[self._features[ii].indicies[2][0,0]].data


            print(np.shape(data))
            print(len(self._features))
            #now see if we need to update the maximum found size
            if( np.shape(data)[0] > max_size[0] ):
                max_size[0] = np.shape(data)[0]
        
            if( np.shape(data)[1] > max_size[1] ):
                max_size[1] = np.shape(data)[1]


        #return the list of maximum found height and width
        return max_size
                
        
class window(view_scan):

    """
    __init__()

    Description:
    Function will just set up a the main window for the application

    """
    
    def __init__(self):
        #run view_scan init function first
        view_scan.__init__(self)
        
        
        #main menu buttons
        self.view_mammogram_btn = QtGui.QPushButton('View Mammogram')
        self.options_btn = QtGui.QPushButton('Options')
        #button functionalities
        self.view_mammogram_btn.clicked.connect(self.view_mammogram)
        
        #initialise layout
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        
        #set layout
        self.layout.addWidget(self.view_mammogram_btn,120,0,50,100)
        self.layout.addWidget(self.options_btn,170,0,50,100)
        
        #file dialog member that will be used for opening the files
        self._file_dialog = QtGui.QFileDialog
        #lets center the window
        #self.resize(500,500)
        self.center()

        #setting the logo
        self.pic = QtGui.QLabel(self)
        #use full ABSOLUTE path to the image, not relative
        self.pic.setPixmap(QtGui.QPixmap(os.getcwd() + "/logo.png"))
        self.layout.addWidget(self.pic,0,0,100,100)
        
    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
        
        
    """
    close_window()
    
    Description:
    Overloads the 
    
    """
    def close_window(self):
        print('WINDOW CLOSE METHOD')
        



    """
    view_mammogram()

    Description:
    Function is called when the view mammogram button in the main screen is called.
    Will open up a file select directory, and then will be passed to the load scan function
    to view the data


    """
    def view_mammogram(self):


        self.clear_window()
        #lets set the menu buttons first, by calling the menu_buttons function
        #defined in the view_scan class
        self.menu_buttons()

        #now lets load in a file selected by the user
        self._file_dialog = QtGui.QFileDialog.getOpenFileName(self, directory= './patients' )

        temp = open(self._file_dialog)
        #save the current patient ID
        self._descriptor.current_patient_id = int(temp.read())
        #now we can load in the scan
        self.begin_viewing()
                
