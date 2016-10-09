"""
widget class to handle the gui stuff


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


#import my classes
from breast import breast
from feature_extract import feature
from read_files import spreadsheet




#will be a member class for the window 
class view_scan(QtGui.QWidget):

    def __init__(self):

        #initialise the GUI Widget
        QtGui.QWidget.__init__(self)
        #setting all the member variables
        #member buttons for the viewing screens
        self.dimensions_btn = QtGui.QPushButton('dimensions')
        self.dimensions_btn.clicked.connect(self.get_dimensions)
        self.asymmetry_btn = QtGui.QPushButton('Asymmetry')
        self.asymmetry_btn.clicked.connect(self.load_asymmetry)
        self.previous_btn = QtGui.QPushButton('Previous Scans')
        self.previous_btn.clicked.connect(self.load_previous_scans)
        self.close_btn = QtGui.QPushButton('Close')
        self.close_btn.clicked.connect(self.close_window)
        self.artifacts_btn = QtGui.QPushButton('Remove Artifacts')
        self.artifacts_btn.clicked.connect(self.remove_artifacts)
        #buttons that allow  rotating the scans
        self.rotate_l_btn = QtGui.QPushButton('Rotate')
        self.rotate_r_btn = QtGui.QPushButton('Rotate')
        self.rotate_l_btn.clicked.connect(self.rotate_l)
        self.rotate_r_btn.clicked.connect(self.rotate_r)
        #initialising text box and list Widgets
        self.text = QtGui.QLineEdit('enter text')
        self.listw = QtGui.QListWidget()
        
        #scan data
        self._im_l_data = []   #array that will have the image data stored in here
        self._im_r_data = []
        
        #image view widgets
        self.im_l = pg.ImageView(view = pg.PlotItem()) #initiate with plot item to set axis ticks on
        self.im_r = pg.ImageView(view = pg.PlotItem()) #initiate with plot item to set axis ticks on
        
        #overlay for the waiting indicator
        self.loading = Overlay(self.im_l)
        self.loading.hide()
        
        #boolean variable that will say if the right image is being used or not
        #might change the way this is done later on though
        self._right_used = False
        #hiding the histogram and other stuff
        self._hide_imageview_extras()
        
        
        
        #member variables
        self._data = []
        #create member variable that will hold the features of a patients scan
        self._patient_data_l = feature(levels = 3, wavelet_type = 'haar', no_images = 1)
        self._patient_data_r = feature(levels = 3, wavelet_type = 'haar', no_images = 1)
        self._descriptor = spreadsheet(benign_files=None, run_synapse=False)
        #dictonary that will hold a descriptor and layout id number
        self._load_scan_dict = {'dimensions_btn' : 0, 'asymmetry_btn' : 1,
                                'previous_btn' : 2, 'artifacts_btn': 3, 'close_btn' : 4,
                                'im_l': 5, 'im_r': 6}
        
        
        






    ######################################################################
    #
    #                        Button Functions
    #
    ######################################################################
    


    """
    load_scan_func()
    
    Description:
    Function is called when we are redy to show some mammogram scans
    Will load up the data and then get it ready to ready to show
    
    Also initialise the layout for the GUI design
    
    """
    
    def load_scan_func(self, patientId):
        #clear the current window
        self.clear_window()
        self.showMaximized()
        #read in the data
        self._descriptor.get_most_recent(patientId)
        self._file_path = './pilot_images/' + str(self._descriptor.filename_l[0])[:-3]
        print(self._file_path)
        self._patient_data_l.initialise(self._file_path)
        #temp = dicom.read_file(self._file_path)
        #self.data = np.fromstring(temp.PixelData,dtype=np.int16).reshape((temp.Rows,temp.Columns))
        self._im_l_data = self.align_scan(self._patient_data_l.original_scan)
        self.im_l.setImage(self._im_l_data)
        
        
        
        ## Add widgets to the layout in their proper positions
        self.layout.addWidget(self.dimensions_btn, 0, 0,1,2) 
        self.layout.addWidget(self.asymmetry_btn, 0, 2,1,2)  
        self.layout.addWidget(self.previous_btn, 0, 4,1,2)   
        self.layout.addWidget(self.artifacts_btn, 0, 6,1,2)  
        self.layout.addWidget(self.close_btn, 0, 8,1,2) 
        #self.layout.addWidget(self.text, 1, 0)   # text edit goes in middle-left
        #self.layout.addWidget(self.listw, 0, 8,1,2)  # list widget goes in bottom-left
        self.layout.addWidget(self.im_l, 1, 0,1,10)  # plot goes on right side, spanning
        self.layout.addWidget(self.rotate_l_btn, 10, 0,1,2)  
        
        
    """
    load_asymmetry()
    
    Description:
    Function will load in the data from the right image and place it next to the other scan of
    the breast
    
    """
    def load_asymmetry(self):
        
        self._file_path = './pilot_images/' + str(self._descriptor.filename_l[1])[:-3]
        print(self._file_path)
        self._patient_data_r.initialise(self._file_path)
        self._im_r_data = self.align_scan(self._patient_data_r.original_scan)
        self.im_r.setImage(self._im_r_data)
        self.layout.addWidget(self.im_r, 1, 10,1,10)  # plot goes on right side, spanning 3 rows
        self.layout.addWidget(self.rotate_r_btn, 10, 10,1,2)        
        #set the boolean variable for the right image being used to True
        self._right_used = True
        
        
        
    """
    align_scan()
    
    Description:
    In order to stay consistent, am making the images show as they are supposed to
    For some reason the ImageView widget decides to flip and rotate them, so I am
    just undoing that.
    
    will do a flip up down and rotate 90 degrees clockwise (or 270 degrees counter clockwise)
    """
        
    def align_scan(self, im):
        temp = np.copy(im)
        temp = np.flipud(temp)
        return np.rot90(temp,3)  #the 3 argument is to rotate 270 counter clockwise
        
        

















    #######################################################################
    #
    #            Methods to called by pushing the buttons
    #
    #
    #
    #######################################################################
        
    """
    remove_artifacts()
    
    Description:
    Called when the remove artifacts button is pressed on the gui
    
    Will then use the preprocessing function in the breast class to remove
    labels, skin and anything else that shouldnt be there
    """        
        
    def remove_artifacts(self):
        
        self._patient_data_l.preprocessing()
        self._im_l_data = self.align_scan(self._patient_data_l.data)
        self.im_l.setImage(self._im_l_data)
        #see if the right image is being used. If it is,
        #remove that stuff as well
        if(self._right_used == True):
            self._patient_data_r.preprocessing()
            self._im_r_data = self.align_scan(self._patient_data_r.data)
            self.im_r.setImage(self._im_r_data)
        self.loading.hide()


            
    def load_other_view(self):
        print('TODO: add functionality')
        
        
    def load_previous_scans(self):
        print('TODO: add functionality')
        
    def close_window(self):
        print('TODO: add functionality')
        
        
        
    def get_dimensions(self):
        a = self.im.getView()
        print(a.viewRange())
        #self.layout.removeWidget(self.im)
        #self.layout.addWidget(self.text, 1, 0)   # text edit goes in middle-left
        
        
        
    def rotate_l(self):
        self._im_l_data = np.rot90(self._im_l_data)
        self.im_l.setImage(self._im_l_data)
        
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
            
            #use this method if we want to delete widgets, though probably not
            #what we want
            
            #widgetToRemove = self.layout.itemAt( i ).widget()
            #remove it from the layout list
            #self.layout.removeWidget( widgetToRemove )
            #remove it from the gui
            #widgetToRemove.setParent( None )













            
            
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
        self.load_scan_btn = QtGui.QPushButton('Load Scan')
        self.options_btn = QtGui.QPushButton('Options')
        #button functionalities
        self.load_scan_btn.clicked.connect(self.load_scan_func)
        
        #initialise layout
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        
        #set layout
        self.layout.addWidget(self.load_scan_btn,0,0,10,3)
        self.layout.addWidget(self.options_btn,10,0,10,3)
        
        #file dialog member that will be used for opening the files
        self._file_dialog = QtGui.QFileDialog
        
        
        
    """
    close_window()
    
    Description:
    Overloads the 
    
    """
    def close_window(self):
        print('WINDOW CLOSE METHOD')
        
        
    def load_scan_func(self):
        print('here')
        #self._file_dialog.setDirectory(self._file_dialog, QtCore.QString('/DREAM/pilot_images/'))
        self._file_dialog = QtGui.QFileDialog.getOpenFileName(self, directory= './patients' )
        temp = open(self._file_dialog)

        view_scan.load_scan_func(self, int(temp.read()))
        
        
        
        
        
        
        
        
        
        
        
"""
busy indicator

Will pop up whenever the gui does something that will take a little while,
just to let the user know that the program is loading

Just taken from the PyQt tutorial page

https://wiki.python.org/moin/PyQt/A%20full%20widget%20waiting%20indicator

"""




class Overlay(QWidget):

    def __init__(self, parent = None):
    
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)
        
        """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        """
        
    def paintEvent(self, event):
    
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 127)))
        painter.setPen(QPen(Qt.NoPen))
        
        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QBrush(QColor(127 + (self.counter % 5)*32, 127, 127)))
            else:
                painter.setBrush(QBrush(QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width()/2 + 30 * np.cos(2 * np.pi * i / 6.0) - 10,
                self.height()/2 + 30 * np.sin(2 * np.pi * i / 6.0) - 10,
                20, 20)
        
        painter.end()
        
        
        
    def showEvent(self, event):
    
        self.timer = self.startTimer(50)
        self.counter = 0
    
    def timerEvent(self, event):
    
        self.counter += 1
        self.update()
        if self.counter == 60:
            self.killTimer(self.timer)
            self.hide()

        
        
        
        
        
        
        
