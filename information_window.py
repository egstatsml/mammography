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









"""
information()

Description:

Class that describes the information window that will pop up
Will also include some buttons in here as well



"""



class information(QtGui.QWidget):

    def __init__(self):
        super(information,self).__init__()

        #now declaring the member variables
        #make a table that will hold a table
        self.table = pandas_model()
        self.view = QtGui.QTableView()










class pandas_model(QtCore.QAbstractTableModel):
    def __init__(self, data = None, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data


    def set_data(self, data):
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return QtCore.QVariant(str(
                    self._data.values[index.row()][index.column()]))
        return QtCore.QVariant()











"""
notes


defines window that will pop up to enter notes


"""



class notes(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)
        
        self.text_box = QtGui.QLineEdit(self)
        self.save_btn = QtGui.QPushButton('Save', self)
        self.status_label = QtGui.QLabel(self)
        self.save_btn.clicked.connect(self.update_label)
        self.setWindowTitle('Mammograph-E - Notes')
        
        #lets size the window
        #making it to the golden ratio
        self.resize(1018,720)

        self.text_box.move(10,10)
        self.text_box.resize( 962, 680) #* (3/4), 320 * np.sqrt(2) * (3/4) )

        self.save_btn.move( 10, 680 + 15)
        self.save_btn.resize(100,20)
        self.status_label.move( 110, 680 + 15)
        self.status_label.resize(100,20)
        
        self.text_box.setText('Please enter patient notes')

    def update_label(self):
        self.status_label.setText('Saved')







class classifier_info(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Mammograph-E - Classifier Information')
        self.output_label = QtGui.QLabel(self)
        self.text_box = QtGui.QTextEdit(self)

        #lets set the text
        self.output_label.setText('Classifier Output')
        self.text_box.setText('Cancer Found = False')
        self.text_box.append('Micocalcifications Found = True')
        self.text_box.append('Masses Found = False')
        self.text_box.append('Architectual Distortion Present = False')
        self.text_box.append('Confidence = 50%')

        self.resize(600, 338)
        self.output_label.resize(600,30)
        self.output_label.move(5,5)
        self.text_box.resize(600, 350)
        self.text_box.move(5, 40)
        


