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
        
        #now lets set any other member variables we might need
        #probably some bottons or something
        
        






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
