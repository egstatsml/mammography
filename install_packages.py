#!/usr/bin/env python
"""

Python shell script to install all of the packeges for the DREAM challenge

"""



import os
import re
import sh
import subprocess
import optparse
import sys




def main():

    #install python packages
    my_call('pip install pydicom')
    my_call('pip install scikit-image')
    my_call('pip install PyWavelets')
    my_call('pip install -U scikit-learn')

    #now will unzip all of the images
    #install gzip and gunzip
    my_call('sudo apt-get install gzip & gunzip')
    
    #change directory to the pilot images directory
    my_call('cd ./pilot_images')
    my_call('gunzip ./*.gz')

    


    
def my_call(arg = ' '):
    subprocess.call(arg, shell=True)
    print arg



if __name__ == "__main__":
    main()
