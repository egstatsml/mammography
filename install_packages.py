#!/usr/bin/env python
"""

Python shell script to install all of the packeges for the DREAM challenge

"""



import os
import re
import subprocess
import optparse
import sys




def main():
    
    my_call('pip install pydicom')
    my_call('pip install scikit-image')
    my_call('pip install PyWavelets')
    my_call('pip install -U scikit-learn')
    


    
def my_call(arg = ' '):
    subprocess.call(arg, shell=True)
    print arg



if __name__ == "__main__":
    main()
