#!/usr/bin/python

import os
import sys
import getopt




"""
arguments()

Description:
Created a class that will handle and store input from command line arguments.





"""

class arguments(object):
    
    
    """
    __init__()
    
    Description:
    Initialise member variables.
    Option is givin to parse the command line arguments from the init function,
    if the commands are givin as part of the input.
    If they arent supplied, the argv variable in the function will default to False
    and therefore wont parse the arguments
    
    @param argv = If supplied should contain a list of the arguments supplied from the command line
                  Otherwise will retain default value of False, and the user can manually parse
                  the arguments at a later stage by calling the parse_command_line function
    
    """
    
    
    
    def __init__(self, argv = False):
        
        self.training = False
        self.preprocessing = False
        self.input_path = []
        self.save_path = []
        self.log_path = []
        self.metadata_path = []
        
        
        
        if(argv != False):
            print argv
            self.parse_command_line(argv)
        
        
    """
    parse_command_line():
    
    Description:
    Gives some flexibility in program, to allow us to specify arguments on command line.
    
    Argument Flags:
    
      -h = Help: print help message
      -t = Train: if we are training the model
      -p = Preprocessing: if we are preprocessing the data
      -i = Input directory = where we will be reading the data from
           Eg. if we are preprocessing, read data from the initial folder.
               If not, will want to specify folder where scans have already been
               preprocessed.
      -s = Save Directory: If we are preprocessing, where should we save the preprocessed
           Images.
      -l = Log Directory: Where should we save the log file to track our progress.
      -m = Metadata patha: where the metadata csv file is
    
    """
    
    def parse_command_line(self, argv):
        
        try:
            
            opts,args = getopt.getopt(argv, "htp:i:s:l:")
            
        except getopt.GetoptError as err:
            print(str(err))
            self.usage()
            sys.exit(2)
            
            
        #lets make sure all of the required arguments are there
        required = ['-i', '-s', '-l', '-m']
        count = 0
        print opts
        for opt, arg in opts:
            print opt
            if((opt == 'i') | (opt == '-s') | (opt == '-l') | (opt == '-m')):
                count = count + 1
                
        #if count doesn't equal the length of required arguments, then not all required arguments
        #were supplied
        if( count < len(required)):
            print count
            print('Not all required arguments were supplied')
            self.usage()
            sys.exit(2)
            
        #now lets go and get the arguments that are here
        for opt, arg in opts:
            if opt == "-h":
                selfusage()
                sys.exit(2)
                
            elif opt == '-t':
                self.training = True
                
            elif opt == '-p':
                self.preprocessing = True
                
            elif opt == '-i':
                self.input_path = str(arg)
                
            elif opt == '-s':
                self.save_path = str(arg)
                
            elif opt == '-l':
                self.log_path = str(arg)
                
            elif opt == '-m':
                self.metadata_path = str(arg)
                
                
                
                
    """
    usage()
    
    Description: Function to be called if the correct input arguments have not been supplied.
                 Will just print a long message explaining how to supply the correct arguments

    """
    def usage(self):
        
        print("Argument Flags: \n -h = Help: print help message \n -t = Train: if we are training the model. If this flag is not specified, the system will not train. \n -p = Preprocessing: if we are preprocessing the data. If this flag is not supplied the system will assume the data in the input path has already been preprocessed. Note if this argument is supplied, then a save path must be specified to save the preprocessed scans. This is done with the -s variable and is described below. \n -i = Input directory = where we will be reading the data from \n Eg. if we are preprocessing, read data from the initial folder. \n If not, will want to specify folder where scans have already been preprocessed. \n -s = Save Directory: If we are preprocessing, where should we save the preprocessed Images. \n -l = Log Directory: Where should we save the log file to track our progress. -m = Metadata directory: Where the metadata csv file is stored \n \n --------------------------------------- \n Required Arguments: \n -i, -s, -l \n If these arguments are not supplied, along with a correct path for each argument, the program will not run, and you will see this message :) \n \n Example - run on Synapse Server: \n sudo python main_thread.py -p -t -i /trainingData -s /preprocessedData -l /modelState -m /metadata \n \n Example - Run on Dimitri's Machine \n sudo python main_thread.py -p -t -i /media/dperrin/ -s /media/dperrin/preprocessedData/ -l ./ -m ./")

