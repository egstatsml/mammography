#!/usr/bin/python

import os
import sys
import getopt
import ast

"""
arguments()

Description:
Created a class that will handle and store input from command line arguments.





"""
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



class arguments(object):    
    
    def __init__(self, argv = False):
        
        self.training = False
        self.validation = False
        self.preprocessing = False
        self.input_path = []
        self.save_path = []
        self.log_path = []
        self.metadata_path = []
        self.validation = 0
        self.kernel = 1
        self.degree = 3
        self.gamma = 0
        self.weight = 40
        self.epsilon = 0.1
        self.probability = False
        self.weight = {'0':1,'1':1}
        self.train_string = []
        self.validation_string = []
        
        if(argv != False):
            print(argv)
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
      -v = Validation. If we are validating the data, we want to train on a portion and
           validate the set on another portion. The number of scans used for validating will be
           supplied along with this flag
      -k = Kernel to be used. Will stick with LIBSVM notation  (Default of 1 : Polynomial)
           0 = linear
           1 = polynomial
           2 = RBF
           3 = Sigmoid
          
           NOTE: Default used here is polynomial, whilst default in LIBSVM is RBF. Have had
           troubles with RBF, so should avoid using it at the moment.
    
       -d = Degree of kernel. Only valid for polynomial and Sigmoid kernels (Default of 3)
       -g = Gamma. (Default of 1/n_features)
       -e = Epsilon. Tolerance for termination. (Default of 0.001)
       -b = Probability estimates (Default of False)
       -w = Weight for each class. usage here should be a dictionary like reference for each class
    
    """
    
    def parse_command_line(self, argv):
        
        try:
            
            opts,args = getopt.getopt(argv, "htpbm:i:s:l:v:k:d:g:e:w:")
            
        except getopt.GetoptError as err:
            print(str(err))
            self.usage()
            sys.exit(2)
            
            
        #lets make sure all of the required arguments are there
        required = ['-i', '-s', '-l', '-m']
        count = 0
        for opt, arg in opts:
            if((opt == '-i') | (opt == '-s') | (opt == '-l') | (opt == '-m')):
                count = count + 1
                
        #if count doesn't equal the length of required arguments, then not all required arguments
        #were supplied
        if( count < len(required)):
            print(count)
            print('Not all required arguments were supplied')
            self.usage()
            sys.exit(2)
            
        #now lets go and get the arguments that are here
        for opt, arg in opts:
            print opt
            if opt == "-h":
                self.usage()
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
                
            elif opt == '-v':
                self.validation = int(arg)
                print 'hereherehe'
            elif opt == '-k':
                self.kernel = int(arg)
                
            elif opt == '-d':
                self.degree = int(arg)
                
            elif opt == '-g':
                self.gamma = float(arg)
                
            elif opt == '-e':
                self.epsilon = float(arg)
                
            elif opt == '-e':
                self.epsilon = float(arg)
                
            elif opt == '-b':
                self.probability = True
                
            elif opt == '-w':
                #convert the string argument into a dictionary
                init_dict = ast.literal_eval('{' + str(arg) + '}')
                #now will iterate over the input dictionary to see which weights
                #for which classes need updating
                #if the user input the weights incorrectly in the dictionary, it will throw a
                #detailed error and explain how to use it
                try:
                    print self.weight.keys()
                    print init_dict.keys()
                    for jj in self.weight.keys():
                        self.weight[jj] = init_dict[int(jj)]
                        
                except Exception as e:
                    print('There was an error with your weight inputs.')
                    print('You should enter weights for each class in a dictionary style format,but with no spaces please :)')
                    print('Example: -w 0:1,1:20')
                    print()
                    print(e)
                    
        #and lets do some error checking on the inputs
        self.check_inputs(opt)
        
        #now that everything is set and valid, we should set the strings to run the training and validation
        if(self.training):
            weight_string = ''
            for ii in self.weight.keys():
                weight_string += '-w%s %s ' %(ii, self.weight[ii])
                
            self.train_string = '~/CUDA/src/linux/svm-train-gpu  -t %s -d %s -m 5000 -e %s %s %s/train_file_libsvm %s/model_file' %(self.kernel, self.degree, self.epsilon, weight_string, self.log_path, self.log_path)
            
            
            
            if(self.validation != 0):
                self.validation_string = '~/libsvm/svm-predict %s/predict_file_libsvm %s/model_file %s/results.txt' %(self.log_path, self.log_path, self.log_path)
            
            
            
            
            
    """
    check_inputs()
    
    Description:
    Will check that the command line arguments supplied are valid, and make sense.
    This is here mostly as a sanity check for the user, so they can't provide inputs
    that either won't be used or aren't valid.
    
    Eg. Supplying inormation on the classifier if we haven't explicitly said that we want to train
        the model
    
    
    Is here to ensure that the user knows completely what is going on, and that they don't 
    do something by accident.
    
    @param opts = options from command line
    
    """
            
    def check_inputs(self, opts):
        #now lets do some quick checking of our inputs to make sure no un-used inputs were provided
        #example if we aren't training, we shouldn't expect any information on the type or degree of the
        #kernel to be supplied.
        #This shouldn't cause any errors in the code, but it probably would cause an unwanted run path of the model
        
        if(not self.training):
            #if we aren't validating, make sure no classifier parameters are set
            unwanted = ['-k', '-d', '-g', '-e', '-b', '-w']
            for opt, arg in opts:
                if opt in unwanted:
                    print('parameter %s supplied, but not training' %opt)
                    print('You have supplied classifier paramaters, but haven\'t specified that we should be training the model')
                    print('To specify that you wan\'t to train, use the -t argument')
                    print('For more usage information, run with -h argument')
                    sys.exit(2)
                    
                    
        #if we do want to do some training, we should ensure that the data has already been
        #preprocessed, or that it will be done in this run of the program
        if(self.training):
            #if we are doing the preprocessing right now, we are all good so don't need to
            #check any further
            if(not self.preprocessing):
                #if we aren't preprocessing, check that preprocessing has already been done
                #to do this, will just list the number of files in the preprocessed directory
                #or the 'input_path'
                if len([name for name in os.listdir(self.input_path) if os.path.isfile(os.path.join(self.input_path, name))]) < 100:
                    print('ERROR')
                    print('You have suggested training the model, but we haven\'t preprocessed the data yet')
                    print('Before training can be done, the model must be run with the -p aregument to preprocess the data')
                    print('For more usage information, run with -h argument')
                    sys.exit(2)        
                    
                
        #some more error checking
        #if they have specified to do validation, we should expect either training to be completed,
        #or the file of the model to exist. If it doesn't, we should let the user know
        
        if(self.validation):
            #if we aren't training
            if(not self.training):
                #the file should already exist then
                if(not os.path.isfile(self.log_path + 'model_file')):
                    print('ERROR')
                    print('You have suggested running validation, but the model doesn\'t exist yet, and you haven\'t specified that you want to train the model')
                    print('Before validation can be done, the model must be run with the -t aregument to train')
                    print('For more usage information, run with -h argument')
                    sys.exit(2)        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
    """
    usage()
    
    Description: Function to be called if the correct input arguments have not been supplied.
                 Will just print a long message explaining how to supply the correct arguments
    
    """
    
    def usage(self):        
        print("""Argument Flags:
        
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
        -v = Validation. If we are validating the data, we want to train on a portion and
           validate the set on another portion. The number of scans used for validating will be
        supplied along with this flag
        -k = Kernel to be used. Will stick with LIBSVM notation  (Default of 1 : Polynomial)
            0 = linear
            1 = polynomial
            2 = RBF
            3 = Sigmoid
           
           NOTE: Default used here is polynomial, whilst default in LIBSVM is RBF. Have had
           troubles with RBF, so should avoid using it at the moment.
           
        -d = Degree of kernel. Only valid for polynomial and Sigmoid kernels (Default of 3)
        -g = Gamma. (Default of 1/n_features)
        -e = Epsilon. Tolerance for termination. (Default of 0.001)
        -b = Probability estimates (Default of False)
        -w = Weight for each class. usage here should be a dictionary like reference for each class
        
        Example 1 - run on Synapse Server to preprocess and train: 
        python main_thread.py -p -t -i /trainingData -s /preprocessedData -l /modelState -m /metadata -k 1 -d 4 -e 0.001 -b -w 0:1,1:20

        Example 2 - Run on Dimitri's Machine  to train and validate:
        sudo python main_thread.py -p -t -i /media/dperrin/ -s /media/dperrin/preprocessed/ -l ./ -m ./ -v 100 -k 1 -d 4 -e 0.001 -b -w 0:1,1:20")
        
        Example 3 - Run just training and validation on Dimitri's Machine
        sudo python main_thread.py -t -i /media/dperrin/preprocessed -s /media/dperrin/preprocessed/ -l ./ -m ./ -v 100 -k 1 -d 4 -e 0.001 -b -w 0:1,1:20")
        
        
        
        """)
              
              
        print("Argument Flags: \n -h = Help: print help message \n -t = Train: if we are training the model. If this flag is not specified, the system will not train. \n -p = Preprocessing: if we are preprocessing the data. If this flag is not supplied the system will assume the data in the input path has already been preprocessed. Note if this argument is supplied, then a save path must be specified to save the preprocessed scans. This is done with the -s variable and is described below. \n -i = Input directory = where we will be reading the data from \n Eg. if we are preprocessing, read data from the initial folder. \n If not, will want to specify folder where scans have already been preprocessed. \n -s = Save Directory: If we are preprocessing, where should we save the preprocessed Images. \n -l = Log Directory: Where should we save the log file to track our progress. -m = Metadata directory: Where the metadata csv file is stored \n \n --------------------------------------- \n Required Arguments: \n -i, -s, -l \n If these arguments are not supplied, along with a correct path for each argument, the program will not run, and you will see this message :) \n \n Example - run on Synapse Server: \n sudo python main_thread.py -p -t -i /trainingData -s /preprocessedData -l /modelState -m /metadata \n \n Example - Run on Dimitri's Machine \n sudo python main_thread.py -p -t -i /media/dperrin/ -s /media/dperrin/preprocessedData/ -l ./ -m ./ -v 100")
        
    
