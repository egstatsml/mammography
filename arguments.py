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
Enjoy

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
        self.model_path = []
        self.validation = False
        self.kernel = 1
        self.degree = 3
        self.gamma = 0
        self.cost = 1.0
        self.weight = 1
        self.epsilon = 0.1
        self.probability = False
        self.weight = {'0':1,'1':1}
        self.train_string = []
        self.validation_string = []
        self.challenge_submission = False
        self.sub_challenge = 1
        self.principal_components = 0
        self.extract_features = False
        self.kl_divergence = 0
        self.rcnn = 0
        
        
        
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
      -m = Metadata path: where the metadata csv file is
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
       -c = Challenge Submission
       --cost = cost value for training svm
       --rcnn = whether we are using detections the rcnn implementation
       --sub = Sub challenge we are running for
             1 = Sub Challenge 1 - Don't use any metadata as features for classification
             2 = Sub challenge 2 - Use metadata for classification
       --pca = If we want to do PCA on the textural based features we find
               Should supply the number of principal components to be used
               Note will not do PCA on any metadata or density estimate features
       --model = Path of directory where model should be saved
       --kl = Use Kl Divergence to specify which features to use for classification
              should supply the number or arguments to use for classification
       -f = extract features from preprocessed data. NOTE that this is done by default
            when you specify you wan't to do preprocessing with the -p flag.
            If you have already done the preprocessing though and you want to extract 
            features again, you should use this flag, but you shouldn't use this flag in
            conjunction with the preprocessing flag
    """
    
    def parse_command_line(self, argv):
        
        try:
            
            opts,args = getopt.getopt(argv, "htpbvfcm:i:s:l:k:d:g:e:w:", ['rcnn=', 'sub=', 'pca=', 'kl=', 'model=', 'cost='])
            
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
                self.validation = True
                
            elif opt == '-k':
                self.kernel = int(arg)
                
            elif opt == '-d':
                self.degree = int(arg)
                
            elif opt == '-g':
                self.gamma = float(arg)
                
            elif opt == '-e':
                self.epsilon = float(arg)
                
            elif opt == '-b':
                self.probability = True
                
            elif opt == '-c':
                self.challenge_submission = True            
                
            elif opt == '--rcnn':
                self.rcnn = str(arg)
                
            elif opt == '--sub':
                self.sub_challenge = int(arg)
                
            elif opt == '--pca':
                self.principal_components = int(arg)
                print self.principal_components
                
            elif opt == '--kl':
                self.kl_divergence = int(arg)
                    
            elif opt == '--cost':
                self.cost = float(arg)
                
            elif opt == '--model': 
                self.model_path = str(arg)
                print('found model path')
                print(self.model_path)
                
            elif opt == '-f':
                print 'here'
                self.extract_features = True
                
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
                        self.weight[jj] = init_dict[float(jj)]
                                                
                except Exception as e:
                    print('There was an error with your weight inputs.')
                    print('You should enter weights for each class in a dictionary style format,but with no spaces please :)')
                    print('Example: -w 0:1,1:20')
                    print()
                    print(e)
                    
        #and lets do some error checking on the inputs
        self.check_inputs(opts)
        
        #now that everything is set and valid, we should set the strings to run the training and validation
        if(self.training):
            weight_string = ''
            for ii in self.weight.keys():
                weight_string += '-w%s %s ' %(ii, self.weight[ii])
                
            #This is here so that we can run training and preprocessing in the same
            #step if we like
            #if we are preprocessing, the train file will be in the save_path
            #if we aren't, it should be in the input_path
            if(self.preprocessing | self.extract_features):
                train_file_path = self.save_path
            else:
                train_file_path = self.input_path
                    
            print train_file_path
            
            
            self.train_string ='./CUDA/svm-train-gpu -c %f -t %s -d %s -m 8000 -e %s %s %s/model_data/data_file_libsvm %s/model_file' %(self.cost, self.kernel, self.degree, self.epsilon, weight_string, train_file_path, self.model_path)
            
            
            print self.train_string
            

        #the model path is read only in the challenge submissions
        #and for sub challenge 2 I want to add the metadata so I do write to
        #this path
        #this is why I am setting it slightly differently
        if(self.validation):
            if(self.sub_challenge == 2) & (self.challenge_submission):
                data_file_string = '/scratch/'
            else:
                data_file_string = self.save_path + '/model_data/'

            self.validation_string = './LIBSVM/svm-predict %s/data_file_libsvm %s/model_file %s/model_data/results.txt' %(data_file_string, self.model_path, self.save_path)            
            #if we want probability measures, just add the -b flag
            if(self.probability):
                self.validation_string += ' -b'
                
            print self.validation_string
            
            
            
            
            
            
            
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
            unwanted = ['-k', '-d', '-g', '-e', '-w']
            for opt, arg in opts:
                if opt in unwanted:
                    print('parameter %s supplied, but not training' %opt)
                    print('You have supplied classifier paramaters, but haven\'t specified that we should be training the model')
                    print('To specify that you wan\'t to train, use the -t argument')
                    print('For more usage information, run with -h argument')
                    sys.exit(2)
                    
                    
                    
        #if we are doing the preprocessing, we will be extracting features while we are there
        #if the user supplies flags for both feature extraction and preprocessing, just
        #display a little warning to let them know what is going to happen
        if(self.preprocessing & self.extract_features & self.training):
            print("""
            WARNING!
            You have supplied flags for both preprocessing (-p) and feature extratction (-f)
            Feature extraction is done by default if preprocessing is selected.
            
            This can cause a problem if you have specified that you would like to do training
            as well with the (-t) flag, which you have also done.
            
            To avoid doing it twice by accident, during the preprocessing and training stage,
            I am disabling the (-f) flag so feature extracted is only done once.
            Trust me, this is what you wanted to really happen
            """)
            self.extract_features = False
            
            
            
        #if we do want to do some training, we should ensure that the data has already been
        #preprocessed, or that it will be done in this run of the program
        if(self.training):
            #if we are doing the preprocessing right now, we are all good so don't need to
            #check any further
            if(not self.preprocessing):
                #if we aren't preprocessing, check that preprocessing has already been done
                #to do this, will just list the number of files in the preprocessed directory
                #or the 'input_path'
                if len([name for name in os.listdir(self.input_path) if os.path.isfile(os.path.join(self.input_path, name))]) < 50:
                    print('WARNING')
                    print('You have suggested training the model, but I can\'t find any of the preprocessed data saved')
                    print('I will attempt to train using just the feature vectors, assuming they have been saved')
                    
                
        #some more error checking
        #if they have specified to do validation, we should expect either training to be completed,
        #or the file of the model to exist. If it doesn't, we should let the user know
        
        if(self.validation):
            #if we aren't training
            if(not self.training):
                #the file should already exist then
                if(not os.path.isfile(self.model_path + 'model_file')):
                    print('ERROR')
                    print('You have suggested running validation, but the model doesn\'t exist yet, and you haven\'t specified that you want to train the model')
                    print('Before validation can be done, the model must be run with the -t argument to train')
                    print('For more usage information, run with -h argument')
                    sys.exit(2)        
                    
                    
                    
                if(self.model_path == []):
                    print("""
                    ERROR: Have specified you want to run validation, but haven't included the model path
                    using the flag (-a).
                    """)
                    sys.exit(2)
                    
                    
        #if we are doing a challenge submission, we should make sure that the correct argument is supplied
        #should be 1 for sub challenge 1, or 2 for sub challenge 2
        if(self.challenge_submission):
            if not( (self.sub_challenge == 1) | (self.sub_challenge == 2) ):
                print('ERROR')
                print('You supplied the -c flag to say that it is a challemge submission, but haven\'t supplied a correct argument.')
                print('It should be either 1 for sub challenge 1 or 2 for sub challenge 2.')
                print('You supplied %d ' %(self.sub_challenge))
                print('For more usage information, run with -h argument')
                sys.exit(2)
                    
                
                
                
                
        #if we have supplied both kl and pca arguments, throw an error, can only use 1 at a time
        #should be 1 for sub challenge 1, or 2 for sub challenge 2
        if(self.principal_components & self.kl_divergence):
            print('ERROR')
            print('You supplied the --pca flag and the --kl flag')
            print('The features are used for dimensionality reduction of features before classifying')
            print('Should be one or the other')
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
        -c = Challenge Submission
        --sub = Sub challenge we are running for
             1 = Sub Challenge 1 - Don't use any metadata as features for classification
             2 = Sub challenge 2 - Use metadata for classification
        --pca = If we want to do PCA on the textural based features we find
                Should supply the number of principal components to be used
                Note will not do PCA on any metadata or density estimate features
        --model = Path of directory where model should be saved
        -f = extract features from preprocessed data. NOTE that this is done by default
            when you specify you wan't to do preprocessing with the -p flag.
            If you have already done the preprocessing though and you want to extract 
            features again, you should use this flag, but you shouldn't use this flag in
            conjunction with the preprocessing flag
        
        Note it is best if you split the tasks up, say preprocess (-p) then train (-t)
        and then validate (-v) though in validation you may want to use the preprocessing
        flag to specify that you want to preprocess the data. This option is given so that
        if you preprocess the data you don't have to run it all again unless you specify it.
        
        
        
        Example 1 - run on Synapse Server to preprocess 
        python main_thread.py -p -t -i /trainingData -s /preprocessedData -l /modelState -m /metadata 


        Example 2 - Run on Dimitri's Machine  to train:
        sudo python main_thread.py -t -i /media/dperrin/preprocessed/preprocessedTrain/ -s /media/dperrin/preprocessed/preprocessedTrain -l ./ -m ./  -k 1 -d 4 -e 0.001 -b -w 0:1,1:20 -a /media/dperrin/preprocessed/preprocessedTrain/model_data
        
        Example 3 - Run validation and preprocess the validation data on Dimitri's Machine
        
        sudo python main_thread.py -p -v -i /media/dperrin/val_images/ -s /media/dperrin/preprocessed/preprocessedVal -l ./ -m ./  -a /media/dperrin/preprocessed/preprocessedTrain/model_data
        
        Example 4 - Run validation and on Dimitri's Machine, assuming we have previously preprocessed it
        
        sudo python main_thread.py -v -i /media/dperrin/preprocessed/preprocessedVal/ -s /media/dperrin/preprocessed/preprocessedVal -l ./ -m ./  -a /media/dperrin/preprocessed/preprocessedTrain/model_data
        """)


