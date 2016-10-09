"""
file will just get the filenames of the different patients and save them 
as a seperate file.
Not the most efficient way to do it, but the most intuitive for a user

"""

import pandas as pd
import numpy as np



metadata =pd.read_excel('../exams_metadata_pilot.xlsx')
for ii in range(0, metadata.shape[0]):
    file_name = str(metadata.iloc[ii,0])[:-2]
    print(file_name)
    text_file = open('./' + file_name, "w")
    text_file.write(file_name)
    text_file.close()
    
    
    
    
