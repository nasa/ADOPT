#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notices:

Copyright © 2019 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

Created on Tue Nov 12 14:32:35 2019

@author: dweckler
"""




from shutil import copyfile,copy,copytree
import pandas as pd
import json
import glob
import os
import errno
from pathlib import Path




def get_filepath(input_message):
    while True:
        dataset_folder_path = input(input_message)

        if not os.path.exists(dataset_folder_path) or dataset_folder_path == "":
            print("please enter a valid filepath")
            continue        
        else:
            return dataset_folder_path       
           # break
        
def copy_files_or_directories(source, dest):
    try:
        copytree(source, dest)
    except OSError as exception:
        if exception.errno == errno.ENOTDIR:
            try:
                copy(source, dest)
            except:
                print("Error copying file")                
        else:
            raise
     
        
#recursively list visible files 
def list_visible_files(path):
    filepath = os.path.join(path,'*')
    
    all_files = [os.path.basename(f) for f in glob.glob(filepath)]
    
    return all_files


def generate_file_list(directory,dataset_folder_path):
    dir_name = os.path.basename(directory)

    file_list = glob.glob(os.path.join(directory,"*.csv"))
    file_list = [os.path.basename(pth) for pth in file_list]
    file_list = [os.path.join(dir_name,file) for file in file_list]
    
    file_list_txt = os.path.join(dataset_folder_path,f"{dir_name}.txt")
    
    with open(file_list_txt,'w') as f:
        for filename in file_list:
            f.write(filename)
            f.write('\n')
            
    return file_list_txt


#TODO: Also maybe generate DTMIL_config_dir.json
def make_directory(folder_path):
        
    try:  
        os.mkdir(folder_path)
    except OSError:  
        print ("Creation of the dataset folder %s failed" % folder_path)
        
#TODO: add a menu option to just export this by itself
def export_json_cfg(directory = "", json_filename = "DTMIL_config.json", fl_nominal = "filelist_nominal.txt", fl_adverse = "filelist_adverse.txt",param_names = []):
    
    
    sep = os.path.sep
    default_config_path = "dtmil{}configuration{}".format(sep,sep,sep)
    
    
    
    with open(os.path.join(default_config_path,"DTMIL_config_default.json")) as jsonfile:
        json_data =  json.load(jsonfile)
    
    json_data["importing"]["nominal_filename"] = fl_nominal
    json_data["importing"]["adverse_filename"] = fl_adverse
    json_data["preprocessing"]["all_parameter_names"] = param_names
    

    
    json_cfg_string = json.dumps(json_data,sort_keys=True, indent=4, separators=(',', ': '))
    
    with open(os.path.join(directory,json_filename),'w') as outfile:
        outfile.write(json_cfg_string)
        outfile.close()
    
    


print("Dataset Formatter: Command Line Version")

#filelist toggle
while True:
    fl_toggle = input("Do you have a list of nominal files and a list of adverse files? (y/n)\n")
    if fl_toggle not in ['y','n']:
        print("please enter a valid response")
        
    else: 
        filelist_toggle = True if fl_toggle=='y' else False
        data_description = "filelist" if filelist_toggle else "folder"
        break

    
dataset_folder_path = get_filepath("Input the folder of your dataset\n")
nominal_filelist_path = get_filepath(f"Input the path of the Nominal {data_description}\n")
adverse_filelist_path = get_filepath(f"Input the path of the Adverse {data_description}\n")
        
if(not filelist_toggle): 
    nominal_filelist_path = generate_file_list(nominal_filelist_path,dataset_folder_path)
    adverse_filelist_path = generate_file_list(adverse_filelist_path,dataset_folder_path)
        
print("Attempting to read filepath from nominal filelist")
data_filenames = []
with open('{}'.format(nominal_filelist_path),'r') as f:
    data_filenames = f.readlines()
    data_filenames = [x.strip() for x in data_filenames]

#test to see if the files are there before doing anything
test_file_path = data_filenames[0]
csv_filepath = os.path.join(dataset_folder_path,test_file_path)

try:
    df = pd.read_csv(csv_filepath)
    parameter_list = list(df.columns.values)
    
except:
    print("Error", f"Could not find any data files in the specified Data Folder path:\n{dataset_folder_path}\n\nAttempted to open file:\n{csv_filepath}")
    
while True:
    dataset_name = input("type the name of your dataset\n")
    
    if (dataset_name == ""):
        print("please enter a valid name")
        continue
    
    else:
        break
        

dataset_path = input("type the path you'd like to save the dataset in (or press enter to save to desktop)\n")

if dataset_path == "":
    dataset_path = "{}/Desktop".format(str(Path.home())
)
    
dataset_path = os.path.join(dataset_path,dataset_name)

print("Generating folder structure for dataset {}".format(dataset_path))


#create new dataset directory (if one doesn't exist)
make_directory(dataset_path)

#create data folder
data_path = os.path.join(dataset_path,'data')
make_directory(data_path)

#within data folder, create parameters and raw_data folder

parameters_path = os.path.join(data_path, "parameters")
make_directory(parameters_path)

raw_data_path = os.path.join(data_path,'raw_data')
make_directory(raw_data_path)

#create misc, model_storage, and model_output folders
model_storage_path = os.path.join(dataset_path,'model_saves')
make_directory(model_storage_path)

model_output_path = os.path.join(dataset_path, "output")
make_directory(model_output_path)

misc_path = os.path.join(dataset_path,"misc")
make_directory(misc_path)
#create model_saves within the misc folder

model_saves_path = os.path.join(misc_path,"model_archive")
make_directory(model_saves_path)


print("Directory creation process complete")
  

sep = os.path.sep
adverse_filename = adverse_filelist_path.split(sep)[-1]
nominal_filename = nominal_filelist_path.split(sep)[-1]

#place the specified adverse and nominal filelists inside the parameters folder
copyfile(adverse_filelist_path,os.path.join(parameters_path,adverse_filename))
copyfile(nominal_filelist_path,os.path.join(parameters_path,nominal_filename))

#move the directory of the dataset folder

print("copying")
files = list_visible_files(dataset_folder_path)

for f in files:
    if f not in [adverse_filename, nominal_filename]:
        #copytree(os.path.join(dataset_folder_path,f),raw_data_path)
        copy_files_or_directories(os.path.join(dataset_folder_path,f),os.path.join(raw_data_path,f))  




#generate parameter_names.txt
#import one file list, grab the header, then make the files from said header
   

with open(os.path.join(parameters_path,"parameter_names.txt"),'w') as f:
    for parameter in parameter_list:
        f.write("{}\n".format(parameter))


#generate DTMIL_config.json and add the path 
    #grab code from the json generating ipynb
    
export_json_cfg(directory = dataset_path,
                fl_nominal=nominal_filename,
                fl_adverse=adverse_filename,
                param_names=parameter_list)


print("Dataset Formatting Process Completed")
        

