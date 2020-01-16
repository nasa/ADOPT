#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notices:

Copyright © 2019 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

Created on Tue Oct  1 14:48:34 2019

@author: dweckler
"""

import os 
from dtmil.configuration.config_dtmil import get_json_config_data
from time import time
from dtmil.feature_ranking.feature_ranking import Feature_Ranking 
from enum import Enum
import numpy as np
from dtmil.visualizations import Visualizer

from dtmil.model_container import ModelContainer


class Dataset_Type(Enum):
    Train = 1
    Validation = 2
    Test = 3
    

json_data_block = get_json_config_data()
json_dir_data, json_group_data,dataset_dir = json_data_block


model_storage_directory = os.path.join(dataset_dir,json_dir_data['model_storage_directory'])


visualization_parameters = json_group_data['visualization']
importing_parameters = json_group_data['importing']
model_io = json_group_data['model_io']


myModel = ModelContainer.reload_all_data(dataset_dir)
myData = myModel.myData


input_val = ""

while(input_val == ""):
    set_types = {1:"Train",2:"Validation",3:"Test"}


    prompt = ("Which part of the dataset would you like to visualize? If multiple, input the numbers separated by commas.\n\n" 
              "1. Training Set\n"
              "2. Validation Set\n"
              "3. Test Set\n"
              "\n")

    set_nums = input(prompt)
    input_val = set_nums
    
    if (input_val == ""):
        print("no input selected, try again, or press control-c to exit\n")
        
    else:
        sets_list = [int(num) for num in set_nums.split(',')]

dataset_types_list = [Dataset_Type(num) for num in sets_list]


num_parameters_to_rank = 5



prompt = ("Input the magnitude(s) of standard deviation responses to test. Separate with commas (eg, 1,2,-1,-2). \n"
          "Or press enter for the default value of 2,-2 \n")

disturbance_nums = input(prompt)

if disturbance_nums == "":
    sd_disturbances = [2,-2]
    
else:
    sd_disturbances = [int(num) for num in disturbance_nums.split(',')]



data_list = []

#print(dataset_types_list)
if(Dataset_Type.Train in dataset_types_list):
    
    data_list.append(myData.I_bad)
    
if(Dataset_Type.Validation in dataset_types_list):
    data_list.append(myData.I_bad_valid)
    
if(Dataset_Type.Test in dataset_types_list):
    data_list.append(myData.I_bad_ho)
    
data_list = np.concatenate(data_list)
    


print("Performing feature ranking analysis...")
start = time()
#ranking1 = New_FR([651,345,231],myData,myModel,standard_deviation_disturbances=[2,-2])
ranking1 = Feature_Ranking(data_list,myData,myModel,standard_deviation_disturbances=sd_disturbances)
end = time()

print(f"full_time  {end-start}")


print("saving figures")
#group = ranking1.ranking_group_list[0]

ranking_list = ranking1.ranking_group_list
vis=Visualizer(myData,myModel,input_json_data = json_group_data)
print("generating ranking data plots")

for group in ranking_list:
    vis.visualize_top_ranking_parameters(group,feature_num_limit= None,num_columns=5)


#save graphs to disk here
#save ranking output function to disk 
#eventually rewrite to map index to filename when outputting stuff

print("Done")
