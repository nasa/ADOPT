
'''

Notices:

Copyright © 2019 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.s

@author: vjanakir
This is the code for deep temporal multiple instance learning (DTMIL). This is the version of ADOPT that is based on deep learning.
The code assumes Keras with Theano or Tensorflow backend.
uses Anaconda virtual env with Python 2.7 and keras. It should also work in Python 3.x but not tested.

Code updated by dweckler
'''


import os
import numpy as np
import time
from sys import argv

from dtmil.data_container import DataContainer
from dtmil.model_container import ModelContainer
from dtmil.configuration.config_dtmil import get_json_config_data
from types import SimpleNamespace



seed=0
np.random.seed(seed)

#print(argv)
#%% user defined variables

if (len(argv) > 1):
    dataset_cfg_file_dir = argv[1] 
    
else:
    dataset_cfg_file_dir = None
    
#print(dataset_cfg_file_dir)

json_data_block = get_json_config_data(dataset_cfg_file_dir,new_run=True)

json_dir_data, json_group_data,dataset_dir = json_data_block

#TODO: Add automatically updating JSON file here
    #get current version
    #check hold parameter
    #increment version
    #make sure to use this to update the file storage path in the code (to include the new counting parameter appended to the data_id)




cfg = SimpleNamespace(**json_group_data)
cfg_dir = SimpleNamespace(**json_dir_data)

model_storage_directory = os.path.join(dataset_dir,cfg_dir.model_storage_directory)

train_flag= cfg.training['train_flag'] # 0 to use a pretrained model, 1 to create a new model
state_cache = cfg.importing['state_cache'] # 0 to use the cache, 1 to load from csvs
pre_trained_model = cfg.training['pre_trained_model']
pre_trained_json = cfg.training['pre_trained_json']


 #%% load data and preprocess
time_start = time.time()

myData = DataContainer(json_data_block, state_cache=state_cache)
print("Total time to load in data: {}\n".format(time.time()-time_start))
myData.preprocess()

#%% model build
myModel = ModelContainer(myData)
myModel.configure_model(train_flag,pre_trained_model,pre_trained_json)
myModel.train_model(train_flag)
myModel.evaluate_model()


print("Saving model to: {}".format(os.path.abspath(model_storage_directory)))
myModel.save_model()


#%% end
