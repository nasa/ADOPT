#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Notices:

Copyright © 2019 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

Created on Fri Jul  6 12:11:30 2018

@author: dweckler
"""



from dtmil.visualizations import Visualizer, save_figure 
from dtmil.utilities import load_something

from dtmil.model_container import ModelContainer

from dtmil.configuration.config_dtmil import get_json_config_data

import os

import numpy as np
import time
import matplotlib.pyplot as plt
from dtmil.prediction_data import Prediction_Data

#visualize the data from a saved model
class Batch_Visualizer:
    
    def __init__(self,dataset_dir = None):
        print("reloading data")

        time_start = time.time()
        
        json_data_block = get_json_config_data(dataset_dir)
        
        json_dir_data, json_data,dataset_dir = json_data_block
    
        model_container_path = os.path.join(dataset_dir,json_dir_data['model_storage_directory'],json_data['model_io']["model_container_filename"])
        
        myModel = ModelContainer.reload_all_data(dataset_dir)
        
       
        #os.path.join(model_storage_directory,model_filename)
        
        self.dataset_dir = dataset_dir
        self.myModel = myModel
        self.myData = myModel.myData
        # self.myData = load_something(myModel.data_path)
        print("Total time to load in data: {}".format(time.time()-time_start))


    def save_precursor_graphs(self):
        self.get_precursor_graph(self.myData.I_bad,filename = "I_bad_precursors")
        self.get_precursor_graph(self.myData.I_bad_ho,filename = "I_bad_ho_precursors")
        self.get_precursor_graph(self.myData.I_bad_valid,filename = "I_bad_valid_precursors")
        self.get_precursor_graph(self.myData.I_opt,filename = "I_opt_precursors")
        self.get_precursor_graph(self.myData.I_opt_ho,filename = "I_opt_ho_precursors")
        self.get_precursor_graph(self.myData.I_opt_valid,filename = "I_opt_valid_precursors")
                      
    def __get_instance_probabilities(self,sample_idx):
    
        prediction_data = Prediction_Data(self.myData,self.myModel,sample_idx)
        return prediction_data.precursor_score

    def __gather_instance_probabilities(self,sample_group):
        sample_group_list = []
        for sample_idx in sample_group:
            L = self.__get_instance_probabilities(sample_idx)
            sample_group_list.append(L)
            
        return sample_group_list
    
          
    def get_precursor_graph(self,sample_group,filename = None ,figScale = 1.0):
        print("Getting precursor scores...")
        sorted_sample_group = np.sort(sample_group)
        precursorScores = self.__gather_instance_probabilities(sorted_sample_group)
        print("Generating plot...")
        
        numColumns = 10
        numRows = len(sorted_sample_group)//numColumns + (len(sorted_sample_group) % numColumns >0) #this is an integer divide that rounds up
        figsizeX = 25 *figScale
        figsizeY = 1.25*numRows * figScale
        
        fig, axs = plt.subplots(numRows,numColumns, figsize= (figsizeX,figsizeY))
        axs=axs.ravel()
        fig.tight_layout()
        xvec=np.arange((self.myData.maxlen-1)*0.25,-0.25,-0.25)
    
        lastRowIdx = (numRows-1)*numColumns-1
        
        for index,instance in enumerate(precursorScores):
            axs[index].plot(xvec,instance,'r',linewidth=2)
            axs[index].set_title("precursor score",fontsize=10)
            #axs[index].hold
            axs[index].set_ylim([0,1])
            axs[index].invert_xaxis()
            
            ##FIXME: fix this hard coded threshold when this function is cleaned up
            precursor_score_guideline = np.full(xvec.shape[0],0.5)
            axs[index].plot(xvec,precursor_score_guideline,'k--')    
        
            
            if index>lastRowIdx:
                #axs[index].set_xticks([0,10,20])
                #axs[index].set_xticklabels(['1000 ft \n altitude', '10 mi', '20 mi'],rotation=0)
                axs[index].set_xlabel('Precursor Timeline',fontsize=10)
            axs[index].set_title('{}'.format(sorted_sample_group[index]),fontsize=10)
    
        if (filename):
            
            save_figure(self.myModel,"",plt,"precursor_graphs",filename)
            
            #plt.savefig("{}.pdf".format(filename),format= "pdf") 
            
            
    def save_sample_parameters(self, sampleData, file_output_dir = "parameter_graphs", maxNumImages= None,file_output_type= "pdf", num_columns = 4): 
        count = 0
        vis = Visualizer(self.myData,self.myModel,sample_idx = -1, dataset_dir = self.dataset_dir)
        
        for sample_idx in sampleData:
            vis.current_sample = sample_idx
            count += 1
            vis.visualize_sample_parameters(file_output_dir= file_output_dir, saveFig = True,file_output_type=file_output_type,num_columns=num_columns)
            
            if(maxNumImages):
                if count >= maxNumImages:
                    break

