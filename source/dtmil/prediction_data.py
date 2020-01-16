#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:55:10 2019

@author: dweckler
"""



import numpy as np, matplotlib.pyplot as plt
from keras import backend as T
import time
import os
from dtmil.utilities import flat_avg
from dtmil.model_container import ModelContainer
from dtmil.data_container import DataContainer

#%%class def


class Prediction_Data:
    
    def __init__(self,myData:DataContainer,myModel:ModelContainer,sample_id:int = None, data_padding:bool = False, input_window = None):
        
        self.myData = myData
        self.myModel = myModel
        self.current_sample = sample_id
        
        #FIXME: Figure out what shape the input window will be. For now, it just assumes the same shape as the data sample (two indeces: [time,feature])
        if input_window is not None:
            self.data_sample = input_window    
        else:
            #TODO: make states and states_orig have the same "shape order"
            #both the arrays below are the same shape
            if sample_id is None:
                sample_id = 0
                print(f"no value provided for sample_id, setting to default value of {sample_id}")
            self.data_sample = myData.states[sample_id,:,:] 
            
        self.data_length = len(self.data_sample)
        self.visualization_sample = myData.states_orig[:,sample_id,:]
        
        inst_layer_output_fn = T.function([myModel.model.layers[0].input],[myModel.model.layers[-2].output])
        self.instance_layer_output_function = inst_layer_output_fn
        
        if(data_padding):
            self.pad_data()    
            
            #self.pad_original_precursor_score()
            
        else:
            self.data_window = self.data_sample
            self.visualization_window = self.visualization_sample
            
            self.padded_sample = None
            self.padded_vis_sample = None
            
            self.start_index = 0
            self.end_index = self.data_length - 1 
            
        self.update_predictions()
        
    def update_predictions(self):
     
        data_window = self.data_window
        data_length = len(data_window)
        num_features = len(data_window[0])
        
        #TODO: get the states from myData if there isn't another type of input
        input_values=np.reshape(data_window,(1,data_length,num_features))
        self.input_values = input_values
        
        # get instance probabilities (precursor score)
        L=self.instance_layer_output_function([input_values])[0]
        self.L = L
        
        self.precursor_score = L[0,:,0]
        
        # get precursor indeces
        #FIXME: Make this work with updating visualization params, or let the visualization module take it
        self.precursor_threshold = self.myData.json_data['visualization']["precursor_threshold"]
        self.precursor_indeces=np.where(self.precursor_score>self.precursor_threshold)[0]  
        

  #This is only until we get actual streaming working        
    def update_data_window(self,step_size = 1):
        
        new_start_index = self.start_index + step_size
        end_index = new_start_index + self.data_length
        
        if end_index >= len(self.padded_sample):
            #array would be out of bounds so we set it to the last value
            end_index = len(self.padded_sample)
            #new_start_index = end_index - self.data_length +1
            new_start_index = end_index - self.data_length

    
        self.start_index = new_start_index
        self.data_window = self.padded_sample[new_start_index:end_index]
        self.visualization_window = self.padded_vis_sample[new_start_index:end_index]
        
        #self.orig_prec_score_window = self.padded_orig_prec_score[new_start_index:end_index]
        
        self.update_predictions()            
    
    #####TODO: Remove once demos are done
    
    def pad_data(self):
        
        data_sample = self.data_sample
        vis_sample = self.visualization_sample
        self.padded_sample, self.data_window = self.pad_sample(data_sample)
        self.padded_vis_sample, self.visualization_window = self.pad_sample(vis_sample)
        
        self.start_index = 0               
        
    def pad_sample(self, sample):
        data_length = self.data_length
        pad_left = np.stack([sample[0]]*data_length)
        pad_right = np.stack([sample[-1]]*data_length)
        
        padded_sample = np.concatenate((pad_left,sample,pad_right))
        start_index = 0
        #end_index = data_dlength - 1
        end_index = data_length
        
        data_window = padded_sample[start_index:end_index]
        
        return padded_sample, data_window
        


