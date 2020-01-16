#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:42:33 2019

@author: dweckler
"""


from typing import List
import numpy as np
from dtmil.prediction_data import Prediction_Data

class Parameter_Score_Window:
    
    def __init__(self, start_idx:List[int],end_idx:List[int], parent_group, disturbed_parameter:int):
        self.prediction_data:Prediction_Data = parent_group.prediction_data
        self.sd_disturbances:List[int] = parent_group.parent.standard_deviation_disturbances
        self.disturbed_parameter:int = disturbed_parameter
        self.start_indeces:List[int] = start_idx
        self.end_indeces:List[int] = end_idx
        self.modified_precursor_scores:List[float]
        self.subwindows:List[Precursor_Event_Window]
        self.parent_group = parent_group
        
        precursor_score = self.prediction_data.precursor_score
        window_count = len(start_idx)
        
        if window_count == 0:
            self.modified_precursor_scores = [self.prediction_data.precursor_score]
            window = Precursor_Event_Window(precursor_score,None,self)
            self.subwindows = [window]
            
        else:
            self.__disturb_parameters()
            
            subwindows = []
            for i in range(window_count):
                start = start_idx[i]
                end = end_idx[i]
                score_window = self.prediction_data.precursor_score[start:end]
                modified_windows = [window[start:end] for window in self.modified_precursor_scores]
                subwindows.append(Precursor_Event_Window(score_window,modified_windows,self))
                
            self.subwindows = subwindows
        
        
    def __disturb_parameters(self):
        
        param_list = self.prediction_data.myData.parameter_selection.tolist() 
        self.modified_precursor_scores = []
        
        for standard_deviation_scale in self.sd_disturbances:
            modified_input_data = np.copy(self.prediction_data.input_values)
    
            i = param_list.index(self.disturbed_parameter)
            
            singleFeature = modified_input_data[:,:,i]        
            standard_dev = np.std(singleFeature) * standard_deviation_scale
            singleFeature += standard_dev
            modified_input_data[:,:,i] = singleFeature

            L=self.prediction_data.instance_layer_output_function([modified_input_data])[0]
            modified_precursor_score = L[0,:,0].tolist()
            self.modified_precursor_scores.append(modified_precursor_score)
                
    @property
    def most_negative_percent_differences(self):
        return [abs(sw.most_negative_percent_diff) for sw in self.subwindows]
    
    @property
    def most_important_sd_responses(self):
        return [abs(sw.most_important_sd_response) for sw in self.subwindows]

    
       
class Precursor_Event_Window:
    
    def __init__(self,precursor_score_window:List[float],modified_score_windows:List[List[float]], parent_window:Parameter_Score_Window):
        self.precursor_score_window = precursor_score_window
        self.modified_score_windows = modified_score_windows
        self.parent_window = parent_window
        
        self.most_negative_percent_diff:float
        
        if modified_score_windows is None:
            self.most_negative_percent_diff = 0 
            self.most_important_sd_response = None

        else:
            self.__compare_precursor_scores()
        
    #complare all the scores with each SD disturbance to see which surpresses the precursor score the most
    def __compare_precursor_scores(self):
    
        percent_differences = []
        precursor_window = self.precursor_score_window
        
        if len(precursor_window == 1):
            integrate = np.mean     
        else:
            integrate = np.trapz
            
        avgDefault = integrate(precursor_window)
     
        for modified_window in self.modified_score_windows:                     
          
            avgCurrent = integrate(modified_window)            
                
            diff_percent = (avgDefault-avgCurrent)/(avgDefault)*100
            percent_differences.append(diff_percent)
            
        most_negative_diff = 0
        most_important_sd_response = None
        
        
        for i,percent_diff in enumerate(percent_differences):
            if percent_diff > most_negative_diff:
                most_negative_diff = percent_diff
                most_important_sd_response = self.parent_window.sd_disturbances[i]
                
        self.most_negative_percent_diff = most_negative_diff
        self.most_important_sd_response = most_important_sd_response
        
        
    @property
    def ranking_score(self):
        return self.most_negative_percent_diff
    
    @property
    def attribute_index(self):
        return self.parent_window.disturbed_parameter
    
    @property
    def attribute_label(self):
        return self.parent_window.prediction_data.myData.param_index_to_label(self.attribute_index)
        
        
    
