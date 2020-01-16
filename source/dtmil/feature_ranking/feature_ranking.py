#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:41:04 2019

@author: dweckler
"""


from dtmil.utilities import dual_sort, get_labels_from_indeces
from dtmil.visualizations import Visualizer


from typing import List
import numpy as np
from dtmil.prediction_data import Prediction_Data
from dtmil.data_container import DataContainer
from dtmil.model_container import ModelContainer
from dtmil.feature_ranking.ranking_window import Parameter_Score_Window
import pandas as pd

#%% Feature Ranking Class           
#TODO: ADD Placehodler data window padding options
class Feature_Ranking:

    def __init__(self, data_ID_list:List[int], myData:DataContainer, myModel:ModelContainer, standard_deviation_disturbances:List[int] = [-2,2]):
          
        self.data_ID_list: List[int] = data_ID_list
        self.myData: DataContainer = myData
        self.myModel: ModelContainer = myModel
        self.standard_deviation_disturbances: List[int] = standard_deviation_disturbances
        self.ranking_group_list: List[Ranking_Group] = []  
        
        #create a ranking group for each data ID.
        for i,idx in enumerate(data_ID_list):
            
            print (f" progress:{i+1}/{len(data_ID_list)}", end="\r")

            ranking_group = Ranking_Group(idx,standard_deviation_disturbances,self)
            self.ranking_group_list.append(ranking_group)
        
        print("\n")
            
    def get_ranking_scores(self, attribute_type = 'label', top_number_of_features:int = None):
        feature_scores_list = []
        
        for group in self.ranking_group_list:
            score_lists = group.all_ranking_scores
            #possibly add weights for rankings here
            for score_list in score_lists:
                feature_scores_list.append(score_list)
            
        #check to make sure we have at least one array
        if len(feature_scores_list) == 0:
               print("no feature scores!")
               return
        
        attributeIdx = self.ranking_group_list[0].parameter_list
        attributeSum = [sum(x)/len(feature_scores_list) for x in zip(*feature_scores_list)]
        sorted_sums, sorted_attributes = dual_sort(attributeSum,attributeIdx,reverse = True )
        
        
        if attribute_type == 'label':
            #TODO: replace this with 
            sorted_attributes = get_labels_from_indeces(sorted_attributes,self.myData.header)
        elif attribute_type == 'index':
            pass #since it's set to index by default
        else:
            print(f"invalid attribute type \"{attribute_type}\" specified, using \"index\" instead")
            
        if top_number_of_features is not None:
            sorted_sums = sorted_sums[:top_number_of_features]
            sorted_attributes = sorted_attributes[:top_number_of_features]

        return sorted_sums, sorted_attributes 
    
    #TODO: expand this and the function it calls
    def export_graphs(self, top_number_of_features:int = None):
        #"default is none"
        parameter_selection = None
        if top_number_of_features is not None:
            sorted_ranking_sums, sorted_ranking_attributes = self.get_ranking_scores("index",top_number_of_features)
            parameter_selection = sorted_ranking_attributes
            
        vis = Visualizer(self.myData,self.myModel)
        for feature_group in self.ranking_group_list:
            vis.visualize_ranking_data(feature_group, parameter_selection = parameter_selection)
            
    def batch_output(self):
        sorted_ranking_sums, sorted_ranking_attributes = self.get_ranking_scores("index",6)

        for feature_group in self.ranking_group_list:
            vis = Visualizer(self.myData,self.myModel,feature_group.data_ID)

            vis.special_ranking_visualization(sorted_ranking_attributes,sorted_ranking_sums)
            
        
#TODO: implement previous ranking features
class Ranking_Group:
    
    def __init__(self,data_ID:int,standard_deviation_disturbances:List[int],parent:Feature_Ranking):     
 
        self.data_ID:int = data_ID
        self.parent:Feature_Ranking = parent
        #self.default_feature:Sample_With_Disturbance = None
        self.parameter_list:List[int] = self.parent.myData.parameter_selection.tolist() 
        self.parameter_windows:List[Parameter_Score_Window]
        self.score_weights:List[float]
        
        self.prediction_data:Prediction_Data = Prediction_Data(self.parent.myData,self.parent.myModel,data_ID)
        self.__define_window_region()
        self.__generate_parameter_windows()
        
        
    def __define_window_region(self):
        precursor_scores = self.prediction_data.precursor_score
           
        #TODO: apply smoothing to the precursor scores for graphs that are not as consistent
        threshold_list= np.array([i>0.5 for i in precursor_scores])
        tl_padded = np.r_[False,threshold_list, False]
        # Get indices of shifts, which represent the start and stop indices
        shift_idx = np.flatnonzero(tl_padded[:-1] != tl_padded[1:])
        
        # Get the start and stop indeces for all the windows
        
        #TODO: end_idx goes out of bounds if the graph ends on the precursor score. This only impacts graphing (and marginally at that), but should be fixed eventually
        self.start_idx:List[int] = shift_idx[:-1:2]
        self.end_idx:List[int] = shift_idx[1::2] 
        
    def __generate_parameter_windows(self):
        
        self.parameter_windows = []
        for parameter in self.parameter_list:
            windows = Parameter_Score_Window(self.start_idx,self.end_idx,self, parameter)
            self.parameter_windows.append(windows)
            

        
    def display_ranking_scores(self, num_scores= None):
        
        parameter_response_windows_list = self.ordered_response_windows_list
        print(type)
        num_windows = len(parameter_response_windows_list)
        for index, _response_windows in enumerate(parameter_response_windows_list):
            
            print("Window {} of {}".format(index+1,num_windows))
             
            response_windows = _response_windows[:num_scores]
            
            scores = [window.ranking_score for window in response_windows]
            attribute_labels = [window.attribute_label for window in response_windows]

            df = pd.DataFrame(list(zip(attribute_labels,scores)),columns = ["Attribute", "Score"])
            print(df.to_string(index=False))
            print("\n")
    
#    @property 
#    def ranking_scores(self):
#        return [abs(window.most_negative_percent_diff) for window in self.parameter_windows]
    
    def top_response_windows(self,percent_cutoff = None):
        
        #TODO: add this to the config file 

        if percent_cutoff is None:
            percent_cutoff = 0.4

        print("percent cutoff",percent_cutoff)
        top_response_windows = []
        for response_windows in self.ordered_response_windows_list:
            
            sorted_scores = [window.ranking_score for window in response_windows]
            print("sorted score length",len(sorted_scores))
            
            score_sum = np.sum(sorted_scores)
            cutoff_sum = percent_cutoff*score_sum
            #print(cutoff_sum)
            partial_sum = 0
    
            
            cutoff_index = 0
            for index, score in enumerate(sorted_scores):     
                partial_sum += score
                if partial_sum >= cutoff_sum:
                    cutoff_index = index
                    break
                
            top_windows = response_windows[:cutoff_index]
            top_response_windows.append(top_windows)
            
        return top_response_windows
        
    
        
    @property
    def all_ranking_scores(self):
        rs = np.array([np.array(window.most_negative_percent_differences) for window in self.parameter_windows])
        
        return np.swapaxes(rs,0,1)
    
    
    @property
    def all_subwindows(self):
        sw =  np.array([np.array(window.subwindows) for window in self.parameter_windows])
        #swap the axes so we select by subwindow rather than parameter
        return np.swapaxes(sw,0,1)

    
    @property
    def ordered_response_windows_list(self):
        subwindows = self.all_subwindows
        
        parameter_windows_lists = []
        for param_windows in subwindows:
            #sort each set of subwindows by their ranking score
            sorted_parameter_windows_list = sorted(param_windows,key=lambda subwindow: subwindow.ranking_score,reverse = True)  
            
            parameter_windows_lists.append(sorted_parameter_windows_list)
        
        #return each subwindow list, ranked by the parameter response
        return parameter_windows_lists
        
    @property
    def window_count(self):
        return len(self.start_idx)
        

        


 
    