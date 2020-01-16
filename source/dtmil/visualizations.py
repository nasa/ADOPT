#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:55:46 2019

@author: dweckler
"""


import numpy as np, matplotlib.pyplot as plt
from keras import backend as T
import time
import os
from .utilities import flat_avg
from dtmil.configuration.config_dtmil import get_json_config_data
from .prediction_data import Prediction_Data
import math

 #%%class def

class Visualizer:
    
    #TODO: Redesign this to work with multiple sources without depending on having all the data at once
    def __init__(self, myData, myModel, sample_idx = None, guidelines = True, prediction_data = None, dataset_dir = None, input_json_data = None):
        
        self.myData = myData
        self.myModel = myModel
        self._current_sample = sample_idx
        
        ##FIXME: make this update the visualization parameters every run (grab location of config file from myData?) 
        
        if (input_json_data is not None):
            json_data = input_json_data
            
        else:
            _, json_data, _ = get_json_config_data(dataset_dir)
        
        
        
        self.visualization_params = json_data['visualization']
     
        ##FIXME: Make this more able to be manually defined
        sf = 0.25
        self.xvec_scale_factor = sf
        
        self.xvec_timeline=np.arange((self.myData.maxlen-1)*sf,-sf,-sf)
        
        #this is to account for the extra value in the start and end indeces. Will be best practice to fix in the future
        self.xvec_temp_time_lookup = np.copy(self.xvec_timeline)
        self.xvec_temp_time_lookup = np.append(self.xvec_temp_time_lookup,self.xvec_timeline[-1])
        
        

        if sample_idx == None:
            print(f"sample index is set to None, using default value")
            sample_idx = 0
            
        if prediction_data:
            self.prediction_data = prediction_data
        else:
            self.prediction_data = Prediction_Data(myData,myModel,sample_idx)
            
        self.guidelines = guidelines  
        if (guidelines):
            self.get_guidelines()
        
    @classmethod
    def frompredictiondata(cls, prediction_data, guidelines = True):
        #initialize from preditcion data
        
        return cls(prediction_data.myData, prediction_data.myModel, prediction_data.current_sample, prediction_data = prediction_data)
 
    #%%plot sample timeline function

    @property
    def current_sample(self):
        return self._current_sample
    
    @current_sample.setter
    def current_sample(self,value):
        self._current_sample = value
        self.prediction_data = Prediction_Data(self.myData,self.myModel,value)
    
    def plot_sample_timeline(self, figure_size = None, saveFig = True):
        
        myModel = self.myModel
        model_output_directory = myModel.model_output_directory
        xtest =  myModel.xtest
        
        if (saveFig):
            plt.switch_backend('agg')
            
        # function to get an intermediate layer's output (instance probabilities)
        inst_layer_output_fn = T.function([myModel.model.layers[0].input],[myModel.model.layers[-2].output])
        
        temp=xtest
        L=inst_layer_output_fn([temp])[0]
        nex=int(temp.shape[0]/2)
        
        plt.figure(figsize=figure_size)
        plt.subplot(2,1,1)
        plt.plot(np.transpose(L[:nex,:,0]),'g')
        plt.ylim([-0.1,1.1])
        #plt.xlabel('Time to adverse event',fontsize=14)
        #plt.xlabel('Sample timeline',fontsize=14)
        plt.ylabel('Probability of \n adverse event',fontsize=14)
       # plt.xticks([0,10,20],['1000 ft \n altitude', '10 mi', '20 mi'],rotation=0)
        #plt.gca().invert_xaxis()
        plt.subplot(2,1,2)
        plt.plot(np.transpose(L[nex:,:,0]),'r')
        plt.ylim([-0.1,1.1])
        #plt.gca().invert_xaxis()
        plt.xlabel('sample timeline',fontsize=14)
        #plt.xticks([0,10,20],['1000 ft \n altitude', '10 mi', '20 mi'],rotation=0)
        plt.ylabel('Probability of \n adverse event',fontsize=14)
        
        temp=self.myData.xvalid
        L=inst_layer_output_fn([temp])[0]
        nex=int(temp.shape[0]/2)
        np.where(L[nex:,80:,0]>0.5)[0][:10]
        
        if(saveFig):
            plt.savefig(os.path.join(model_output_directory,"timeline.png"))

    #%%batch visualization function
    #FIXME: text sizing
    def visualize_sample_parameters(self,figure_size = None, saveFig = False, file_output_dir = "",file_output_type = "pdf",num_columns = 5, subplot_aspect_ratio = (1,1), subplot_size = 3.6):
        myData = self.myData
 #       myModel = self.myModel
    
    
        if (saveFig):
            plt.switch_backend('agg')
    
        #specify the variables to be included in the plot
        correlated_states = myData.correlated_states.tolist()
        trained_states = myData.parameter_selection.tolist()
        parameters_to_plot=correlated_states + trained_states 
        correlated_indeces = len(correlated_states)    
        
        num_plots = len(parameters_to_plot) + 1
        num_rows = math.ceil(float(num_plots)/float(num_columns))
        
        if figure_size is None:
            width = 4*num_columns
            height = num_rows * 3.5
            
            figure_size = (width,height)
    

        fig, axs = plt.subplots(num_rows,num_columns, figsize= figure_size)
        axs=axs.ravel()
        
        starting_index = -1-myData.maxlen+1
        
        for pltIdx in np.arange(len(parameters_to_plot)):
            selected_parameter = parameters_to_plot[pltIdx]
            
            plot_title = "{}".format(myData.header[selected_parameter])
             #add holdout to the title if it's within the correlated indeces
            if (pltIdx < correlated_indeces):
                plot_title = plot_title + "(H/O)"
            
            self.plot_parameter(selected_parameter,axs[pltIdx],starting_index, plot_title = plot_title)
            
        # plot precursor score in a separate subplot
        pltIdx=pltIdx+1
        self.plot_precursor_score(axs[pltIdx],'Precursor Score')        
        fig.tight_layout()
        
        # save figure if needed
        if saveFig:
            
            suffix = "_{}".format(self.myData.get_filename(self.current_sample))

            file_label, file_dataset_type = self.myData.get_grouping(self.current_sample)
            
            filename = "{}_{}".format(file_label,file_dataset_type)
            
            save_figure(self.myModel,suffix,fig,file_output_dir,filename,file_output_type = 'pdf')
            #self.save_figure(fig,file_output_dir)
            

    
    def special_ranking_visualization(self, states_to_visualize,sorted_ranking_sums,figure_size = (10,10), saveFig = False, file_output_dir = "",file_output_type = "pdf"):
        myData = self.myData
        
        fig, axs = plt.subplots(3,3, figsize= figure_size)
        axs=axs.ravel()
        
        self.plot_precursor_score(axs[1],'Precursor Score')        
        
        for i in range(6):
            selected_parameter = states_to_visualize[i]
            
            plot_title = "{} ({})".format(myData.header[selected_parameter],sorted_ranking_sums[i])
             #add holdout to the title if it's within the correlated indeces
          
            self.plot_parameter(selected_parameter,axs[i+3],0, plot_title = plot_title)
    
    
    
    #TODO: same as below except ordered ranking parameters with a variable number of columns and such
    #output with values of ranking
    #figure out what the values mean to report to bryan tomorrow 
    def visualize_top_ranking_parameters(self,ranking_group,feature_num_limit=None,num_columns = 4,displayfig = False):
        
        file_output_dir = "feature_ranking"
        myData = self.myData
        
        if (not displayfig):
            plt.switch_backend('agg')
            
        #get as many as we can
        #score_pair_lists = ranking_group.top_ranking_scores(1)
        
        #response_windows_lists = ranking_group.top_response_windows(1)
        response_windows_lists = ranking_group.ordered_response_windows_list
        
        if(feature_num_limit is not None):     
            if len(response_windows_lists[0])> feature_num_limit:
                response_windows_lists = [lst[0:feature_num_limit] for lst in response_windows_lists]
                
        num_windows = len(response_windows_lists)
        #print(feature_num_limit,len(response_windows_lists[0]),len(response_windows_lists[1]))
        
        for idx,response_windows in enumerate(response_windows_lists):
            
            parameter_selection = [window.attribute_index for window in response_windows]
            
#            print([window.ranking_score for window in response_windows])
#            print([window.most_important_sd_response for window in response_windows])
            score_list = [round(window.ranking_score,3) for window in response_windows]
            
            sd_response_list = []
            for window in response_windows:
                most_important_response = window.most_important_sd_response
                if most_important_response is not None:
                    sd_response_list.append(str(most_important_response))
                else:
                    sd_response_list.append("n/a")
            
            #sd_response_list = [round(window.most_important_sd_response,3) for window in response_windows]

            
            num_plots = len(response_windows) + 1
            num_rows = math.ceil(float(num_plots)/float(num_columns))            
            
            width = 4*num_columns
            height = num_rows * 3.5
            
            figsize = (width,height)
            fig, axs = plt.subplots(num_rows,num_columns, figsize= figsize)
            
            axs=axs.ravel()
            fig.tight_layout()
            
            xvec_timeline = self.xvec_timeline 
            plot_idx = 0
            
            axs[plot_idx].plot(xvec_timeline,ranking_group.prediction_data.precursor_score,'r',linewidth=2,label = "Default")
            axs[plot_idx].set_title("Precursor Score",fontsize=10)
            axs[plot_idx].set_ylim([0,1])
            axs[plot_idx].invert_xaxis()
            
            if(self.guidelines):
                axs[plot_idx].plot(self.xvec_timeline,self.precursor_score_guideline,'k--')    
        
            graph_colors = ['b','g','k','y','c','m','k','w']
            color_idx = 0
            
            sd_disturbances = ranking_group.parent.standard_deviation_disturbances
            
            #TODO: condense everything below into one function (rather than writing the same code twice)
            parameter_window_indeces = [ranking_group.parameter_list.index(i) for i in parameter_selection]
            parameter_windows = [ranking_group.parameter_windows[i] for i in parameter_window_indeces]
            
            #if this process isn't behind an if statement, the algorithm will output blank graphs
            #furthermore, it will cause some of the following graphs to come out blank as well
            #the cause of this is unknown, but may be useful to investigate in the future
            if len(parameter_windows)>0:        
                
                #TODO: Figure out why this conditional became necessary and the one above stopped working? (maybe some revisions impacted it?)
                if len(parameter_windows[0].start_indeces)>0:
                    
                    start_index = parameter_windows[0].start_indeces[idx]
                    end_index = parameter_windows[0].end_indeces[idx]
    
                    window_start_idx = self.xvec_temp_time_lookup[start_index]
                    window_end_idx = self.xvec_temp_time_lookup[end_index]
                    
                    axs[plot_idx].axvspan(window_start_idx, window_end_idx, alpha=0.1, color='k')
                    for index,window in enumerate(parameter_windows):
                        color_idx = 0
                        plot_idx = index+1
                        
                        axs[plot_idx].invert_xaxis()
                        #axs[plot_idx].set(adjustable='box', aspect=1)
                        axs[plot_idx].plot(xvec_timeline,ranking_group.prediction_data.precursor_score,'r', label = "Default",linewidth=2)
                        axs[plot_idx].axvspan(window_start_idx, window_end_idx, alpha=0.1, color='k')
    
            
                        for precursor_score in window.modified_precursor_scores:
                            selected_parameter = parameter_selection[index]
                            
                            disturbance = sd_disturbances[color_idx]
                            
                            if disturbance > 0:
                                label = "+ {} σ response".format(disturbance)
                                
                            else:
                                label = "- {} σ response".format(abs(disturbance))
    
    
                            axs[plot_idx].plot(xvec_timeline,precursor_score,graph_colors[color_idx],linewidth=2,label = label)
                            axs[plot_idx].set_title("{} \n({}, {} σ response)".format(myData.header[selected_parameter],score_list[index],sd_response_list[index]),fontsize=10)
                            axs[plot_idx].set_ylim([0,1])
                            if(self.guidelines):
                                axs[plot_idx].plot(self.xvec_timeline,self.precursor_score_guideline,'k--')    
                            color_idx += 1                    
                    
                    if(plot_idx>1):
                        handles, labels = axs[plot_idx].get_legend_handles_labels()
                        fig.legend(handles, labels, loc='lower right')
                            
                    #save the figure
                    plt.tight_layout()
    
                    file_label, file_dataset_type = self.myData.get_grouping(ranking_group.data_ID)
                    filename = "{}_{}_ranking".format(file_label,file_dataset_type)
                    
                    suffix = "_{}".format(self.myData.get_filename(ranking_group.data_ID))
                    
                    if num_windows > 1:
                        suffix = "{}_precursor_event_{}".format(suffix,idx)
                    
                    save_figure(self.myModel,suffix,fig,file_output_dir,filename,output_time = False)
                    
                else:
                    #TODO: 
                    print("Precursor score for {} does not cross threshold?".format(self.myData.get_filename(ranking_group.data_ID)))
                
            else:
                print("Precursor score for {} does not cross threshold!".format(self.myData.get_filename(ranking_group.data_ID)))
            
    
#    def visualize_ranking_data(self,ranking_group, output_file = None, parameter_selection = None, num_columns = 7, subplot_aspect_ratio = (1,1), subplot_size = 3.6):
#        myData = self.myData
#        print("generating ranking data plot")
#        
#        if parameter_selection is None:
#            parameter_selection = myData.parameter_selection.tolist()
#
#        #all the paramaeters plus the precursor score in its own plot
#        num_plots = len(parameter_selection) + 1
#        num_rows = math.ceil(float(num_plots)/float(num_columns))
#        dx, dy = subplot_aspect_ratio
#        figsize = plt.figaspect(float(dy * num_rows) / float(dx * num_columns)) * subplot_size
#        
#        fig, axs = plt.subplots(num_rows,num_columns, figsize= figsize)
#        #fig, axs = plt.subplots(numRows,numColumns)
#        axs=axs.ravel()
#        fig.tight_layout()
#        #xvec_timeline=np.arange((myData.maxlen-1)*0.25,-0.25,-0.25)
#        
#        xvec_timeline = self.xvec_timeline 
#        
#        axs[0].plot(xvec_timeline,ranking_group.prediction_data.precursor_score,'r',linewidth=2)
#        axs[0].set_title("Normal",fontsize=10)
#        axs[0].set_ylim([0,1])
#        axs[0].invert_xaxis()
#        
#        graph_colors = ['b','g','k','y']
#        color_idx = 0
#        
#        parameter_window_indeces = [ranking_group.parameter_list.index(i) for i in parameter_selection]
#        parameter_windows = [ranking_group.parameter_windows[i] for i in parameter_window_indeces]
#        
#        for index,window in enumerate(parameter_windows):
#            color_idx = 0
#            plot_idx = index+1
#            axs[plot_idx].invert_xaxis()
#
#            for precursor_score in window.modified_precursor_scores:
#                selected_parameter = parameter_selection[index]
#                
#                axs[plot_idx].plot(xvec_timeline,precursor_score,graph_colors[color_idx],linewidth=2)
#                axs[plot_idx].set_title("{} ({})".format(myData.header[selected_parameter],selected_parameter),fontsize=10)
#                axs[plot_idx].set_ylim([0,1])
#                axs[plot_idx].plot(xvec_timeline,ranking_group.prediction_data.precursor_score,'r',linewidth=1)
#                color_idx += 1

        
    #%%save figure

    def save_figure(self, fig,file_output_dir,file_output_type = 'pdf'):
        
        save_figure(self.myModel,self.current_sample,fig,file_output_dir,"parameters_graph",file_output_type = 'pdf')

    
        #%%plot precursor score

    def plot_precursor_score(self, plot_axis, plot_title = "Precursor Score", start_index = None, end_index = None):
        precursor_score = self.prediction_data.precursor_score     
        plot_axis.plot(self.xvec_timeline[start_index:end_index], precursor_score[start_index:end_index],'r',linewidth=2)     
        
        if(self.guidelines):
            plot_axis.plot(self.xvec_timeline[start_index:end_index],self.precursor_score_guideline[start_index:end_index],'k--')    
        
        plot_axis.invert_xaxis()
        plot_axis.set_title(plot_title,fontsize=10)
        plot_axis.set_ylim([0,1])
        
    
            #%%plot indivudual parameter

    def plot_parameter(self, selected_parameter, plot_axis,starting_index = 0,end_index = None,plot_title = "", precIdx = None):
     
        ##FIXME: Make this more able to be manually defined
            xvec_timeline=self.xvec_timeline
             
            #FIXME: Make Prediction Data update states_orig ("visualization_sample")
            parameter_values = self.prediction_data.visualization_window[starting_index:end_index,selected_parameter]
          
            # plot time series variable 
            plot_axis.plot(xvec_timeline[starting_index:end_index],parameter_values,linewidth=2)
            
            ##plot the guidelines
            # if discrete variable, use discrete nominal data as guideline, else use continuous nominal data
            if selected_parameter in self.visualization_params["binary_parameters"]: 
                plot_axis.plot(xvec_timeline[starting_index:end_index],self.discrete_nominal_guideline[starting_index:end_index,selected_parameter],'k--',linewidth=2)
                plot_axis.set_ylim([-0.1,1.1])
            else:
                plot_axis.plot(xvec_timeline[starting_index:end_index],self.nominal_guideline[0,starting_index:end_index,selected_parameter],'k--',linewidth=2)
                plot_axis.plot(xvec_timeline[starting_index:end_index],self.nominal_guideline[1,starting_index:end_index,selected_parameter],'k--',linewidth=2)
            
            ##use this if we are dealing with multiple precursor score predictions, otherwise use the one genereated upon class initialization
            if (precIdx):
                precursor_indeces = precIdx
            else:
                precursor_indeces = self.prediction_data.precursor_indeces
            
            # plot precursor time instants as an overlay
            if len(precursor_indeces)>0:
                
                precursor_overlay_values = self.prediction_data.visualization_window[precursor_indeces,selected_parameter]
                
                self.precursor_overlay_values = precursor_overlay_values
                if(end_index):
                    if end_index >= precursor_indeces[0]:
                        precursor_end_index = (np.abs(precursor_indeces - (end_index))).argmin()
                        print(precursor_end_index,end_index)
                        
                        plot_axis.plot(xvec_timeline[precursor_indeces][0:precursor_end_index],precursor_overlay_values[0:precursor_end_index],'ro', alpha = 0.4)        
                else:
                    plot_axis.plot(xvec_timeline[precursor_indeces],precursor_overlay_values,'ro', alpha = 0.4)
                    
#                    
            if plot_title == "":
                plot_title = "{} ({})".format(self.myData.header[selected_parameter],selected_parameter)
            
            plot_axis.set_title(plot_title,fontsize=10)
           
#            # invert x-axis so that distance to touchdown reduces as we go towards rightside of the plot
            plot_axis.invert_xaxis()
    
    #%%get guidelines

    def get_guidelines(self):
        myData = self.myData
        optimal_values=myData.states_orig[:,np.concatenate((myData.I_opt,myData.I_opt_valid),axis=0)]
                #determine guidelines
        guideline_type = self.visualization_params["guideline_type"]
        if guideline_type == 1:
            optimal_standard_dev = np.std(optimal_values, axis=1)
            optimal_mean = np.mean(optimal_values,axis = 1)
            
            avg_guideline =flat_avg(optimal_mean)
            sdev_guideline = flat_avg(optimal_standard_dev)
                
            sdev_scale = 2.5
            upper_guideline = avg_guideline + sdev_scale * sdev_guideline
            lower_guideline = avg_guideline - sdev_scale * sdev_guideline
            nominal_guideline = np.array([lower_guideline, upper_guideline])       
        else:
            # get nominal percentiles for plotting
            nominal_guideline=np.percentile(optimal_values,[10,90],axis=1)
            
        self.nominal_guideline = nominal_guideline
        # Get nominal values assuming binary (note that we will only use this if the variable is binary)
        self.discrete_nominal_guideline=np.mean(optimal_values,axis=1)
        self.precursor_score_guideline = np.full(optimal_values.shape[0],self.prediction_data.precursor_threshold)
        



def save_figure(myModel, figure_suffix, fig,file_output_dir,filename,file_output_type = 'pdf', output_time = True):
    time_start = time.time()
    print("Saving figure: {}".format(figure_suffix))
    model_output_directory = myModel.model_output_directory

    if model_output_directory != "":
        model_output_directory = os.path.join(model_output_directory,file_output_dir)
        if not os.path.exists(model_output_directory):
            print(f"creating directory {model_output_directory}")
            os.makedirs(model_output_directory)
    
    
    
    filename = "{}{}.{}".format(filename,figure_suffix,"pdf")
    filepath = os.path.join(model_output_directory,filename)
    
    #print("Saving figure: {}".format(filepath))

    fig.savefig(filepath,format= file_output_type)

#    if(output_time):
#        print("Total time to save figure: {}".format(time.time()-time_start))

def visualize(myData, myModel,sample_idx = 0, savefig = False):

    vis = Visualizer(myData,myModel,sample_idx)
    
    vis.plot_sample_timeline(figure_size = (8,6), saveFig = savefig)
    
    print("Visualizing Sample {}".format(sample_idx))
    vis.visualize_sample_parameters(figure_size=(32,24),saveFig = savefig)
    


