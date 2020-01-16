#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: vjanakir
This is the code for deep temporal multiple instance learning (DTMIL). This is the version of ADOPT that is based on deep learning.
The code assumes Keras with Theano or Tensorflow backend.
uses Anaconda virtual env with Python 2.7 and keras. It should also work in Python 3.x but not tested.

Created on Tue Jun 19 14:44:18 2018

@author: dweckler

"""

from __future__ import print_function

import os
import sys
import numpy as np
import time
import h5py
import random
import pandas as pd
from pathlib import Path

##FIXME: This does not work properly for values other than 10%, the cause is currently unknown

def holdout_split(indeces, holdout_percent=10):
  
    full_arr = np.asarray(indeces)

    if holdout_percent == 0:
        full_arr = np.asarray(indeces)
        full_arr.shape = (1,len(indeces)) 
        
        return full_arr, np.asarray([])

    holdout_num = int(len(indeces)*holdout_percent)        
    ho_idx = indeces[-holdout_num:]
    main_idx = indeces[:len(indeces)-holdout_num]
        
    main_arr = np.asarray(main_idx)
    main_arr.shape = (1,len(main_idx))
    
    ho_arr = np.asarray(ho_idx)
    ho_arr.shape = (1,len(ho_idx))
    
    return main_arr[0], ho_arr[0]
    

class DataContainer:

    def __init__(self, json_config_data, state_cache = False):
        
        
        
        self.json_dir_data, self.json_data,self.dataset_dir = json_config_data
        self.preprocessing_params = self.json_data["preprocessing"]
        self.importing_params = self.json_data["importing"]

        self.parameters_directory = os.path.join(self.dataset_dir,self.json_dir_data['parameters_directory'])
        self.raw_data_directory = os.path.join(self.dataset_dir, self.json_dir_data['raw_data_directory'])
        
        self.load_data(state_cache)
    

                
    def load_data(self,state_cache):
     
 #       print('Loading data from {} CSV files...'.format(maxItemsInList))
        time_start = time.time()
        self.sample_list = []
        self.dropped_sample_filenames = []
        self.seqlabels = []
        
        nominal_filename = self.importing_params["nominal_filename"]
        adverse_filename = self.importing_params["adverse_filename"]
        
        #read file list
        read_lines_from_file = self.read_lines_from_file
        nominal_file_list = read_lines_from_file(nominal_filename)
        adverse_file_list = read_lines_from_file(adverse_filename)

        #the csv from which we get our default header
        default_csv_filename = os.path.join(self.raw_data_directory,nominal_file_list[0])
        df = pd.read_csv(default_csv_filename)
        parameter_list = list(df.columns.values)
        self.header = parameter_list
        self.all_parameter_names = parameter_list  
       
        self.preprocessing_params["all_parameter_names"] = parameter_list
        set_sample_length = self.preprocessing_params["set_sample_length"]
        if set_sample_length == None:
            self.max_seqlen = len(df)
            print("No sample length specified, assuming all samples are of equal length {}".format(self.max_seqlen))

        else:
            self.max_seqlen = set_sample_length              


        self.mismatched_files = []
        
        I_opt_idx,nominal_imported_csvs = self.__import_sample_list(nominal_file_list,label = 0)
        I_bad_idx,adverse_imported_csvs = self.__import_sample_list(adverse_file_list, label = 1)
        
        
        if len(self.mismatched_files)> 0:
            out = ("{}/{} labels don't match the default csv header. This will either cause a shaping error (and subsequent crash), " 
                   "or cause some parameters to be labeled incorrectly (possibly leading to nonsensical data). Make sure to double check the headers "
                   "for your csv files to make sure they all match".format(len(self.mismatched_files),len(I_bad_idx)+len(I_opt_idx)))
            print("\n\nMismatched CSV headers found!")
            
            print("Default CSV (used for comparison):",default_csv_filename)
            print("{}/{} mismatched csv files (for reference):".format(len(self.mismatched_files[:5]),len(self.mismatched_files)))
            print(self.mismatched_files[:5])
            #save to model output
            
            
            print(out)
            
            choice = input("Are you sure you want to continue? (y/n)\n")
            
            if choice == 'y':
                pass
            else:
                sys.exit(0)
            
            
        
        
        random.Random(42).shuffle(I_opt_idx)
        random.Random(42).shuffle(I_bad_idx)
        
        #FIXME:this shouldn't need processing here, do it in the holdout split function
        holdout_percent = self.importing_params['holdout_percent']
        
        #self.temp_I_opt_idx = I_opt_idx
        self.I_bad,self.I_bad_ho = holdout_split(I_bad_idx, holdout_percent)
        self.I_opt,self.I_opt_ho = holdout_split(I_opt_idx, holdout_percent)
        
        
        print("Dropped {} files that were too short".format(len(self.dropped_sample_filenames)))
        
        #this is just in case we come up with an algorithm that can handle differing sequence lengths
        self.seqLabels = np.asarray([self.seqlabels])[0] 

        finalList = np.asarray(nominal_imported_csvs + adverse_imported_csvs)
        finalArray = np.swapaxes(finalList,0,1)
        del finalList
        
        #splice percentage
        time_splice = self.importing_params["time_splice"]
        self.time_splice = time_splice
        
        if ((time_splice > 0) and (time_splice<1)):
        
            sample_index = 0 
            set_slice = int(finalArray.shape[sample_index]*time_splice)
            #self.seqlen[0,:] = set_slice
            finalArray = finalArray[0:set_slice,:,:]
            self.time_splice = time_splice
        else:
            self.time_splice = None
          
        print("saving sample_list")
        self.states_orig = finalArray
        self.states = finalArray 
        self.save_to_cache()
        
        print("Time to load: {} seconds".format(time.time()-time_start))       
        
        
    def __import_sample_list(self,X_file_list,label):
        I_X_idx = []
        
        imported_csv_list = []
        for filename in X_file_list:
            #TODO load,verify, and filter data here
                #throw an error/exception if lengths do not match and filtering is not set in dtmil_config
            
            imported_csv = self.import_sample(filename)
            
            if imported_csv is not None:
                imported_csv_list.append(imported_csv)
                self.sample_list.append(filename)
                I_X_idx.append(len(self.seqlabels))
                self.seqlabels.append(label)
                
        return  I_X_idx,imported_csv_list
        
     
    def import_sample(self,filename):
        
        filepath = os.path.join(self.raw_data_directory,filename)
        
        
        df = pd.read_csv(filepath)
        header = list(df.columns.values)
        
        
        if (header != self.header):
            self.mismatched_files.append(filename)
                
        imported_csv = df.values[-self.max_seqlen:]

        if len(imported_csv) != self.max_seqlen:
            self.dropped_sample_filenames.append(filename)
            return None
        
        return imported_csv    

        
    #FIXME: Make this actually work, maybe skip the whole my_data creation process?
    #load files from cache with CSV as backup if the cache isn't there
    def load_from_cache(self,backup_sample_list):
        print('Loading states from cache...')
        cache_dir = os.path.join(self.dataset_dir,self.json_dir_data['cache_file'])
        
        try:
            with h5py.File(cache_dir, 'r') as hf:
                self.states_orig = hf['states_orig'][:]
                self.states = hf['states_orig'][:]
                
                #TODO: have this in a different place? Also check to see if loading from cache breaks anything
                #splice percentage
                time_splice = self.importing_params["time_splice"]
                self.time_splice = time_splice
                
        except EnvironmentError:
            print('cache file not found, loading from CSV files instead')
            self.import_all_samples(backup_sample_list)
        
        
    def save_to_cache(self):
        print('saving states to cache...')
        cache_dir = os.path.join(self.dataset_dir,self.json_dir_data['cache_file'])

        with h5py.File(cache_dir, 'w') as hf:
            hf.create_dataset('states_orig', data = self.states_orig)
            

        
        
    def reshape_and_process(self):
        # after loading, the variables have the following shape
        # states_orig is of shape (T, N, D) where T is max length of sample, N is the total number of samples, D is the number of time series in each sample.
        # The length of sample i is given by seqlen[0,i]. 
        # If length is less than T, the sample data is prepended with NAN to make it length T.
        # A sample i belongs to "opt" (or "bad") if i belongs to array I_opt (or I_bad).
        # I_opt_ho and I_bad_ho are hold-out "test" sets.
        # I_bad, I_opt, I_bad_ho, I_opt_ho are one-dimensional arrays
        # seqlen is of shape (1, N). 
        # header is a list of D feature names
        # seqLabels is of shape (N,). sample i has a label seqLabels[i] - 1 if sample i has adverse event and 0 otherwise.        
        # removing variables which are correlated with target (to avoid finding trivial precursors)
                  
        correlated_states = self.preprocessing_params["redundant_parameters"]
        correlated_states = [self.decode_parameter_label(i) for i in correlated_states]
                
        #convert to a Numpy array that avoids redundant choices (just in case something was mistakenly added)
        self.correlated_states =np.unique(np.array(correlated_states))
        
        dropped_states = self.preprocessing_params["drop_parameters"]
        dropped_states = [self.decode_parameter_label(i) for i in dropped_states]
        self.dropped_states = np.unique(np.array(dropped_states))
        
        #make sure not to delete the same state twiceo
        states_to_remove = np.unique(np.array(correlated_states + dropped_states))
        
        self.parameter_selection=np.delete(np.arange(self.states.shape[2]),states_to_remove,0)
        self.states=self.states[:,:,self.parameter_selection]
        
        # get max length of trjectories
        self.maxlen = np.shape(self.states)[0]
        
        # get total number of trajectories
        #        Ntraj= np.shape(self.states)[1]
        
        # number of features (time series variables)
        self.nfeat=np.shape(self.states)[-1]
        
        # center the data - subtract mean and divide by STD. If variable is constant, remove it from analysis
        temp=np.reshape(self.states,(np.shape(self.states)[0]*np.shape(self.states)[1],np.shape(self.states)[2]))

        mean=np.nanmean(temp,0)
        std=np.nanstd(temp,0)
        elimidx=np.where(std<1E-5)[0]
        if elimidx.shape[0]>0:
            selidx=np.array(list(set(np.arange(self.nfeat).tolist()).difference(elimidx)))
            self.states=self.states[:,:,selidx]
            mean=mean[selidx]
            std=std[selidx]
            temp=temp[:,selidx]
            self.parameter_selection=self.parameter_selection[selidx]
        temp=(temp-mean)/std
        self.states=np.reshape(temp,(np.shape(self.states)[0],np.shape(self.states)[1],np.shape(self.states)[2]))
        del temp
        self.nfeat=np.shape(self.states)[-1]
        
        # Replace NAN by an arbitrary mask_val
        mask_val=int(np.nanmax(self.states)+1000)
        self.states[np.isnan(self.states)]=mask_val
        
        # reshape to match keras' definitions
        self.states=np.transpose(self.states,(1,0,2))
        
        
    def train_test_split(self):
        # Split train data into train (60%) and validation (40%) sets
        #FIXME: maybe change this to use a more traditional validation set approach. The numbers don't match the output for some reason
        
        validation_percent = self.importing_params["validation_percent"]
        self.validation_percent = validation_percent *100
        
        nvalid=int(validation_percent*len(self.I_bad)) 
        self.I_bad_valid=self.I_bad[len(self.I_bad)-nvalid:]
        self.I_bad=self.I_bad[:len(self.I_bad)-nvalid]
        self.I_opt_valid=self.I_opt[len(self.I_opt)-nvalid:]
        self.I_opt=self.I_opt[:len(self.I_opt)-nvalid]
       
        print(self.states.shape)
        temp=np.array([self.I_opt.tolist()+self.I_bad.tolist()])[0]
        
        self.xtrain=self.states[temp,:,:]
        self.ytrain=self.seqLabels[temp]
        temp=np.array([self.I_opt_valid.tolist()+self.I_bad_valid.tolist()])[0]
        self.xvalid=self.states[temp,:,:]
        self.yvalid=self.seqLabels[temp]
        del temp
        
        self.ytrain = np.expand_dims(np.expand_dims(self.ytrain,-1),-1)
        self.yvalid = np.expand_dims(np.expand_dims(self.yvalid,-1),-1)
        
        # currently data is balanced. If there is an imbalance, adjust this parameter.
        self.class_weight = {0 : 1,1: 1}
        
    def preprocess(self):
        self.reshape_and_process()
        self.train_test_split()
        
    def get_grouping(self, num):
        
        label = "<Unknown>"
        dataset = "<Unknown_Dataset>"
        
        if num in np.concatenate([self.I_bad,self.I_bad_ho,self.I_bad_valid]):
            label = "Anomalous"
        elif num in np.concatenate([self.I_opt, self.I_opt_valid,self.I_opt_ho]):
            label = "Nominal"
        else:
            print("index doesn't exist in the dataset")
        
        if num in np.concatenate([self.I_bad, self.I_opt]):
            dataset = "Train"
        elif num in np.concatenate([self.I_bad_valid,self.I_opt_valid]):
            dataset = "Validation"     
        elif num in np.concatenate([self.I_bad_ho, self.I_opt_ho]):
            dataset = "Test"
        else: 
            print("Invalid dataset id")
            
        return label, dataset
            
    
    
    
    def get_filename(self, index):

        return Path(self.sample_list[index]).stem
    
    
    def read_lines_from_file(self,filename):
        
        with open(os.path.join(self.parameters_directory,filename),'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        
        return content

    
    ##TODO: have a decode type argument 
    def decode_parameter_label(self,param):
        if (isinstance(param,int)):
            return param
        
        else:
            return self.all_parameter_names.index(param)
        
    
    def param_index_to_label(self, param_index):
        return self.all_parameter_names[param_index]
        
        
        
        
        
        
    
    
    
