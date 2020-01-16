#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:56:09 2019

@author: dweckler
"""



import numpy as np, math
from keras.engine.topology import Layer, InputSpec
from keras import backend as T
import json
import pickle
import sys
import os



source_path = os.path.dirname(os.path.realpath(__file__))
directory_config_filename = "DTMIL_config_dir.json"


def get_json_config_data(input_directory = None, new_run = False):

    dir_config_file_path = os.path.join(source_path, directory_config_filename)

    with open(dir_config_file_path, 'r') as dirfile:
        dir_data = json.load(dirfile)
        
        ##make sure the defined separators of the directory config file match that of the os
        for key,val in dir_data.items():
            new_val = val.replace("/",os.path.sep)
            dir_data[key] = new_val
    
    ####Load a dataset from a specified directory
    if(input_directory):
        cfg_data, cfg_file_path, dataset_dir = get_from_input_directory(input_directory,dir_data)
        
    #get from "datasets" folder   
    else:      
        cfg_data, cfg_file_path, dataset_dir = get_from_datasets_directory(dir_data)
  
    #make sure the json file is up to date
    update_JSON(cfg_data,cfg_file_path)


    cfg_id_label = "config_id"
    name_label = "config_name"
    
    config_name = cfg_data[name_label]
    id_hold = cfg_data["id_hold"]
    config_id = cfg_data[cfg_id_label]
    
    if (new_run and not id_hold):
        #increment for this new run
        config_id += 1
        cfg_data[cfg_id_label] = config_id
        save_JSON(cfg_data,cfg_file_path)

    
    full_config_name = "{}_{}".format(config_name,config_id)

    
    #set up the model storage and model output directories and check to see if the directory was there before
    directory_existed_already = __get_directory_with_ID(dir_data,dataset_dir,"model_storage_directory",full_config_name)
    __get_directory_with_ID(dir_data,dataset_dir,"model_output_directory",full_config_name)

    #TODO: determine if we want this warning if hold is set
    #check to make sure we don't overwrite an existing run
    if(new_run and directory_existed_already):
        
        if(id_hold):
            print("ID hold is ON")
        
        choice = input("Existing run (\"{}\") already found, do you wish to overwrite? (y/n)\n".format(full_config_name))

        if choice == 'y':
            print("Overwriting existing run")

        else:
            if choice != 'n':
                print("Invalid input")
                
            output_string = ("If you do not wish to overwrite the existing run, change the \"{}\" field in the JSON file "
                             "to a number greater than or equal to the current run.\n"
                             "Alternatively, change the \"{}\" field to a name that doesn't conflict\n".format(cfg_id_label,name_label))
            
            print(output_string)
            
            print("Exiting program...")
            sys.exit(0)
        
        
    
    return dir_data, cfg_data, dataset_dir



def get_from_input_directory(input_directory,dir_data):
    
    if(input_directory.endswith(".json")):
        json_name = os.path.basename(input_directory)
        cfg_file_path = input_directory
        dataset_dir = os.path.dirname(input_directory)
        
    else:
        json_name = 'DTMIL_config.json'
        cfg_file_path = os.path.join(input_directory,json_name)
        dataset_dir = input_directory

    with open(cfg_file_path) as cfgfile:
        cfg_data = json.load(cfgfile)
        
        
    datasets_dir = os.path.abspath(os.path.join(source_path, dir_data["datasets_directory"]))    
    filename = dir_data["selected_dataset"]
    file_dir = os.path.join(datasets_dir, filename)
    
    
    with open (file_dir, 'w') as selected_dataset_file:
        selected_dataset_file.write(dataset_dir)
        selected_dataset_file.close()
        
      
    return cfg_data, cfg_file_path, dataset_dir



def get_from_datasets_directory(dir_data):
    
    datasets_dir = os.path.abspath(os.path.join(source_path, dir_data["datasets_directory"]))    
    filename = dir_data["selected_dataset"]
    file_dir = os.path.join(datasets_dir, filename)
    
    prior_selected_dataset_file = False
    
    #check for selected dataset file
    try:
        selected_dataset_file = open(file_dir,'r')
        dataset_name = selected_dataset_file.readline()
        #print(dataset_name)
        prior_selected_dataset_file = True
        selected_dataset_file.close()
    
    #if it's not found, create it
    except IOError:
        dataset_name = input("{} not found, type the path of the dataset you wish to open\n".format(filename))
        selected_dataset_file = open(file_dir,'w')
        selected_dataset_file.write(dataset_name)
        selected_dataset_file.close()
    
    #open the dataset 
    if(prior_selected_dataset_file):
        new_name = input("Type the path of the dataset you wish to open, or press enter to open '{}'\n".format(dataset_name))
        if(new_name != ""):
            dataset_name = new_name
            os.remove(file_dir)
            
            
            selected_dataset_file = open(file_dir,'w')
            selected_dataset_file.write(dataset_name)
            selected_dataset_file.close()
    
    
    if os.path.exists(dataset_name):
        dataset_dir = dataset_name
    
    else:
        dataset_dir = os.path.join(datasets_dir,dataset_name)
        
        
    #check to see if the file is there, if not, then clear the selected dataset and exit the program
    try:
        cfg_file_name ="DTMIL_config.json"
        cfg_file_path = os.path.join(dataset_dir,cfg_file_name)
                
        cfgfile = open(cfg_file_path)
        cfg_data = json.load(cfgfile)
        cfgfile.close()
        
    except IOError as e:
        print("{}".format(e))
        print("config file(s) not found. The dataset and/or config files may not exist in the specified directory. Clearing {}".format(filename))
        
        os.remove(file_dir)
        
        sys.exit()
        
    return cfg_data, cfg_file_path, dataset_dir



    
def find_missing_keys(d,old_d):
   # old_subdict = getFromDict(old_dict,map_list)
    added_keys = [] 
    for key,val in d.items():
        #print(key)
        if key not in old_d:
            old_d[key] = val
            added_keys.append(key)
        if isinstance(val,dict):
            old_subdict= old_d[key]
            added_keys = added_keys + find_missing_keys(val,old_subdict)
        
    return added_keys
        
def delete_extra_keys(old_d,orig_dict):   
    keys_to_pop = []
    all_popped_keys = []
    for key,val in old_d.items():
        if key not in orig_dict:
            keys_to_pop.append(key)
            #print("pop attempted")            
        if isinstance(val,dict):
            if key not in keys_to_pop:
                orig_subdict = orig_dict[key]
                all_popped_keys = all_popped_keys + delete_extra_keys(val,orig_subdict)  

    for key in keys_to_pop:
        old_d.pop(key)
        
    return keys_to_pop + all_popped_keys


def update_JSON(json_to_change,json_to_change_filepath):
    
    config_path = os.path.join(source_path,"DTMIL_config_default.json")
    with open(config_path) as json_file:
        orig_json = json.load(json_file)
    
    added_stuff = find_missing_keys(orig_json,json_to_change)
    deleted_stuff = delete_extra_keys(json_to_change,orig_json)
    stuff_added = len(added_stuff)>0
    stuff_removed = len(deleted_stuff)>0

    if stuff_added or stuff_removed:
        print("The current json file is outdated")

        if (stuff_removed):
            print(f"Entries that will be removed: {deleted_stuff}")
        if (stuff_added):
            print(f"Entries that will be added (program will crash otherwise): {added_stuff}")

        choice = input("Do you wish to overwrite the current JSON file? (y/n)\n")

        if choice == 'y':
            save_JSON(json_to_change,json_to_change_filepath)

        else:
            if choice != 'n':
                print("Invalid choice")

            print("Exiting program")
            sys.exit()
    
def save_JSON(json_data_to_save,json_to_save_filepath):
        print("Writing to JSON file")
        json_cfg_string = json.dumps(json_data_to_save,sort_keys=True, indent=4, separators=(',', ': '))
        with open(os.path.join(json_to_save_filepath),'w') as outfile:
            outfile.write(json_cfg_string)
            print("Write successful")
            outfile.close() 

    

    
def __get_directory_with_ID(dir_data, dataset_dir, directory_string,ID):
    
    #folder_name = 'run'
    
    updated_dir = os.path.join(dir_data[directory_string], "{}".format(ID))
    dir_data[directory_string] = updated_dir    
    full_dir_data = os.path.join(dataset_dir,dir_data[directory_string])
    
    directory_exists = os.path.exists(full_dir_data)
    if not directory_exists:
            print("Creating:", full_dir_data)
            os.makedirs(full_dir_data)
            
    return directory_exists