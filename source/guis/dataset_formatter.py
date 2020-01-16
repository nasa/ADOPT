#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 00:09:03 2019
@author: dweckler
"""
from tkinter import Tk, Label, Button, Entry, mainloop, filedialog, messagebox, Grid, Checkbutton, IntVar, END
import os
#import tkSimpleDialog as simpledialog
import matplotlib
matplotlib.use('qt5agg')



from shutil import copyfile, move,copytree,copy
import pandas as pd
import json
import glob
from pathlib import Path

import errno

#if nominal filelist and adverse filelist don't exist, generate a file list from the inputted folders
#ask via a popup if you want to do this. "directories were found instead of file lists. 
#Would you like to use the inputted directories to generate their respective file lists?"
#use listdir to get the lists, make sure to parse out everything but the relative file path



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

def set_entry_text(entry,text):
    entry.delete(0,END)
    entry.insert(0,text)


class DatasetFormatter:
    
    def __init__(self, master):
        
        data_row = 0
        nominal_flist_row = 1
        adverse_flist_row = 2
        last_row = 3
        
        label_column = 0
        filelist_column = 1
        button_column = 2
        
        self.copy_or_move = IntVar()
        self.copy_or_move.set(1)
        self.checkbutton = Checkbutton(master,text="Copy",variable = self.copy_or_move).grid(row=last_row)
        
        
        self.use_filelists = IntVar()
        self.use_filelists.set(0)
        self.filelist_check = Checkbutton(master,text="Filelists", command = self.list_swap, variable = self.use_filelists)
        self.filelist_check.grid(row = last_row,column = button_column)
        
        self.default_adverse_filelist_path = ""
        self.default_nominal_filelist_path = ""
        self.default_adverse_folder_path = ""
        self.default_nominal_folder_path = ""

        
        
        
        self.folder_label = Label(master, text="Data Folder:")
        self.folder_label.grid(row=data_row)
        
        self.nominal_filelist_label = Label(master, text="Nominal Directory:")
        self.nominal_filelist_label.grid(row=nominal_flist_row,column = label_column)
    
        self.adverse_filelist_label = Label(master, text="Adverse Directory:")
        self.adverse_filelist_label.grid(row=adverse_flist_row,column = label_column)
        
        #Define the entry fields
        self.data_folder_entry = Entry(master)
        self.nominal_path_entry = Entry(master)
        self.adverse_path_entry = Entry(master)
        
        Grid.columnconfigure(master,filelist_column,weight=1)
        
        self.data_folder_entry.grid(row=data_row, column=filelist_column,sticky = 'we')
        self.nominal_path_entry.grid(row=nominal_flist_row, column=filelist_column,sticky = 'we')
        self.adverse_path_entry.grid(row = adverse_flist_row, column = filelist_column,sticky = 'we')
        
        #define the "choose file" button
        self.data_folder_button = Button(master,text="Choose Folder", command = self.get_the_folder)
        self.nominal_filelist_button = Button(master,text="Choose Folder", command = self.get_nominal_filelist)
        self.adverse_filelist_button = Button(master,text="Choose Folder", command = self.get_adverse_filelist)
        
        self.data_folder_button.grid(row=data_row,column=button_column)
        self.nominal_filelist_button.grid(row=nominal_flist_row,column=button_column)
        self.adverse_filelist_button.grid(row=adverse_flist_row,column=button_column)
        
        self.save_button = Button(master,text="Generate Folder Hierarchy",fg ="#8b0000" , command = self.generate_folder_structure)
        self.save_button.grid(row = last_row,column = filelist_column)

        

            
    def get_the_folder(self):
        
        home_dir = str(Path.home())
        print("Choosing folder")
        filename = filedialog.askdirectory(initialdir =home_dir)
        
        self.data_folder_entry.delete(0, 'end')
        self.data_folder_entry.insert(0,filename)
    
    def get_nominal_filelist(self):
        print("Choosing nominal")
        self.get_filelist(self.nominal_path_entry)
        
         
    def get_adverse_filelist(self):
        print("Choosing adverse")
        self.get_filelist(self.adverse_path_entry)

        
    def get_filelist(self,my_filelist_entry):
        
        dataset_folder_path = self.data_folder_entry.get()
        if dataset_folder_path == '':
            initial_directory = str(Path.home())       
        else:
            initial_directory = dataset_folder_path
        
        
        if (self.use_filelists.get()):
            filename = filedialog.askopenfilename(initialdir = initial_directory)
            
        else:
            filename = filedialog.askdirectory(initialdir = initial_directory)
        
        my_filelist_entry.delete(0, 'end')
        my_filelist_entry.insert(0,filename)
        
        
    def list_swap(self):
        
        
        if not (self.use_filelists.get()):
            the_label = "Directory"
            the_button_label = "Choose Folder"
            
            self.default_nominal_filelist_path = self.nominal_path_entry.get()            
            set_entry_text(self.nominal_path_entry,self.default_nominal_folder_path)
            
            self.default_adverse_filelist_path = self.adverse_path_entry.get()
            set_entry_text(self.adverse_path_entry,self.default_adverse_folder_path)
            
            
        else:
            the_label = "Filelist"
            the_button_label = "Choose File"   
            
            self.default_nominal_folder_path = self.nominal_path_entry.get()
            set_entry_text(self.nominal_path_entry,self.default_nominal_filelist_path)

            
            self.default_adverse_folder_path = self.adverse_path_entry.get()
            set_entry_text(self.adverse_path_entry,self.default_adverse_filelist_path)
            
        
        self.nominal_filelist_button.config(text = the_button_label)
        self.adverse_filelist_button.config(text = the_button_label)
        
        
        self.nominal_filelist_label.config(text = "Nominal {}".format(the_label))
        self.adverse_filelist_label.config(text = "Adverse {}".format(the_label))
    

    def generate_file_list(self,directory,dataset_folder_path):
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
            
        
    
    def generate_folder_structure(self):
        
        
        nominal_entry_path = self.nominal_path_entry.get()
        adverse_entry_path = self.adverse_path_entry.get()
        dataset_folder_path = self.data_folder_entry.get()
        
        
        blank_path = ""
        
        if (dataset_folder_path == blank_path):
            messagebox.showerror("Error",'Please enter a valid dataset folder path')
            return
        
        
        #TODO: make this generate the filelists paths if it's a directory (indicated by the checkmark)
        #TODO: add errors if the directory isn't where it's supposed to be

        if(self.use_filelists.get()):
            
            
            if (not (nominal_entry_path.endswith('.txt')) or nominal_entry_path == ""):
                messagebox.showerror("Error",'Please enter a valid path for the nominal file list')
                return
                
            
            if not (adverse_entry_path.endswith('.txt')):
                messagebox.showerror("Error",'Please enter a valid path for the adverse file list')
                return
            
        else:
            if((nominal_entry_path == blank_path) or (adverse_entry_path == blank_path)):
                messagebox.showerror("Error","Please enter a valid directory path")
                return
            
            nominal_entry_path = self.generate_file_list(nominal_entry_path,dataset_folder_path)
            adverse_entry_path = self.generate_file_list(adverse_entry_path,dataset_folder_path)
                
    
        #Ask for name of dataset + entry
        #dataset_name = simpledialog.askstring("Input", "Please enter the name of the dataset")
        
        
           
        print("Attempting to read filepath from nominal filelist")
        data_filenames = []
        with open('{}'.format(nominal_entry_path),'r') as f:
            data_filenames = f.readlines()
            data_filenames = [x.strip() for x in data_filenames]
    
       
        
        #test to see if the files are there before doing anything
        test_file_path = data_filenames[0]
        csv_filepath = os.path.join(dataset_folder_path,test_file_path)
        
        try:
            df = pd.read_csv(csv_filepath)
            parameter_list = list(df.columns.values)
            
        except:
            
            messagebox.showerror("Error", f"Could not find any data files in the specified Data Folder path:\n{dataset_folder_path}\n\nAttempted to open file:\n{csv_filepath}")
            return
            
        
        #TODO: set initial directory to current dir initialdir = os.path.sep
        home_dir = str(Path.home())
        dataset_path = filedialog.asksaveasfilename(initialdir = home_dir)
        if dataset_path is None or dataset_path == "":
            messagebox.showerror("Error", "No name entered!")
            return


        else:
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
        adverse_filename = adverse_entry_path.split(sep)[-1]
        nominal_filename = nominal_entry_path.split(sep)[-1]
        
        #place the specified adverse and nominal filelists inside the parameters folder
        copyfile(adverse_entry_path,os.path.join(parameters_path,adverse_filename))
        copyfile(nominal_entry_path,os.path.join(parameters_path,nominal_filename))
        
        #move the directory of the dataset folder
        
        files = list_visible_files(dataset_folder_path)
        
        if (self.copy_or_move.get() == 1):
            print("copying")
            
            files = list_visible_files(dataset_folder_path)

            
            for f in files:
                

                if f not in [adverse_filename, nominal_filename]:
                    #copytree(os.path.join(dataset_folder_path,f),raw_data_path)
                    copy_files_or_directories(os.path.join(dataset_folder_path,f),os.path.join(raw_data_path,f))  


            
        elif(self.copy_or_move.get() == 0):
            print("Moving all files from the dataset directory to our raw data directory")
            for f in files:
                
                if f not in [adverse_filename, nominal_filename]:
                    #copytree(os.path.join(dataset_folder_path,f),raw_data_path)
                    move(os.path.join(dataset_folder_path,f),raw_data_path)            
            
        else: 
            print("this shouldn't happen")
            

        
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
        
        
        messagebox.showinfo(title = "Info", message="Dataset Formatting Process Completed")
        
#TODO: Also maybe generate DTMIL_config_dir.json
def make_directory(folder_path):
        
    try:  
        os.mkdir(folder_path)
    except OSError:  
        print ("Creation of the dataset folder %s failed" % folder_path)
        
#TODO: add a menu option to just export this by itself
def export_json_cfg(directory = "", json_filename = "DTMIL_config.json", fl_nominal = "filelist_nominal.txt", fl_adverse = "filelist_adverse.txt",param_names = []):
    
    
    sep = os.path.sep
    default_config_path = "..{}dtmil{}configuration{}".format(sep,sep,sep)
    
    
    
    with open(os.path.join(default_config_path,"DTMIL_config_default.json")) as jsonfile:
        json_data =  json.load(jsonfile)
    
    json_data["importing"]["nominal_filename"] = fl_nominal
    json_data["importing"]["adverse_filename"] = fl_adverse
    json_data["preprocessing"]["all_parameter_names"] = param_names
    

    
    json_cfg_string = json.dumps(json_data,sort_keys=True, indent=4, separators=(',', ': '))
    
    with open(os.path.join(directory,json_filename),'w') as outfile:
        outfile.write(json_cfg_string)
        outfile.close()
    


master = Tk()
master.title("Dataset Formatter")

data_formatter = DatasetFormatter(master)
#master.minsize(width = 100,height = 50)


mainloop( )


