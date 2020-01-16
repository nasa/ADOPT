#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tkinter import Tk, Listbox, Grid, Label, END,filedialog, Scrollbar, VERTICAL, RIGHT, Y, EXTENDED
from tkinter.ttk import Frame, Button, Entry, Style

from tkinter.messagebox import showinfo, showerror
import os
import sys
import json

sep = os.path.sep
source_path = "..{}dtmil{}configuration{}".format(sep,sep,sep)
sys.path.append(source_path)
#from config_dtmil import get_json_config_data

def exit_program():
    root.destroy()
    exit()
    

def read_lines_from_file(directory,filename):
        
        with open(os.path.join(directory,filename),'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        
        return content 

def write_lines_to_file(list_of_text,directory,filename):
    filepath =  os.path.join(directory,filename)
#    print(f"Filepath: {filepath}")
    with open(filepath,'w') as f:
            for text in list_of_text:
                f.write("{}\n".format(text))
    

class ParameterSelector:
    
    def __init__(self,master):
        self.master = master
        sel_col = 0
        hold_col = 1
        
        lbl_row = 0
        lst_row = 1
        btn_row = 2
        save_row = 3
        
        self.edited = False
        
        scale_factor = 0.5
        window_width = int(root.winfo_screenwidth()*scale_factor)
        window_height = int(root.winfo_screenheight()*scale_factor)
          
        self.master.geometry(f"{window_width}x{window_height}")
        
        frame = Frame(master)
        
        Grid.columnconfigure(master,sel_col,weight=1)
        Grid.columnconfigure(master,hold_col,weight=1)

        Grid.rowconfigure(master, lst_row, weight=1)

        frame.grid()
        
        
        self.full_listbox = Listbox(master,selectmode = EXTENDED, width = 20)
        scrollbar = Scrollbar(self.full_listbox, orient=VERTICAL)
        self.full_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.full_listbox)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.full_listbox.grid(row=lst_row,column=sel_col,padx=(20,20),pady=(5,10),sticky = 'news')  
        
        self.holdout_listbox = Listbox(master,selectmode = EXTENDED, width = 20)
        scrollbar = Scrollbar(self.holdout_listbox, orient=VERTICAL)
        self.holdout_listbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command=self.holdout_listbox)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.holdout_listbox.grid(row=lst_row,column=hold_col,padx=(20,20),pady=(5,10),sticky = 'news')
        
        
        self.selected_param_label = Label(master,text = "Selected Parameters")
        self.selected_param_label.grid(row = lbl_row, column = sel_col)
        
        self.holdout_param_label = Label(master,text = "Holdout Parameters")
        self.holdout_param_label.grid(row = lbl_row,column = hold_col)
        
        self.selected_to_holdout_button = Button(master,text= "->", command = self.move_to_holdout)
        self.selected_to_holdout_button.grid(row= btn_row,column = sel_col,padx=(40,40),sticky = 'ew')
        
        self.holdout_to_selected_button = Button(master, text= "<-", command = self.move_to_selected)
        self.holdout_to_selected_button.grid(row= btn_row,column = hold_col,padx=(40,40),pady=(5,10),sticky = 'ew')
        
        self.save_button = Button(master,text = "Save", command = self.save_lists)
        self.save_button.grid(row = save_row, column = sel_col,pady=(5,10))
        
        self.reset_button = Button(master,text= "Reset", command = self.config_listboxes)
        self.reset_button.grid(row = save_row,column = hold_col)
           
        self.dataset_dir = ""
        self.params_path = ""
        
    
    def select_dataset(self):
        #FIXME: Add an exception handler just in case the file isn't found
        
        #get DTMIL_config_dir.json
        directory_config_file = "DTMIL_config_dir.json"
        config_file = os.path.join(source_path, directory_config_file)
        
        with open(config_file, 'r') as dirfile:
            self.dir_data = json.load(dirfile)
            dirfile.close()    
            
        
        #select the dataset directory
        self.dataset_dir = filedialog.askdirectory(title = "Choose Dataset Folder")
        
        if self.dataset_dir == "":
            exit_program()
            
        self.params_path = os.path.join(self.dataset_dir,self.dir_data["parameters_directory"])
        
        self.dataset_cfg_filepath = os.path.join(self.dataset_dir,"DTMIL_config.json")
                
        with open(self.dataset_cfg_filepath) as cfg_file:
            
            self.dataset_config_file = json.load(cfg_file)
            cfg_file.close()
    
            
    def config_listboxes(self):
        
        self.edited = False

        #get the holdout state names + the selected variables
    
        ##load here from the config file
        preprocessing_parameters = self.dataset_config_file['preprocessing']

        all_params = set(preprocessing_parameters["all_parameter_names"])
        holdout_params = set(preprocessing_parameters["redundant_parameters"])
        selected_params = all_params - holdout_params

        
        
        sel_list = sorted(list(selected_params))
        hold_list = sorted(list(holdout_params))
        
        #make sure the listboxes are clear
        self.full_listbox.delete(0,END)
        self.holdout_listbox.delete(0,END)
        
        self.full_listbox.insert(END,*sel_list)
        self.holdout_listbox.insert(END,*hold_list)
        
        #TODO: check with the dataset and verify that all the parameters are the same.  
        # If there are missing ones, add them, if there is an extra in the set, throw and error, maybe give an option to delete it
    
        
    
    def after_startup_setup(self):
        self.select_dataset()
        self.config_listboxes()
    
    
            
    def move_to_holdout(self):
        self.move_listbox(self.full_listbox, self.holdout_listbox)
    
    
    def move_to_selected(self):
        self.move_listbox(self.holdout_listbox, self.full_listbox)
    
        
    def move_listbox(self,listbox_source, listbox_destination):
        self.edited = True

        print(listbox_source.curselection())
        selected_idx = list(listbox_source.curselection())
        
        selected_values = []
        for i in selected_idx:
            selected_values.append(listbox_source.get(i))

        for i in selected_idx[::-1]:
            listbox_source.delete(i)

        print(selected_values)    
        listbox_destination.insert(END,*selected_values)
                
    def save_lists(self):
        #output lists to file
        if(not self.edited):
            showerror(title= "Error",message= "No changes made!")
            return
            
        #TODO: Have it save to the json file instead
        
        print("saving lists")
        selected_param_list = sorted(self.full_listbox.get(0,END))
        holdout_param_list = sorted(self.holdout_listbox.get(0,END))
        print(selected_param_list,holdout_param_list)
        
        ##write these to the files and overwrite
        
        
        preprocessing_parameters = self.dataset_config_file['preprocessing']
    
        preprocessing_parameters["redundant_parameters"] = holdout_param_list
        
        json_cfg_string = json.dumps(self.dataset_config_file,sort_keys=True, indent=4, separators=(',', ': '))
    
        with open(os.path.join(self.dataset_dir,"DTMIL_config.json"),'w') as outfile:
            outfile.write(json_cfg_string)
            outfile.close()
        
        showinfo("Info", "Successfully saved files\n" + "Sorting both lists")
        self.config_listboxes()

        
        
        

root = Tk()
root.title("Parameter Selector")

param_sel = ParameterSelector(root)

root.after(10,param_sel.after_startup_setup)

root.mainloop()  



