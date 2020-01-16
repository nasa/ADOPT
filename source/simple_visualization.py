from sys import argv
from enum import Enum
from batch_visualization import Batch_Visualizer

num_columns = 4

def visualize_event(event_types,i_bad,i_good):
    if (Event_Type.Nominal in event_types):
        print("visualizing nominal")
        viz.save_sample_parameters(i_good,num_columns=num_columns)
        
    if (Event_Type.Anomalous in event_types):
        print("visualizing adverse")

        viz.save_sample_parameters(i_bad,num_columns=num_columns)
        

class Dataset_Type(Enum):
    Train = 1
    Validation = 2
    Test = 3
    
class Event_Type(Enum):
    Nominal = 1
    Anomalous = 2


print(argv)
#%% user defined variables

if (len(argv) > 1):
    dataset_input = argv[1] 
    
else:
    dataset_input = input("Input the path of the dataset:\n")


viz = Batch_Visualizer(dataset_input)
input_val = ""

while(input_val == ""):
    set_types = {1:"Train",2:"Validation",3:"Test"}


    prompt = ("Which part of the dataset would you like to visualize? If multiple, input the numbers separated by commas.\n\n" 
              "1. Training Set\n"
              "2. Validation Set\n"
              "3. Test Set\n"
              "\n")

    set_nums = input(prompt)
    input_val = set_nums
    
    if (input_val == ""):
        print("no input selected, try again, or press control-c to exit\n")
        
    else:
        sets_list = [int(num) for num in set_nums.split(',')]

dataset_types_list = [Dataset_Type(num) for num in sets_list]

event_prompt = ("Which event would you like to visualize? If multiple, input the numbers separated by commas.\n\n"
                "1. Nominal\n"
                "2. Adverse\n"
                "\n")
event_nums = input(event_prompt)
event_list = [int(num) for num in event_nums.split(',')]
event_types = [Event_Type(num) for num in event_list]


    
print(dataset_types_list)
if(Dataset_Type.Train in dataset_types_list):
    
    visualize_event(event_types,viz.myData.I_bad,viz.myData.I_opt)
    
if(Dataset_Type.Validation in dataset_types_list):
    visualize_event(event_types,viz.myData.I_bad_valid,viz.myData.I_opt_valid)
    
if(Dataset_Type.Test in dataset_types_list):
    visualize_event(event_types,viz.myData.I_bad_ho,viz.myData.I_opt_ho)

    

