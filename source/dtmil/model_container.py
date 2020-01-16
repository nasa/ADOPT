#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:47:39 2018

@author: dweckler

@author: vjanakir
This is the code for deep temporal multiple instance learning (DTMIL). This is the version of ADOPT that is based on deep learning.
The code assumes Keras with Theano or Tensorflow backend.
uses Anaconda virtual env with Python 2.7 and keras. It should also work in Python 3.x but not tested.
'''
"""


import os, numpy as np, time
import datetime

from keras.layers.core import Dense, Dropout
from keras.layers import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam
from keras.models import load_model
import json

from sklearn.metrics import precision_recall_fscore_support


from dtmil.configuration.config_dtmil import get_json_config_data
from dtmil.utilities import aggregationLayer
from dtmil.utilities import save_something
from dtmil.utilities import get_auc

from dtmil.utilities import load_something



## model parameters
#batch_size = 32 # mini-batch size (number of samples)
#epochs=100 # number of training passes through data
#nhr=5 # number of units in recurrent layer
#nhd=500 # number of hidden units in fully connected layer
#lr = 0.001 # Specify learning rate lr.
#optim=Nadam(lr=lr) # ADAM optimizer with nestrov momentum (see keras documentation). 
#dr=0 # dropout rate (0-1)
#lam=0.01 # regularization 

# path to data. 


class ModelContainer:
   
    #TODO: Update data_path and model_path whenever a model is reloaded, perhaps have a "reload" initializer
    def __init__(self,data_container):
        self.myData = data_container
        self.load_config_data()
        
        model_io_data = data_container.json_data['model_io']
        model_filename = model_io_data["model_filename"]
        model_container_filename = model_io_data["model_container_filename"]
        data_container_filename = model_io_data["data_container_filename"]
        
        model_archive_directory = os.path.join(data_container.dataset_dir,data_container.json_dir_data['model_archive_directory'])
        self.model_archive_directory = model_archive_directory
        
        model_output_directory = os.path.join(self.myData.dataset_dir,self.myData.json_dir_data['model_output_directory'])
        self.model_output_directory = model_output_directory
        
        model_storage_directory = os.path.join(self.myData.dataset_dir,self.myData.json_dir_data['model_storage_directory'])
        self.model_storage_directory = model_storage_directory
        self.model_path = os.path.join(model_storage_directory,model_filename)
        self.model_container_path = os.path.join(model_storage_directory,model_container_filename)
        self.data_path = os.path.join(model_storage_directory,data_container_filename)
        
        optim = Nadam(lr=self.lr) 
        pars = "_".join([str(k) for k in [self.batch_size, self.epochs, self.nhr, self.nhd, self.dr, self.lam, optim.__class__.__name__, self.lr]])

        
        fname_add=model_archive_directory+"temporary".split(os.path.sep)[-1].split('.')[0]+"_"+pars+'_'
        self.model_fname=fname_add+"bestModel-{epoch:02d}-{val_acc:.4f}.hdf5"
        self.json_fname=fname_add+'.json'
        if not os.path.exists(model_archive_directory):
            os.makedirs(model_archive_directory)
    

        print(self.model_path)
        
    
    @classmethod
    def reload_all_data(cls,dataset_dir, json_data_block = None):
        
        print("reloading model and data")
        if json_data_block == None:
            json_data_block = get_json_config_data(dataset_dir)
        
        json_dir_data, json_group_data,dataset_dir = json_data_block
        model_storage_directory = os.path.join(dataset_dir,json_dir_data['model_storage_directory'])
        model_container_path = os.path.join(model_storage_directory, json_group_data['model_io']["model_container_filename"])

        myModel = load_something(model_container_path)
        myModel.update_paths(model_container_path,dataset_dir)
        
        myData = myModel.myData

        myData.dataset_dir = dataset_dir
        model = load_model(myModel.model_path)

        myModel.model = model
        
        return myModel
        
        
    def update_paths(self,model_container_path, new_dataset_dir = None):
        
        if new_dataset_dir is not None:
            dataset_dir = new_dataset_dir
            self.myData.dataset_dir = new_dataset_dir
            
        else:
            dataset_dir= self.myData.dataset_dir
        
        
        model_io_data = self.myData.json_data['model_io']
        model_filename = model_io_data["model_filename"]
        model_container_filename = model_io_data["model_container_filename"]
        data_container_filename = model_io_data["data_container_filename"]
    
        model_archive_directory = os.path.join(dataset_dir,self.myData.json_dir_data['model_archive_directory'])
        self.model_archive_directory = model_archive_directory
        
        model_output_directory = os.path.join(dataset_dir,self.myData.json_dir_data['model_output_directory'])
        self.model_output_directory = model_output_directory
        
        model_storage_directory = os.path.join(dataset_dir,self.myData.json_dir_data['model_storage_directory'])
        self.model_storage_directory = model_storage_directory
        self.model_path = os.path.join(model_storage_directory,model_filename)
        self.model_container_path = os.path.join(model_storage_directory,model_container_filename)
        self.data_path = os.path.join(model_storage_directory,data_container_filename)
        
        

        
        
            
    def load_config_data(self):
        data = self.myData.json_data["training"]       
        self.epochs = data['epochs']
        self.batch_size = data['batch_size']
        self.nhr = data['nhr']
        self.nhd = data['nhd']
        self.lr = data['lr']
        self.dr = data['dr']
        self.lam = data['lam']
        
    def configure_model(self,train_flag, pre_trained_model = None, pre_trained_json = None):
        # create model configuration
        myData = self.myData
        self.train_flag = train_flag
        
        self.pre_trained_model = pre_trained_model
        self.pre_trained_json = pre_trained_json
        
   
        if train_flag:
        
            # standard sequential model in Keras where layers can be added.
            model = Sequential()
            
            # masking layer to make sure masked time-steps are not considered in the gradient calculations    
            # model.add(Masking(mask_value=mask_val, input_shape=(maxlen, nfeat)))
            lam = self.lam
            dr = self.dr
            optim = Nadam(lr=self.lr) 

            # GRU layer (RNN)
            model.add(GRU(
                input_shape=(myData.maxlen, myData.nfeat),
                units=self.nhr,
                return_sequences=True,
                stateful=False, 
                unroll=False, 
                implementation='gpu',
                activation='tanh',
                kernel_regularizer=l2(lam), 
                recurrent_regularizer=l2(lam), 
                bias_regularizer=l2(lam)))
            model.add(Dropout(dr))
            
            # fully connected layer - note the timedistributed type which processes data at every time step.
            model.add(TimeDistributed(Dense(units=self.nhd,
                                            activation='tanh',
                                            kernel_regularizer=l2(lam),
                                            bias_regularizer=l2(lam),
                                            kernel_constraint = None)))
            model.add(Dropout(dr))
        
            # logistic layer (the output of this layer gives instance probabilities)
            model.add(TimeDistributed(Dense(units=1, 
                                            activation='sigmoid', 
                                            kernel_regularizer=l2(lam), 
                                            bias_regularizer=l2(lam),
                                            kernel_constraint = None),name="inst_prob"))
            model.add(Dropout(0))    
            
            # multiple-instance aggregation layer
            # model.add(aggregationLayer(name="mil_layer"))
            model.add(MaxPooling1D(pool_size=myData.maxlen))
            start = time.time()
        
            # compile model 
            model.compile(loss="binary_crossentropy", optimizer=optim, metrics=['accuracy'])
            print("Compilation Time : ", time.time() - start)
        
            # serialize (save) model to JSON
            model_json = model.to_json()
            with open(self.json_fname, "w") as json_file:
                json_file.write(model_json)
            print('saved model json to disk')
        else:
                        
            
            ##Check filepath here, if it doesn't exist, load existing model
            # load json and create model
            
            print("Train_Flag set to false, loading pre-trained model")
            model = self._load_pretrained_model()
            
            
#            json_file = open(load_jsonName, 'r')
#            loaded_model_json = json_file.read()
#            json_file.close()
#            model = model_from_json(loaded_model_json,{'aggregationLayer':aggregationLayer})
#        
#            # load weights into new model
#            model.load_weights(load_h5Name)
#            model.compile(loss="binary_crossentropy", optimizer=optim, metrics=['accuracy'])
#            print("Loaded and compiled model from disk")
            
        
        self.model = model

#TODO: Raise better errors for pretrained models
    def _load_pretrained_model(self):
        ##FIXME: Make this have better error handling than "None"

        json_filename = self.pre_trained_json
        pre_trained_model_filename = self.pre_trained_model
        
        
        if (pre_trained_model_filename== "") :
            print("No filepath specified, attempting to load from the default path")
            pre_trained_model_filename = self.model_path
            
 
    #FIXME: This is inconsistent with the above. Fix later somehow (probably with yet another JSON argument)
        if json_filename == "":
            json_filename = None
        
        if (json_filename):
           
            json_file = open(json_filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json,{'aggregationLayer':aggregationLayer})
            
            weights_filename = pre_trained_model_filename
            model.load_weights(weights_filename)
            model.compile(loss="binary_crossentropy", optimizer=Nadam(lr=self.lr) , metrics=['accuracy'])
            print("Loaded and compiled model from disk")
            
        else:
            model_filename = pre_trained_model_filename
            
            print("attempting to load: {}".format(model_filename))
            model = load_model(model_filename)
            print("Loaded model from disk")      
            
        return model
            


#%% train model

    def train_model(self,trainNeeded):
        myData = self.myData

        if trainNeeded:
            try:
                # define checkpoint so that model is saved if it is better than previously saved model
                checkpoint = ModelCheckpoint(self.model_fname, 
                                             monitor='val_accuracy', 
                                             verbose=0,
                                             save_best_only=True,
                                             
                                             mode='auto')
                
                #FIXME: fix the callbacks list bug and model checkpoints
                callbacks_list = [checkpoint]
        
                start = time.time()
                self.training_history = self.model.fit(myData.xtrain, 
                          myData.ytrain, 
                          validation_data=(myData.xvalid,myData.yvalid),
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          validation_split=0.33,
                          verbose = 1, 
                          #callbacks=callbacks_list,
                          shuffle=True)
                
                self.train_time =  time.time() - start
                print("Train Time : ",self.train_time)
        
            except KeyboardInterrupt:
                print('interrupted')

#%% evaluate model performance on train set

    def evaluate_model(self):
        myData = self.myData
        
        temp=np.array([myData.I_opt.tolist()+
                       myData.I_bad.tolist()
                       +myData.I_opt_valid.tolist()
                       +myData.I_bad_valid.tolist()])[0]
        # temp=np.array([I_opt_ho.tolist()+I_bad_ho.tolist()])[0]
        xval=myData.states[temp,:,:]
        yval=myData.seqLabels[temp]
        self.xval = xval
        
        #FIXME: Make this more clear that its evaluating on two different sets of the model
        y_pred_prob=self.model.predict_proba(xval)[:,0]
        self.yValidation_prob = y_pred_prob
        
        self.auc_train =  get_auc(yval, y_pred_prob)
        
        #%% evaluate model performance on test set
        
        temp=np.array([myData.I_opt_ho.tolist()+
                       myData.I_bad_ho.tolist()])[0]
        # temp=np.array([I_opt_ho.tolist()+I_bad_ho.tolist()])[0]
        xtest=myData.states[temp,:,:]
        ytest=myData.seqLabels[temp]
        self.xtest = xtest
        
        
        y_pred_prob=self.model.predict_proba(xtest)[:,0]
        
        self.y_pred_prob = y_pred_prob
        
        self.auc_test = get_auc(ytest, y_pred_prob)
        
        #TODO: Add threshold definition

        self.precision,self.recall,self.fscore, _ = precision_recall_fscore_support(ytest,y_pred_prob.round(), average='weighted')
        
        self.xtest = xtest
        self.ytest = ytest
        
        
        
        self.train_date = datetime.datetime.now()
        
        self.generate_output_file()
      
    def save_model(self):
        
                
        #TODO: add this to the m
        timestr = time.strftime("%Y%h%d-%H%M%S")
        
        self.timestamp = timestr

        
        self.model.save(self.model_path)
        save_something(self.myData,self.data_path)
        #save model container separately from the model (otherwise pickle doesn't work)
        temp = self.model
        self.model = None
        save_something(self,self.model_container_path)
        self.model = temp
        
        json_cfg_string = json.dumps(self.myData.json_data,sort_keys=True, indent=4, separators=(',', ': '))
        
        
        
        
        with open(os.path.join(self.model_storage_directory,"DTMIL_config_{}.json".format(timestr)),'w') as outfile:
            outfile.write(json_cfg_string)
            outfile.close()


  
    def generate_output_file(self):
        print("generating output file...\n\n\n")
        
        myData = self.myData
        model_output_directory = self.model_output_directory
        
        dataset_header = "Output Summary:"
        training_samples = self.__format_sample_output("Training",myData.xtrain,myData.I_opt,myData.I_bad)
        validation_samples = self.__format_sample_output("Validation",myData.xvalid, myData.I_opt_valid,myData.I_bad_valid)
        test_samples = self.__format_sample_output("Test", self.xtest,myData.I_opt_ho,myData.I_bad_ho)
        
        auc_train = "AUC Train: {}".format(self.auc_train)
        auc_test = "AUC Test: {}".format(self.auc_test)
        precision = "Precision: {}".format(self.precision)
        recall = "Recall: {}".format(self.recall)
        f1_score = "F1 Score: {}".format(self.fscore)
        
        epochs = "Epochs: {}".format(self.epochs)
        batch_size = "Batch Size: {}".format(self.batch_size)
        regularization_parameter = "Lambda: {}".format(self.lam)
        dropout_rate = "Dropout Rate: {}".format(self.dr)
        train_date = "Trained on: {}".format(self.train_date)
        number_of_features = "Number of features: {}".format(myData.nfeat)
        
        dropped_states = myData.correlated_states.tolist() + myData.dropped_states.tolist()
        
        dropped_parameters = "Dropped Parameters: \n{}".format( dropped_states )
        dropped_parameter_names ="{}".format( [myData.header[p] for p in dropped_states])
        
        #Find a better way to express this within keras
        if(self.train_flag == False):
            train_date = "Reloaded Model"
        
        
        output_string_list = [dataset_header,
                              number_of_features,
                              train_date,
                              "",
                              training_samples,
                              validation_samples,
                              test_samples,
                              "",
                              epochs,
                              regularization_parameter,
                              dropout_rate,
                              batch_size,
                              "",
                              dropped_parameters,
                              dropped_parameter_names,
                              "",
                              auc_train,
                              auc_test,
                              precision,
                              recall,
                              f1_score
                              ]
        
        
        output_string = "\n".join(output_string_list)
        print(output_string)
        print("\n")
        splice = myData.time_splice
        if(not splice):
            splice = 1       
        
        
        #summary_filename = "model_output_summary_{}_percent.txt".format(int(splice*100))
        summary_filename = "model_output_summary.txt"
        
        with open(os.path.join(model_output_directory,summary_filename),'w') as outfile:
            outfile.write(output_string)
        
        
    def __format_sample_output(self,name, total_samples, nominal_samples, adverse_samples):
        
        total_samples = "{} Samples: {}".format(name,len(total_samples))
        nominal_samples = " - Nominal: {}".format(len(nominal_samples))
        adverse_samples = " - Adverse: {}".format(len(adverse_samples))
        
        return "\n".join([total_samples,nominal_samples,adverse_samples])



