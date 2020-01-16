#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:34:58 2018

@author: dweckler
"""
'''
@author: vjanakir
This is the code for deep temporal multiple instance learning (DTMIL). This is the version of ADOPT that is based on deep learning.
The code assumes Keras with Theano or Tensorflow backend.
uses Anaconda virtual env with Python 2.7 and keras. It should also work in Python 3.x but not tested.
'''
# load python libraries
import numpy as np, math
from keras.engine.topology import Layer, InputSpec
from keras import backend as T
import pickle
import os
from scipy.integrate import trapz



source_path = os.path.dirname(os.path.realpath(__file__))


#%% custom model functions

def sigmoid(x,decay,bias):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-decay*(item-bias))))
    return a

def get_weight_fn(maxlen):
    temp=0.1+np.array(sigmoid(np.arange(maxlen).tolist(),decay=0.1,bias=70))
    temp=temp/np.sum(temp)
    return temp
#plt.plot(get_weight_fn(100))    
        
class aggregationLayer(Layer):
    """
    This is a custom Keras layer. This pooling layer accepts the temporal
    sequence output by a recurrent layer and performs multiple instance pooling,
    looking at only the non-masked portion of the sequence. The pooling
    layer converts the instance probabilities (same length as input sequence) into a bag-level probability.
    
    input shape: (nb_samples, nb_timesteps, nb_features)
    output shape: (nb_samples, 1)
    """
    def __init__(self, **kwargs):
        super(aggregationLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def call(self, x, mask=None):
        if mask is None:
            mask = T.mean(T.ones_like(x), axis=-1)
        mask = T.cast(mask,T.floatx())
        
        dr_perc=0.5
        mask1=T.dropout(mask,level=dr_perc)
        mask1=T.clip(mask1, 0, 1)
        
        mod_smax=T.max(x[:,:,0]*mask1,axis=1).dimshuffle(0,'x')
        smax = T.max(x[:,:,0]*mask,axis=1).dimshuffle(0,'x') #(nb_samples, np_features)
        smin = T.min(x[:,:,0]*mask,axis=1).dimshuffle(0,'x') #(nb_samples, np_features)
        
#        mod_smax=T.expand_dims(T.max(x[:,:,0]*mask1,axis=1), 1)
#        smax = T.expand_dims(T.max(x[:,:,0]*mask,axis=1), 1) #(nb_samples, np_features)
#        smin = T.expand_dims(T.min(x[:,:,0]*mask,axis=1), 1) #(nb_samples, np_features)
        
        x_rounded=x[:,:,0]*mask
        sum_unmasked=T.batch_dot(x_rounded,mask,axes=1) # (nb_samples,np_features)
        
        ssum = T.sum(x,axis=-2) #(nb_samples, np_features)
        rcnt = T.sum(mask,axis=-1,keepdims=True) #(nb_samples) # number of unmasked samples in each record
        bag_label=sum_unmasked/rcnt
        smean=ssum/rcnt
        
#        # sigmoid weighted mean:
#        weight_fn=T.reshape(T.transpose(T.tile(T.reshape(T.variable(get_weight_fn(100)),(100,1)),T.shape(x)[0])),(T.shape(x)[0],T.shape(x)[1],1))
#        weighted_x=weight_fn*x
#        wsum=T.sum(weighted_x,axis=-2) #(nb_samples, np_features)
##        weight_sum=T.reshape(T.batch_dot(T.ones_like(x),weight_fn,axes=1),T.shape(rcnt)) # used T.ones_like(x) instead of x to check if I am seeing the outputs..which helped me debug
#        wmean=wsum # because the weights are normalized
        
#        sofmax=(1/largeNum)*T.log(T.sum(T.exp()))
        
#        return bag_label
        return smax # max voting
#        return smin # min voting
#        return smean # temporal mean pooling        
#        return wmean # sigmoid weighted mean
#        return sofmax
#        return mod_smax
            
    def compute_mask(self, input, mask):
        return None 



def get_auc(ytest, ytest_prob):
    tau_mat=np.arange(0,1.01,0.01)
    TPR=np.zeros(len(tau_mat),)
    FPR=np.ones(len(tau_mat),)
    for i in np.arange(len(tau_mat)):
        tau=tau_mat[i]
        ytest_pred=np.zeros(ytest_prob.shape)
        ytest_pred[ytest_prob>tau]=1
        posIdx=np.where(ytest==1)[0]
        TPR[i]=len(np.where(ytest_pred[posIdx]==1)[0])/float(len(posIdx))
        negIdx=np.where(ytest==0)[0]
        FPR[i]=len(np.where(ytest_pred[negIdx]==1)[0])/float(len(negIdx))
    auc_bag=abs(trapz(TPR,FPR))
    return auc_bag 
    
    
#save a file to a specified directory
def save_something(stuffToSave,filename):
    with open ('{}'.format(filename),'wb') as output:
        pickle.dump(stuffToSave,output, pickle.HIGHEST_PROTOCOL)
        
#load a file from a specified directory  
def load_something(filename):
    with open ('{}'.format(filename),'rb') as inFile:
        return pickle.load(inFile)

#grab labels from indeces
def get_labels_from_indeces(label_indeces,label_strings):
    ordered_label_strings = np.asarray([label_strings[p] for p in label_indeces])
    
    if isinstance(label_indeces, list):
        ordered_label_strings = ordered_label_strings.tolist()
        
    return ordered_label_strings

#dual option for multi-sort
def dual_sort(myList, side_list,absolute_value = True,reverse = False):
    sorted_list, side_lists = multi_sort(myList, [side_list],absolute_value,reverse)
    
    return sorted_list, side_lists[0]
    
    
#easily sort multiple arrays at once
def multi_sort(myList,side_lists,absolute_value = True,reverse = False):   
    #preprocess and get our sort arrays
    myArray = np.asarray(myList)     
    if (absolute_value):
        myArray = np.absolute(myArray)
    sorted_indeces = np.argsort(myArray)
    
    ##main array sort
    sorted_array = myArray[sorted_indeces]
    if (reverse):
        sorted_array = np.flip(sorted_array, axis = 0)

    #sort everything else according to main array
    sorted_side_arrays = []
    for sList in side_lists:
        sorted_arr = np.asarray(sList)[sorted_indeces]
        if(reverse):
            sorted_arr = np.flip(sorted_arr,axis=0)
        
        sorted_side_arrays.append(sorted_arr)
          
    return(sorted_array,sorted_side_arrays)
    
    
def flat_avg(avg_array):
    flat_mean = np.mean(avg_array,axis = 0)    
    mean_list = []
    arr_size = avg_array.shape[0]
    for mean_val in flat_mean:
        new_arr = np.full(arr_size,mean_val)
        mean_list.append(new_arr)
    
    avg_guideline = np.array(mean_list)  
    return avg_guideline.swapaxes(0,1)

