# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SequentialSampler

# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pdb
# Uncomment for Python2
# from __future__ import print_function

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax
import numpy as np
import pdb
import pickle


def loadFiles(file):
    
    with open(file, 'r') as inf:
        file = inf.read()
    return file

def createDictionary(train_data):
    uniqueTot = list(set(train_data))
    data = list(uniqueTot)
    values = array(data)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    dictionary = dict(zip(uniqueTot, integer_encoded.tolist()))
    return dictionary

def toNumLabel(abc, dictionary):
    label = np.zeros((len(abc),),dtype=int)
    
    for i in range(len(abc)):
        label[i] = dictionary.get(abc[i])
    print(label)
    return label

def fromNumLabelToOneHot(label,dictionary):   
    onehot=np.zeros((len(label),len(dictionary)),dtype=int)
    for i in range(len(label)):
        onehot[i,label[i]]=1
    return onehot

def toAbc(onehot,dictionary):
    #take in onehot encoded data and transform it to abc
    invert = np.argwhere(onehot==1)[:,1]
    dic_val_list=list(dictionary.values())
    dic_key_list=list(dictionary.keys())
    abc=[]  
    for i in range(len(onehot)):
        abc.append(dic_key_list[dic_val_list.index(invert[i])])
    #can be commented    
    abc=''.join(abc)
    return abc

def createChunks(label):
    chunkSize=100
    chunkNum=int(len(label)/chunkSize)
    chunkedData=label[:chunkNum*chunkSize].reshape((chunkNum,chunkSize))
    return chunkedData

class ABCDataset(Dataset):
    
    def __init__(self,filename):
        self.dictionary=pickle.load(open("dictionary_new.pkl","rb"))
        self.data=loadFiles(filename)
        self.input_data=createChunks(toNumLabel(self.data[:-1], self.dictionary))
        self.labels=createChunks(toNumLabel(self.data[1:], self.dictionary))
        
    def __len__(self):
        
        # Return the total number of data samples
        return len(self.input_data)

    def __getitem__(self, ind):

        data = self.convert_label(self.input_data[ind,:])

        # Convert multi-class label into binary encoding 
        #label = self.convert_label(self.labels[ind,:])
        label = torch.from_numpy(self.labels[ind,:])
        # Return the image and its label
        return (data, label)

    

    def convert_label(self, label):
        """Convert the numerical label to n-hot encoding.
        
        Params:
        -------
        - label: a string of conditions corresponding to an image's class
        Returns:
        --------
        - binary_label: (Tensor) a binary encoding of the multi-class label
        """
        
        np_label=fromNumLabelToOneHot(label,self.dictionary)
        binary_label = torch.from_numpy(np_label)
        return binary_label
    
    

def create_split_loaders(batch_size, shuffle=False,show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. """


    # Get create a ChestXrayDataset object
    train_dataset = ABCDataset('pa4Data/train.txt')
    val_dataset = ABCDataset('pa4Data/val.txt')
    test_dataset = ABCDataset('pa4Data/test.txt')

    # Dimensions and indices of training set
    train_size = len(train_dataset)
    train_ind = list(range(train_size))

    val_size = len(val_dataset)
    val_ind = list(range(val_size))

    test_size = len(test_dataset)
    test_ind = list(range(test_size))
    
    
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SequentialSampler(train_ind)
    sample_test = SequentialSampler(test_ind)
    sample_val = SequentialSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers, 
                              pin_memory=pin_memory)

    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)