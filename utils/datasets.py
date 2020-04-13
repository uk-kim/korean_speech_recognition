"""
This scripts include dataset class for training stt model with ctc loss.
It has functions as data2pickle, pickle2tf_record, batch generate ...
"""

import os
import pickle
import random

import numpy as np
#from random import shuffle


class DataLoadFailException(Exception):
    def __init__(self):
        self.msg = "Couldn't Load Dataset. Plz check and try again."


class FileNotExistsException(Exception):
    def __init__(self, fname):
        self.msg = "File not exists. : {}".format(fname)


class AIHubDataSets:
    def __init__(self, file_list=[], label_list=[], feature_list=[]):
        self.file_list    = file_list
        self.label_list   = label_list
        self.feature_list = feature_list
        self.n_data       = len(self.file_list)
        
        self.idx = 0

    def load_datasets(self, data_dir=None, pickle_dir=None, records_dir=None):
        if data_dir:
            1
        elif pickle_dir:
            # data = (file_list, feature_list, label_list)
            data = self.load_pickle_data(pickle_dir)
        elif records_dir:
            raise DataLoadFailException()
        else:
            raise DataLoadFailException()

        self.file_list    = data[0]
        self.feature_list = data[1]
        self.label_list   = data[2]

    def load_pickle_data(self, file_path):
        """
        Pickle file is composed as (file_list, label_list, feature_list)
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            raise FileNotExistException(file_path)
        return data


    def save_as_pickle(self, file_path):
        data = (self.file_list, self.label_list, self.feature_list)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    
    def __shuffle__(self, seed=299496):
        if seed:
            random.seed(seed)
        data = list(zip(self.file_list, self.label_list, self.feature_list))
        random.shuffle(data)
        file_list, label_list, feature_list = zip(*data)

        self.file_list    = file_list
        self.label_list   = label_list
        self.feature_list = feature_list
    

    def next_batch(self, batch_size, padding=True):
        """
         - label_list   : [Batch Size, Seq Length]     <- List type
         - feature_list : [Batch Size, Seq Length, Feature dim]    <- List type

        Without padding, each data have different sequence length.
        With padding, find maximum sequence length and zero-pad to data with smaller seq len 
        """
        if batch_size == -1:
            file_list    = self.file_list
            label_list   = self.label_list
            feature_list = self.feature_list
        else:
            file_list    = self.file_list[self.idx: self.idx + batch_size]
            label_list   = self.label_list[self.idx: self.idx + batch_size]
            feature_list = self.feature_list[self.idx: self.idx + batch_size]
               
            if self.idx + batch_size >= len(self.file_list):
                self.idx = 0
                self.__shuffle__()
            else:
                self.idx += batch_size

        if padding:
            feature_dim = feature_list[0].shape[1]
            label_seq   = list(map(len, label_list))
            feature_seq = list(map(lambda x: x.shape[0], feature_list)) 

            feature_np  = np.zeros((len(file_list), max(feature_seq), feature_dim), np.float32)
            for i, l in enumerate(feature_seq):
                feature_np[i, :l, :] = feature_list[i]

            label_np = np.zeros((len(file_list), max(label_seq)), np.int)
            for i, l in enumerate(label_list):
                label_np[i, :len(l)] = np.array(label_list[i], dtype=np.int)

            return file_list, label_np, feature_np
        else:
            return file_list, label_list, feature_list
        
