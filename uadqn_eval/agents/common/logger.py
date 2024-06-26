#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:53:23 2019

@author: william
"""

import pickle
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
from datetime import datetime
import os


class Logger:
    def __init__(self,log_folder_details=None,train_details=None):
        self.memory = {}
        self.log_folder_details = log_folder_details
        self.train_details = train_details

        today = datetime.now()
        if self.log_folder_details is None:
            directory = 'results/'+today.strftime('%Y-%m-%d-%H%M%S')
        else:
            directory = os.path.join(self.log_folder_details,today.strftime('%Y-%m-%d-%H%M%S')) #'results/'+today.strftime('%Y-%m-%d-%H%M%S') + '-' + self.log_folder_details
            directory = self.log_folder_details

        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.log_folder = directory

        with open(self.log_folder + '/' + 'experimental-setup', 'w') as handle:
                pprint.pprint(self.train_details, handle)

    def add_scalar(self, name, data, timestep):
        """
        Saves a scalar
        """
        if isinstance(data, torch.Tensor):
            data = data.item()

        self.memory.setdefault(name, []).append([data, timestep])

    def save(self):
        filename = self.log_folder + '/log_data.pkl'
        
        with open(filename, 'wb') as output:
            pickle.dump(self.memory, output, pickle.HIGHEST_PROTOCOL)

        self.save_graphs()

    def save_graphs(self):
        for key in self.memory.keys():
            plt.cla()
            plt.plot(np.array(self.memory[key])[:,1],np.array(self.memory[key])[:,0])
            if self.log_folder is None:
                plt.savefig(key+'.png')
            else:
                plt.savefig(self.log_folder + '/' + key+'.png')

