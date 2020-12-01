from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import seaborn as sns
from design import Dataset
from sklearn.preprocessing import StandardScaler
import copy


class SyntheticDataCreator(Dataset):
    """ create synthetic data
    
        Args:
            :attr:`num_tasks` (int): 
                number of tasks to create
            :attr:`cellsPerTask` (int): 
                number of cells to make for each task, this is dimension n of our ratings matrix
            :attr:`drugsPerTask` (int): 
                number of drugs to make for each task, this is dimension m of our ratings matrix
            :attr:`sparsityPct` (int (0,100)): 
                this is the amount of sparsity to put on the ratings matrix, the a higher percentage corresponds to a more sparse prediction matrix. Must be between 0 and 100.
            :attr:`function` (string): 
                gaussian or cosine indicating recipe for synthetic data
            :attr:`normalize` (boolean):
                Boolean, whether data should be normalized
            :attr:`test_split` (float [0,1]): 
                determines size of training and testing data
            :attr:`noise` (boolean):
                amount of noise to use in creating synthetic data, the higher this value is the less correlation between generated tasks

        Returns object with:
            :attr:`self.datasets` (list): 
                list of strings indicating dataset names
            :attr:`self.data` (dict): 
                multilevel dictionary with keys for train/test, then keys for x,y, then finally keys for dataset name.
                ie: self.data['train']['x'][name1] gives training data for task 1. The models are built correspondingly.
            :attr:`self.trainRatings` (np.array): 
                array of ratings with shape (n,m) where n is the number of training cells / task and m is the number training of drugs / task 
           :attr:`self.testRating` (np.array): 
                array of ratings with shape (n,m) where n is the number of test cells / task and m is the number training of drugs / task 
            
    """

    def __init__(self, num_tasks=1, cellsPerTask=300, drugsPerTask=10, function="gaussian",
                 normalize=True, noise=.1, graph=False, test_split=0.3, **kwargs):
        self.num_tasks = num_tasks
        self.function = function
        self.test_split = test_split
        self.normalize = normalize
        self.noise = noise
        self.graph = graph
        self.num_drugs = drugsPerTask
        self.num_cells = cellsPerTask
        self.ptsPerTask = cellsPerTask * drugsPerTask
        self.cat_point = 10 # placeholder
        self.datasets = [str(i) for i in range(num_tasks)]
        self.read_optional_params()
        
       
    def set_test_split(self, new_test_split):
        """ 
        Update the dict of test split. 
        
        """
        self.test_split = new_test_split
        
    def read_optional_params(self):
        
        # program invoked from experiments folder so chdir one dir up
        with open("../datasets/optional_params.txt","r") as params:
            line = params.readline()
            while line:
                key, val = line.split("=")
                try: 
                    val = int(val)
                except ValueError:
                    print("please give integer args for feature numbers")
                setattr(self,key, val)
                line = params.readline()


    def prepare_data(self):
        """
        Run this public method to prepare data
        """
        if not ("cos" in self.function or "cosine" in self.function):
            self.data = self.generateSynthData(self.num_tasks, self.ptsPerTask, self.noise, self.graph)
            self.trainRatings = {}
            self.testRatings = {}
            train_split = 1 - self.test_split
            for task in self.datasets:
                self.trainRatings[task]=self.data['train']['y'][task].reshape(int(self.num_cells*train_split), self.num_drugs)
                self.testRatings[task]=self.data['test']['y'][task].reshape(int(self.num_cells*self.test_split),self.num_drugs)

        else:
            self.data = self.generateCosSynthData(self.num_tasks, self.ptsPerTask, self.noise, self.graph)
            self.trainRatings = {}
            self.testRatings = {}
            train_split = 1 - self.test_split
            for task in self.datasets:
                self.trainRatings[task]=self.data['train']['y'][task].reshape(int(self.num_cells*train_split), self.num_drugs)
                self.testRatings[task]=self.data['test']['y'][task].reshape(int(self.num_cells*self.test_split),self.num_drugs)

      

    def create_x_and_y(self):   
        """
        Wrapper for prepare data
        
        """
        self.prepare_data()


    def shuffle_and_split(self):
        """
        As stated in the name, shuffles and splits data again. Will fail if data has not been initialized
        
        """
        assert self.data
        train_split = 1 - self.test_split
        for task in self.datasets:
            X = np.append(self.data['train']['x'][task], self.data['test']['x'][task], axis =0)
            y = np.append(self.data['train']['y'][task], self.data['test']['y'][task], axis =0)
            shuffleX = np.random.permutation(len(X))
            shuffleY = np.random.permutation(len(y))
            X = X[shuffleX, :]
            y = y[shuffleY]
            self.test_split = round(self.test_split, 2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_split)
            self.data['train']['x'][task] = X_train
            self.data['train']['y'][task] = y_train
            self.data['test']['x'][task] = X_test
            self.data['test']['y'][task] = y_test
            self.trainRatings[task]= self.data['train']['y'][task].reshape(int(self.num_cells*train_split), self.num_drugs)
            self.testRatings[task]= self.data['test']['y'][task].reshape(int(self.num_cells*self.test_split),self.num_drugs)
            
        
    def generateCosSynthData(self, num_tasks=1,ptsPerTask=1000, noise=.1,graph=False):
        """
        Method used to generate synthetic data with the cosine function. 
        This function selects set of uniform points on interval 0 to 1 and scales them each on intervals of 2c*pi
        Where c is in range [1,nfeatures] user can set nfeatures in optional_params.txt. Each feature maps 
        to the same y value because they are shfited by one period. Then finally, some noise is added to each y,
        in order to control the correlation between tasks. The more noise --> the less correlation.
        
        """
        synth_datasets = {'train': {'x': {}, 'y': {}}, 'test': {'x': {}, 'y': {}}}
        collectXtrain = {}
        collectYtrain = {}
        collectXtest = {}
        collectYtest = {}
        names  = []
        #normal dist, with mean 0 scale 1
        input_data = np.random.uniform(low=0, high=1, size=ptsPerTask)
        total_features = self.cell_features_cosine + self.drug_features_cosine
        self.cat_point = self.cell_features_cosine
        scaling_points = np.arange(2,(total_features+1)*2, 2)  
        for i in range(num_tasks):
            X = np.outer(input_data, (scaling_points * math.pi)) # each point moved along one period, 2pi, 4pi etc.
            y = np.cos(X[:,0]) + np.random.normal(size=X.shape[0], scale=noise) * 0.2
            if self.normalize:
                X = StandardScaler().fit_transform(X)
                y = StandardScaler().fit_transform(y.reshape(-1,1))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_split)
            synth_datasets['train']['x'][str(i)] = X_train
            synth_datasets['train']['y'][str(i)] = y_train
            synth_datasets['test']['x'][str(i)] = X_test
            synth_datasets['test']['y'][str(i)] = y_test
            names.append(str(i))
        #print(1/0)
        return synth_datasets
    
    def generateSynthData(self, num_tasks=1, ptsPerTask=1000, noise=.1,graph=False):
        """
        Generates gaussian synthetic data. Coefficients apply common linear transformation to 
        multivariate gaussian vectors. Number of gaussian vectors = nfeatures and cna be changed in optional_params.txt
        to add/remove features. Some noise added to coefficients to control correlation, similar to cosine function
        more noise --> less correlation.
        
        """
        synth_datasets = {'train': {'x': {}, 'y': {}}, 'test': {'x': {}, 'y': {}}}
        collectXtrain = {}
        collectYtrain = {}
        collectXtest = {}
        collectYtest = {}
        names  = []
        #normal dist, with mean 0 scale 1
        total_features = self.cell_features_gauss + self.drug_features_gauss
        self.cat_point = self.cell_features_gauss
        input_data = np.random.normal(size=(ptsPerTask,total_features))
        coeffs = np.random.normal(size=(total_features))
        for i in range(num_tasks):
            noisy_coeffs = coeffs +  np.random.normal(size=(total_features), scale = noise)
            X = input_data * noisy_coeffs
            y = np.sum(X,axis=1)
            if self.normalize:
                X = StandardScaler().fit_transform(X)
                y = StandardScaler().fit_transform(y.reshape(-1,1))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_split, random_state=42)
            synth_datasets['train']['x'][str(i)] = X_train
            synth_datasets['train']['y'][str(i)] = y_train
            synth_datasets['test']['x'][str(i)] = X_test
            synth_datasets['test']['y'][str(i)] = y_test
            names.append(str(i))
        return synth_datasets
    