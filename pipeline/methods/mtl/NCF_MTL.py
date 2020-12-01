"""
Uses: Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)

@adapted: Xander Ladd
"""
from ..base import BaseMTLEstimator
import datetime
import numpy as np
import tensorflow as tf
import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Concatenate, Flatten, Dropout, Multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from time import time
import sys
import argparse
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
import math





def init_normal(shape, dtype=None, name=None):
    #return initializations.normal(shape, scale=0.01, name=name)
    return tf.random.normal(
    shape, mean=0.0, stddev=1, dtype=dtype, name=name)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #print("lrate", lrate)
    return lrate


class Neural_Collaborative_Filtering_FeaturesMTLMLP(BaseMTLEstimator):
    """
    NCF adapted for multitask model that first trains shared MLP on pooled data, then trains a seperate GMF model for each task.
    
    Combines matrix factorization and Multilayer Perceptron. **Uses cell and drug features**
    
    Args:
        :attr:`hyperparams` (dict):
            dictionary containing keys for each hyperparameter.
            :attr:`num_epochs` (int):
                number of epochs to train for
            :attr:`batch_size` (int):
                size of each batch in training epochs
            :attr:`mf_dim` (int):
                number of factors to be used by matrix factorization
            :attr:`layers` (list):
                list describing architecure for multilayer perceptron. ie: [32,16,8]
            :attr:`reg_mf` (float):
                regularization penalty for matrix factorization
            :attr:`reg_layer` (list):
                list describing architecure for regularizing multilayer perceptron. ie: [32,16,8]. Must match length of layers
            :attr:`learning_rate` (int):
                learning rate for gradient descent weight optimization
            :attr:`learner` (string):
                name of learner to use. Options are sgd, adam, rmsprop, decayed sgd, scheduled sgd, adagrad
            :attr:`mlp_lr` (float):
                (0,1) float for learning rate for pooled MLP model.
                
         :attr:`warmstart` (boolean):
             whether to instantiate a model for each task or keep training the same one.
         
        Ignore the other arguments, they are there to pass in hyperparameters with optuna.
    """
    def __init__(self, hyperparams,name="Neural_Collaborative_Filtering_Features", type_met='feature_based', paradigm='mtl',
                 output_shape=None, warm_start=False, learner=None, learning_rate=None, reg_mf=None, num_factors=None):
        """
        Class initialization.
        Args
            name (str): name to be used as reference
            paradigm:  either mtl or stl

        """
        super().__init__(name, type_met=type_met)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks_list = [ tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.01,patience=3)]#[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

        self.num_epochs = hyperparams['epochs']
        self.batch_size = hyperparams['batch_size']
        self.mf_dim = hyperparams['num_factors']
        self.layers = eval(hyperparams['layers'])
        self.reg_mf = hyperparams['reg_mf']
        self.reg_layers = eval(hyperparams['reg_layers'])
        self.learning_rate = hyperparams['lr']
        self.learner = hyperparams['learner']
        self.mlp_learning_rate = hyperparams['mlp_lr']
        self.num_threads = 1
        self.output_shape = 'array'
        self.mlp_dropout = True
        self.trial =None
        #accomodate for the possibility of being initialized with hyperparams
        if learner is not None and learning_rate is not None and reg_mf is not None and num_factors is not None:
            print("Featurized NCF is using optuna hyperparams from best trial now")
            self.learner = learner
            self.learning_rate = learning_rate
            self.reg_mf = reg_mf
            self.mf_dim = num_factors

#         if self.learning_rate != 0:
#             self.models = self.get_models(self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        self.warm_start = warm_start
        print("model has been defined")

    def __str__(self):
        return self.name
    

    
    def get_mlp(self, layers=[10], reg_layers=[0]):
        
        assert len(layers) == len(reg_layers)
        
        num_layer = len(layers)  # Number of layers in the MLP
        mlp_layers = []
        models = []
    
        # Input variables
        cell_input = Input(shape=(self.cat_point,), dtype='float32', name='cell_input')
        drug_input = Input(shape=(self.featurelen - self.cat_point,), dtype='float32', name='drug_input')
        # MLP part
        mlp_user_latent = Flatten()(cell_input)
        mlp_item_latent = Flatten()(drug_input)
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        
        for idx in range(0, num_layer):
            layer = Dense(layers[idx], kernel_regularizer=l1(reg_layers[idx]), activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)
            
        if self.mlp_dropout:
            mlp_vector = Dropout(.2)(mlp_vector)

        
        mlp_model = Model([cell_input,drug_input], mlp_vector)
                
        mlp_model.compile(optimizer=Adam(lr=self.mlp_learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
        return mlp_model
        
    
    def get_models(self, mlp, num_models ,mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
        """
        Function to constuct a bunch of NCF models with a shared multilayer perceptron
        
        args
        -----------------------------------------------------------------
        num_models = number of models to make
        mf_dim: rank for approximate decompostion for matrix factorization
        layers: list of layer sizes for MLP
        reg_layers: list of regularization layer values, should not be bigger than layers
        reg_mf: penalty on matrix factorization
        
        returns
        -----------------------------------------------------------------
        TODO
        """
        
        models = []
        for model_idx  in range(num_models):
            # MF part
            cell_input = Input(shape=(self.cat_point,), dtype='float32', name='cell_ft')
            drug_input = Input(shape=(self.featurelen - self.cat_point,), dtype='float32', name='drug_ft')
            cell_latent = Dense(mf_dim, activation='linear', kernel_regularizer=l1(reg_mf) ,name="latent_cell")(cell_input)
            drug_latent = Dense(mf_dim, activation='linear', kernel_regularizer=l1(reg_mf), name="latent_drug")(drug_input)

            mf_cell_latent = Flatten()(cell_latent)
            mf_drug_latent = Flatten()(drug_latent)
            mf_vector = Multiply()([mf_cell_latent, mf_drug_latent]) 
            
            mlp_out = mlp([cell_input,drug_input]) 
            
            predict_vector = Concatenate()([mf_vector,mlp_out])

            # Final prediction layer, dropout not in use because there is more data in this version, currently underfitting not overfitting 
            predict_vector = Dropout(.2)(predict_vector)
            prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

            model = Model(inputs={'cell_input':cell_input,'drug_input':drug_input},outputs=prediction)
  
            if self.learner.lower() == "adagrad":
                model.compile(optimizer=Adagrad(lr=self.learning_rate), loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
            elif self.learner.lower() == "rmsprop":
                model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
            elif self.learner.lower() == "adam":
                model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
            elif self.learner.lower() == "scheduled sgd":
                model.compile(optimizer=SGD(momentum=0.9), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
                lrate = LearningRateScheduler(step_decay)
                self.callbacks_list = [lrate]
            elif self.learner.lower() == "decayed sgd":
                learning_rate = 0.00001
                decay_rate = learning_rate / 30
                momentum = 0.9
                sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
                model.compile(optimizer=sgd, loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
            else:
                model.compile(optimizer=SGD(lr=self.learning_rate), loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
                
            models.append(model)
        
        return models
    
    def _pool_data(self, x,y):
        """
        Convenience function to pool data from each task. Also shuffles the data for good measure.
        Args:
        :attr:`x` (dict):
            dictionary containing keys corresponding to each dataset.
        :attr:`y` (dict):
            dictionary containing keys corresponding to each dataset.
        Returns:
            pooled x and y data
        """
        # pool data, maybe make this a function
        counter = 0
        for task in x.keys():
            if counter == 0:
                x_pool = np.array(x[task])
                y_pool = np.array(y[task])
            else:
                x_pool = np.append(x_pool,x[task], axis=0)
                y_pool = np.append(y_pool,y[task], axis=0)
            counter += 1
            
        #shuffle pooled data    
        shuffler = np.random.permutation(len(x_pool))
        x_pool = x_pool[shuffler]
        y_pool = y_pool[shuffler]
        return x_pool, y_pool


    def fit(self, x, y=None, cat_point=None):
        """
        Train method's parameters for multitask data. x and y should be a dictionary with the keys as the datapoints for each set of tasks
        Args
        -----------------------------------------------------------------------
            x: cell line features and drug feature
            y: if we are doing feature based we would need y as the ratings and X would be features instead of ratings
            
        Returns
        -----------------------------------------------------------------------
            one number for training error
        """
        self.cat_point = cat_point # get concatentation point on feature vec
        self.featurelen = x[list(x.keys())[0]].shape[1] # get full length of concatenated feature vec
        ntasks = len(x.keys())
        self.mlp_model =  self.get_mlp(self.layers, self.reg_layers)
        x_pool, y_pool = self._pool_data(x,y)
        
        #train mlp
        mlp_hist = self.mlp_model.fit([np.array(x_pool[:,:self.cat_point]), np.array(x_pool[:,self.cat_point:])], np.array(y_pool), batch_size=self.batch_size, epochs=self.num_epochs, verbose=0, shuffle=True, callbacks= self.callbacks_list)                

        #freeze backprop on mlp
        for layer in self.mlp_model.layers:
            layer.trainable = False
        self.mlp_model.trainable = False
        model_idx = 0
        
        self.models = self.get_models( self.mlp_model,ntasks,self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        for task in x.keys():
            model = self.models[model_idx]
            model_idx += 1
            x_task = x[task]
            y_task = y[task]
            hist = model.fit({'cell_input':np.array(x_task[:,:self.cat_point]), 'drug_input':np.array(x_task[:,self.cat_point:])}, np.array(y_task), batch_size=self.batch_size, epochs=self.num_epochs, verbose=0, shuffle=True, callbacks= self.callbacks_list)

            
           

    def predict(self, x, y=None, cat_point=None):
        """
        predict rating matrix given features
        
        Args
        -----------------------------------------------------------------------
            x: cell line features and drug feature
            y: if we are doing feature based we would need y as the ratings and X would be features instead of ratings
            
        Returns
        -----------------------------------------------------------------------
            predictions: ic50 values for each drug/celll line combo
        """
        print("PREDICTING.....")
        res = {}
        model_idx = 0 
        for task in x.keys():
            x_task =  x[task]
            model = self.models[model_idx]
            predictions = model.predict({'cell_input':x_task[:,:self.cat_point],'drug_input':x_task[:,self.cat_point:]}, batch_size=self.batch_size, verbose=0)
            model_idx += 1
            res[task] = predictions
        return res
    
    # Optuna decorators
    def get_hyper_params(self):
        hparams = {'num_factors': {'type': 'integer', 'values': [4, 100]},
                   'reg_mf': {'type': 'loguniform', 'values': [.00001,.01]},
                   'learning_rate': {'type': 'loguniform', 'values':  [.00001,.01]},
                   'learner': {'type': 'categorical', 'values':  ['sgd','adam','rmsprop', 'adagrad']},
                   'epochs': {'type': 'integer', 'values': [100, 500]}
                  }
        return hparams

    def set_hyper_params(self, **kwargs):
        self.mf_dim = kwargs['num_factors']
        self.reg_mf = kwargs['reg_mf']
        self.learning_rate = kwargs['learning_rate']
        self.learner = kwargs['learner']
        self.num_epochs = kwargs['epochs']
             
        
        
class Neural_Collaborative_Filtering_FeaturesMTLMF(BaseMTLEstimator):
    """
    NCF adapted for multitask model that first trains shared MF on pooled data, then trains a seperate MLP model for each task.
    
    Combines matrix factorization and Multilayer Perceptron. **Uses cell and drug features**
    
    Args:
        :attr:`hyperparams` (dict):
            dictionary containing keys for each hyperparameter.
            
            :attr:`num_epochs` (int):
                number of epochs to train for
            :attr:`batch_size` (int):
                size of each batch in training epochs
            :attr:`mf_dim` (int):
                number of factors to be used by matrix factorization
            :attr:`layers` (list):
                list describing architecure for multilayer perceptron. ie: [32,16,8]
            :attr:`reg_mf` (float):
                regularization penalty for matrix factorization
            :attr:`reg_layer` (list):
                list describing architecure for regularizing multilayer perceptron. ie: [32,16,8]. Must match length of layers
            :attr:`learning_rate` (int):
                learning rate for gradient descent weight optimization
            :attr:`learner` (string):
                name of learner to use. Options are sgd, adam, rmsprop, decayed sgd, scheduled sgd, adagrad
            :attr:`mf_lr` (float):
                (0,1) float for learning rate for pooled MF model.
                
         :attr:`warmstart` (boolean):
             whether to instantiate a model for each task or keep training the same one.
         
        Ignore the other arguments, they are there to pass in hyperparameters with optuna.
    """
    
    def __init__(self, hyperparams,name="Neural_Collaborative_Filtering_Features", type_met='feature_based', paradigm='mtl',
                 output_shape=None, warm_start=False, learner=None, learning_rate=None, reg_mf=None, num_factors=None):
        """
        Class initialization.
        Args
            name (str): name to be used as reference
            paradigm:  either mtl or stl

        """
        super().__init__(name, type_met=type_met)
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # FOR USE WITH TENSORBOARD
        self.callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.01,patience=3)]#[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
        self.num_epochs = hyperparams['epochs']
        self.batch_size = hyperparams['batch_size']
        self.mf_dim = hyperparams['num_factors']
        self.layers = eval(hyperparams['layers'])
        self.reg_mf = hyperparams['reg_mf']
        self.reg_layers = eval(hyperparams['reg_layers'])
        self.learning_rate = hyperparams['lr']
        self.learner = hyperparams['learner']
        self.mf_learning_rate = hyperparams['mf_lr']
        self.num_threads = 1
        self.output_shape = 'array'
        self.trial =None
        
        #accomodate for the possibility of being initialized with hyperparams
        if learner is not None and learning_rate is not None and reg_mf is not None and num_factors is not None:
            print("Featurized NCF is using optuna hyperparams from best trial now")
            self.learner = learner
            self.learning_rate = learning_rate
            self.reg_mf = reg_mf
            self.mf_dim = num_factors

        self.warm_start = warm_start
    
    def get_mf(self, mf_dim=10,reg_mf=0):
        cell_input = Input(shape=(self.cat_point,), dtype='float32', name='cell_ft')
        drug_input = Input(shape=(self.featurelen - self.cat_point,), dtype='float32', name='drug_ft')
        cell_latent_layer = Dense(mf_dim, activation='linear', kernel_regularizer=l1(reg_mf),name="cell_latent")
        cell_latent = cell_latent_layer(cell_input)
        drug_latent_layer = Dense(mf_dim, activation='linear', kernel_regularizer=l1(reg_mf), name="drug_latent")
        drug_latent = drug_latent_layer(drug_input)
        mf_cell_latent = Flatten()(cell_latent)
        mf_drug_latent = Flatten()(drug_latent)
        mf_vector = Multiply()([mf_cell_latent, mf_drug_latent]) 
        mf_model = Model([cell_input,drug_input], mf_vector)
        mf_model.compile(optimizer=Adam(lr=self.mf_learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return mf_model, cell_latent_layer, drug_latent_layer
        
    
    def get_models(self, mf, num_models, cell_latent_layer, drug_latent_layer ,mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0,):
        """
        Function to constuct a bunch of NCF models with a shared multilayer perceptron
        
        args
        -----------------------------------------------------------------
        num_models = number of models to make
        mf_dim: rank for approximate decompostion for matrix factorization
        layers: list of layer sizes for MLP
        reg_layers: list of regularization layer values, should not be bigger than layers
        reg_mf: penalty on matrix factorization
        
        returns
        -----------------------------------------------------------------
        TODO
        """
        
        
        assert len(layers) == len(reg_layers)
        
        num_layer = len(layers)  # Number of layers in the MLP
        mlp_layers = []
        models = []
        

        for model_idx  in range(num_models):
             # Input variables
            cell_input = Input(shape=(self.cat_point,), dtype='float32', name='cellll')
            drug_input = Input(shape=(self.featurelen - self.cat_point,), dtype='float32', name='drug')
            # MLP part
            #cell_latent = cell_latent_layer(cell_input)
            #drug_latent = drug_latent_layer(drug_input)
            mlp_cell_latent = Flatten()(cell_input)
            mlp_drug_latent = Flatten()(drug_input)
            mlp_vector = Concatenate()([mlp_cell_latent, mlp_drug_latent])

            for idx in range(0, num_layer):
                layer = Dense(layers[idx], kernel_regularizer=l1(reg_layers[idx]), activation='relu', name="layer%d" % idx)
                mlp_vector = layer(mlp_vector)
            
            mf_out = mf([cell_input,drug_input]) 
            
            predict_vector = Concatenate()([mf_out,mlp_vector])

            # Final prediction layer, dropout not in use because there is more data in this version, currently underfitting not overfitting 
            #predict_vector_dropout = Dropout(.2)(predict_vector)
            prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

            model = Model(inputs={'cell_input':cell_input,'drug_input':drug_input},outputs=prediction)
  
            # not to use binary crossentropy with regression
            if self.learner.lower() == "adagrad":
                model.compile(optimizer=Adagrad(lr=self.learning_rate), loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
            elif self.learner.lower() == "rmsprop":
                model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
            elif self.learner.lower() == "adam":
                model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
            elif self.learner.lower() == "scheduled sgd":
                model.compile(optimizer=SGD(momentum=0.9), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
                lrate = LearningRateScheduler(step_decay)
                self.callbacks_list = [lrate]
            elif self.learner.lower() == "decayed sgd":
                learning_rate = 0.00001
                decay_rate = learning_rate / 30
                momentum = 0.9
                sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
                model.compile(optimizer=sgd, loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
            else:
                model.compile(optimizer=SGD(lr=self.learning_rate), loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
                
            models.append(model)
        
#         mlp_model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return models


    def _pool_data(self, x,y):
        """
        Convenience function to pool data from each task. Also shuffles the data for good measure.
        Args:
        :attr:`x` (dict):
            dictionary containing keys corresponding to each dataset.
        :attr:`y` (dict):
            dictionary containing keys corresponding to each dataset.
        Returns:
            pooled x and y data
        """
        # pool data, maybe make this a function
        counter = 0
        for task in x.keys():
            if counter == 0:
                x_pool = np.array(x[task])
                y_pool = np.array(y[task])
            else:
                x_pool = np.append(x_pool,x[task], axis=0)
                y_pool = np.append(y_pool,y[task], axis=0)
            counter += 1
            
        #shuffle pooled data    
        shuffler = np.random.permutation(len(x_pool))
        x_pool = x_pool[shuffler]
        y_pool = y_pool[shuffler]
        return x_pool, y_pool


    def fit(self, x, y=None, cat_point=None):
        """
        Train method's parameters for multitask data. x and y should be a dictionary with the keys as the datapoints for each set of tasks
        Args
        -----------------------------------------------------------------------
            x: cell line features and drug feature
            y: ratings vector corresponding to cell line and drug features 
            
            
        """
        self.cat_point = cat_point # get concatentation point on feature vec
        self.featurelen = x[list(x.keys())[0]].shape[1] # get full length of concatenated feature vec
        ntasks = len(x.keys())
        self.mf_model, cell_latent_layer, drug_latent_layer =  self.get_mf(self.mf_dim, self.reg_mf) 
        
        #pooling
        x_pool, y_pool = self._pool_data(x,y)
        errors = []
        shuffler = np.random.permutation(len(x_pool))
        x_pool = x_pool[shuffler]
        y_pool = y_pool[shuffler]
        
        #train MF Pooled
        hist = self.mf_model.fit([np.array(x_pool[:,:self.cat_point]), np.array(x_pool[:,self.cat_point:])], np.array(y_pool), batch_size=self.batch_size, epochs=self.num_epochs, verbose=0, shuffle=True, callbacks= self.callbacks_list)                
        
        # turn off MF training
        for layer in self.mf_model.layers:
            layer.trainable = False
        self.mf_model.trainable = False
                
        #get all models
        self.models = self.get_models( self.mf_model, ntasks, cell_latent_layer, drug_latent_layer, self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        
        model_idx = 0
        for task in x.keys():
            model = self.models[model_idx]
            model_idx += 1
            x_task = x[task]
            y_task = y[task]
            # Training model
            hist = model.fit({'cell_input':np.array(x_task[:,:self.cat_point]), 'drug_input':np.array(x_task[:,self.cat_point:])}, np.array(y_task), batch_size=self.batch_size, epochs=self.num_epochs, verbose=0, shuffle=True, callbacks= self.callbacks_list)

                    
    def predict(self, x, y=None, cat_point=None):
        """
        predict rating matrix given features
        
        Args
        -----------------------------------------------------------------------
            x: cell line features and drug feature
            y: if we are doing feature based we would need y as the ratings and X would be features instead of ratings
            
        Returns
        -----------------------------------------------------------------------
            predictions: ic50 values for each drug/celll line combo
        """
        print("PREDICTING.....")
        res = {}
        model_idx = 0 
        for task in x.keys():
            x_task =  x[task]
            model = self.models[model_idx]
            predictions = model.predict({'cell_input':x_task[:,:self.cat_point],'drug_input':x_task[:,self.cat_point:]}, batch_size=self.batch_size, verbose=0)
            model_idx += 1
            res[task] = predictions

        return res
    
 
    # Optuna decorators
    def get_hyper_params(self):
        hparams = {'num_factors': {'type': 'integer', 'values': [4, 100]},
                   'reg_mf': {'type': 'loguniform', 'values': [.00001,.01]},
                   'learning_rate': {'type': 'loguniform', 'values':  [.00001,.01]},
                   'learner': {'type': 'categorical', 'values':  ['sgd','adam','rmsprop', 'adagrad']},
                   'epochs': {'type': 'integer', 'values': [100, 500]},
                   'mf_lr': {'type': 'loguniform', 'values':  [.00001,.01]}
                  }
        return hparams

    def set_hyper_params(self, **kwargs):
        self.mf_dim = kwargs['num_factors']
        self.reg_mf = kwargs['reg_mf']
        self.learning_rate = kwargs['learning_rate']
        self.learner = kwargs['learner']
        self.num_epochs = kwargs['epochs']
        self.mf_lr = kwargs['mf_lr']
        