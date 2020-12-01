"""
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)

@adapted: Xander Ladd

NOTE: USERS == LINES DRUGS == ITEMS

Basic idea is that this is a neural network that uses multilayer perceptron
and matrix factorization then predicts IC50 
"""
from ..base import BaseEstimator
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


class Neural_Collaborative_Filtering_Features(BaseEstimator):
    """
    Neural Collaborative Filtering adapted from https://github.com/hexiangnan/neural_collaborative_filtering
    
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
            :attr:`num_negatives` :
                        ignore this, deprecated
            :attr:`learning_rate` (int):
                learning rate for gradient descent weight optimization
            :attr:`learner` (string):
                name of learner to use. Options are sgd, adam, rmsprop, decayed sgd, scheduled sgd, adagrad
                
         :attr:`warmstart` (boolean):
             whether to instantiate a model for each task or keep training the same one.
         
        Ignore the other arguments, they are there to pass in hyperparameters with optuna.
    """
    def __init__(self, hyperparams,name="Neural_Collaborative_Filtering_Features", type_met='feature_based', paradigm='stl',
                 output_shape=None, warm_start=False, learner=None, learning_rate=None, reg_mf=None, num_factors=None):
        """
        Class initialization.
        Args
            name (str): name to be used as reference
            paradigm:  either mtl or stl

        """
        super().__init__(name, paradigm=paradigm, type_met=type_met, output_shape=output_shape)
        
        self.callbacks_list = []
        self.num_epochs = hyperparams['epochs']
        self.batch_size = hyperparams['batch_size']
        self.mf_dim = hyperparams['num_factors']
        self.layers = eval(hyperparams['layers'])
        self.reg_mf = hyperparams['reg_mf']
        self.reg_layers = eval(hyperparams['reg_layers'])
        self.learning_rate = hyperparams['lr']
        self.learner = hyperparams['learner']
        self.verbose = hyperparams['verbose']
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

        if self.learning_rate != 0:
            self.model = self.get_model(self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        self.warm_start = warm_start
        print("model has been defined")

    def __str__(self):
        return self.name
    
    def get_model(self, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
        """
        Function to constuct a model with the correct dimensions/parameters
        inputs
        -----------------------------------------------------------------
        mf_dim: rank for approximate decompostion for matrix factorization
        layers: list of layer sizes for MLP
        reg_layers: list of regularization layer values, should not be bigger than layers
        reg_mf: penalty on matrix factorization 
        """
        assert len(layers) == len(reg_layers)
        num_layer = len(layers)  # Number of layers in the MLP
        # Input variables
        cell_input = Input(shape=(10,), dtype='float32', name='drug_input')
        drug_input = Input(shape=(10,), dtype='float32', name='cell_input')
        cell_fts = Dense(mf_dim, kernel_regularizer=l1(reg_mf), activation='linear', name="cell_embedding")(cell_input)
        drug_fts = Dense(mf_dim, kernel_regularizer=l1(reg_mf), activation='linear', name="drug_embedding")(drug_input)

        # MF part
        mf_user_latent = Flatten()(cell_fts)
        mf_item_latent = Flatten()(drug_fts)
        mf_vector = Multiply()([mf_user_latent, mf_item_latent]) 

        # MLP part
        mlp_user_latent = Flatten()(cell_fts)
        mlp_item_latent = Flatten()(drug_fts)
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        for idx in range(1, num_layer):
            layer = Dense(layers[idx], kernel_regularizer=l1(reg_layers[idx]), activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer, dropout not in use because there is more data in this version, currently underfitting not overfitting 
        #predict_vector_dropout = Dropout(.2)(predict_vector)
        prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs={'cell_input':cell_input,'drug_input':drug_input},
                      outputs=prediction)
#         model = tf.keras.Model(inputs=[user_input, item_input],
#                               output=prediction)

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

        return model


    def fit(self, x, y=None, cat_point=None):
        """
        Train method's parameters.
        Args
        -----------------------------------------------------------------------
            x: cell line features and drug feature
            y: if we are doing feature based we would need y as the ratings and X would be features instead of ratings
            
        Returns
        -----------------------------------------------------------------------
            one number for training error
        """
        # Training model
        print("TRAINING....")
        if not self.warm_start:
            self.model = self.get_model(self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
            print("NCF reinitialized")
        errors = []
        num_negatives = 1
        for epoch in range(self.num_epochs):
            t1 = time()
            hist = self.model.fit({'cell_input':np.array(x[:,:10]), 'drug_input':np.array(x[:,10:])}, np.array(y), batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True, callbacks= self.callbacks_list)                
            t2 = time()
            errors.append(hist.history['root_mean_squared_error'])
            if epoch > 10 and np.max(errors[epoch-10:epoch]) - np.min(errors[epoch-10:epoch]) < .05:
                print("break... model converged")
                break
           

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
        print(x.shape)
        predictions = self.model.predict({'cell_input':np.array(x[:,:10]), 'drug_input':np.array(x[:,10:])},
                                      batch_size=self.batch_size, verbose=0)

        return predictions
    
    def evaluate(self, x, y):
        # switch this into the pipeline version
        hist = self.model.evaluate({'cell_input':np.array(x[:,:10]), 'drug_input':np.array(x[:,10:])}, np.array(y),
                                    batch_size=self.batch_size, verbose=0,return_dict=True)
        return hist['root_mean_squared_error']
    
    def plot(self, dataset,pdf):
        """ For plotting to check fit in train and test. Writes plots to pdf
        Also have a notebook for this
        
        parameters
        ---------------------------------------------------------------
        dataset: dataset object
        pdf: pdf to write to
        """
        fig, axs = plt.subplots(4, figsize=(8,20))
        plot_counter = 0
        for k in dataset.datasets:
            self.train_rmses = []
            self.test_rmses = []
            if not self.warm_start:
                self.model = self.get_model(self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
                print("NCF reinitialized")
                
            for epoch in range(self.num_epochs):
                t1 = time()
                # Generate training instances
                train_x = dataset.data['train']['x'][k]
                train_y = dataset.data['train']['y'][k]
                test_x = dataset.data['test']['x'][k]
                test_y = dataset.data['test']['y'][k]
                train_hist = self.model.fit({'user_inputs':np.array(train_x[:,:10]), 'item_inputs':np.array(train_x[:,10:])}, np.array(train_y), batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
                test_hist = self.model.evaluate({'user_inputs':np.array(test_x[:,:10]), 'item_inputs':np.array(test_x[:,10:])}, np.array(test_y),
                                    batch_size=self.batch_size, verbose=0,return_dict=True)
                t2 = time()
                self.train_rmses.append(train_hist.history['root_mean_squared_error'])
                self.test_rmses.append(test_hist['root_mean_squared_error'])
                if epoch > 10 and np.max(self.train_rmses[epoch-10:epoch] - np.min(self.train_rmses[epoch-10:epoch])) < .03:
                    print("BREAK")
                    break
            axs[plot_counter].plot(self.train_rmses)
            axs[plot_counter].plot(self.test_rmses)
            axs[plot_counter].set_title(k)
            axs[plot_counter].legend(['train', 'validation'])
            axs[plot_counter].set_ylabel('RMSE')
            axs[plot_counter].set_xlabel('EPOCH')
            plot_counter += 1
            #axs[i].show()
            print('min train err: ', min(self.train_rmses), "min test err: ", min(self.test_rmses) )
            
        plt.tight_layout()
        pdf.savefig(fig)
        
        
    # Optuna decorators
    def get_hyper_params(self):
        hparams = {'num_factors': {'type': 'integer', 'values': [4, 100]},
                   'reg_mf': {'type': 'loguniform', 'values': [.00001,.01]},
                   'learning_rate': {'type': 'loguniform', 'values':  [.00001,.01]},
                  'learner': {'type': 'categorical', 'values':  ['sgd','adam','rmsprop', 'adagrad']}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.mf_dim = kwargs['num_factors']
        self.reg_mf = kwargs['reg_mf']
        self.learning_rate = kwargs['learning_rate']
        self.learner = kwargs['learner']

