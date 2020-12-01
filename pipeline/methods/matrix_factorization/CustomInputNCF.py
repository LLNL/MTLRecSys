"""
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)

@editor: Xander Ladd

NOTE: USERS == LINES DRUGS == ITEMS
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




def init_normal(shape, dtype=None, name=None):
    #return initializations.normal(shape, scale=0.01, name=name)
    return tf.random.normal(
    shape, mean=0.0, stddev=.01, dtype=dtype, name=name)


class Neural_Collaborative_Filtering(BaseEstimator):
    """
    Neural Collaborative Filtering adapted for regression from https://github.com/hexiangnan/neural_collaborative_filtering
    
    Combines matrix factorization and Multilayer Perceptron. **Does not use cell and drug features**
    
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
    
 
    def __init__(self, hyperparams, name="Neural_Collaborative_Filtering", type_met='non_feature_based', paradigm='stl',
                 output_shape=None,warm_start=False, learner=None, learning_rate=None, reg_mf=None, num_factors=None):  
        """
        Reads initialization parameters, decides to override them with hyperoptimized ones and then gets model
        
        Parameters
        -------------------------------------------------------------------
        name (str): name to be used as reference
        type: non feature based since this is our ratingsmatrix one
        paradigm:  either mtl or stl
        warm start: True if you don't want a new model everytime you train-- needs to be True for HOPT
        the rest are model parameters
        """
        super().__init__(name, paradigm=paradigm, type_met=type_met, output_shape=output_shape)
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
        self.warm_start = warm_start
        #500? TODO: find a better way to initialize network struct
        if learner is not None and learning_rate is not None and reg_mf is not None and num_factors is not None:
            print("Non Featurized NCF is using optuna hyperparams from best trial now")
            self.learner = learner
            self.learning_rate = learning_rate
            self.reg_mf = reg_mf
            self.mf_dim = num_factors
        if self.learning_rate != 0: #check that we arent running hyperopt
            self.model = self.get_model(6000, 6000, self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        print("MODEL HAS BEEN DEFINED")
        self.output_shape = 'array'

    def __str__(self):
        return self.name
    
    def get_model(self, num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
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
        user_input = Input(shape=(1,), dtype='float32', name='user_input')
        item_input = Input(shape=(1,), dtype='float32', name='item_input')

        MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                      embeddings_initializer=init_normal, embeddings_regularizer=l1(reg_mf), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                      embeddings_initializer=init_normal, embeddings_regularizer=l1(reg_mf), input_length=1)

        MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                       embeddings_initializer=init_normal, embeddings_regularizer=l1(reg_layers[0]), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                       embeddings_initializer=init_normal, embeddings_regularizer=l1(reg_layers[0]), input_length=1)
#         MF_Embedding_User = Dense(mf_dim, kernel_regularizer=l2(reg_mf), activation='linear', name="mf_embedding_user")
#         MF_Embedding_Item = Dense(mf_dim, kernel_regularizer=l2(reg_mf), activation='linear', name="mf_embedding_item")
#         MLP_Embedding_User = Dense(int(layers[0] / 2), kernel_regularizer=l2(reg_mf), activation='linear', name="mlp_embedding_user")
#         MLP_Embedding_Item = Dense(int(layers[0] / 2), kernel_regularizer=l2(reg_mf), activation='linear', name="mlp_embedding_item")

  
        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        mf_vector = Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

        # MLP part
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        for idx in range(1, num_layer):
            layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)
            #MLP dropout, probably not useful (Added)
#             if idx == 2:
#                 mlp_vector =  Dropout(.5)(mlp_vector)

        # Concatenate MF and MLP parts
        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer
        predict_vector_dropout = Dropout(.5)(predict_vector) #prediciton dropout (Added)
        prediction = Dense(1, activation='linear', name="prediction", kernel_regularizer=l1(.01))(predict_vector_dropout)

        model = Model(inputs={'user_input':user_input,'item_input':item_input},
                      outputs=prediction)


        # not to use binary crossentropy with regression
        if self.learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=self.learning_rate), loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif self.learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif self.learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        else:
            model.compile(optimizer=SGD(lr=self.learning_rate,decay=.5), loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])

        return model


    def get_vectors(self, train, num_negatives=1):
        """ Can be helpful to refrence template NCF get_vectors with different methods for constructing vectors"""
        lines_input, drugs_input, labels = [], [], []
        num_lines = train.shape[0]
        num_drugs = train.shape[1]
        num_negatives = 1
        for i in range(num_lines):
            for j in range(num_drugs):
                if not np.isnan(train[i, j]):
                    lines_input.append(i)
                    drugs_input.append(j)
                    labels.append(train[i,j])
        return lines_input, drugs_input, np.reshape(labels, (-1,1))


    def fit(self, x, y=None, cat_point=None):
        """
        Train method's parameters. Splits up train ratings to vectors of index i, index j, and ratings
        Sends those to model
        
        Args
        -----------------------------------------------------------------------
            x: self.train_ratings[dataset] 
            
        Returns
        -----------------------------------------------------------------------
            one number for training error
        """
        # Training model
        num_lines, num_drugs = x.shape
        print("TRAINING....")
        if not self.warm_start:
            self.model = self.get_model(num_lines,num_drugs,self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
            print("NCF reinitialized")
        predictions = np.array([])
        num_negatives = 1
        errors = []
        for epoch in range(self.num_epochs):
            if epoch % 75 == 0:
                print(epoch)
            t1 = time()
            # Generate training instances
            cell_lines, drugs, labels = self.get_vectors(x, num_negatives)
            hist = self.model.fit({'user_input': np.array(cell_lines), 'item_input': np.array(drugs)},  # input
                                  np.array(labels),  # labels
                                  batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
            errors.append(hist.history['root_mean_squared_error'])
            if epoch > 10 and np.max(errors[epoch-10:epoch]) - np.min(errors[epoch-10:epoch]) > .1:
                break
            t2 = time()
            # record history here
        return hist.history['root_mean_squared_error']
           

    def predict(self, x, y=None, cat_point=None):
        """"
        Splits up test ratings to vectors of index i, index j, and ratings
        Sends those to model
        
        Args
        -----------------------------------------------------------------------
            x: self.test_ratings[dataset] 
        """
        print("PREDICTING.....")
        predictions = np.array([])
        cell_lines, drugs, labels = self.get_vectors(x, 0)
        predictions = self.model.predict({'user_input': np.array(cell_lines), 'item_input': np.array(drugs)},
                                          batch_size=self.batch_size, verbose=0)            
        return predictions
    
    def evaluate(self, x, y=None, cat_point=None,evaluate=False):
        """ 
        probably still works to return evaluation but not really used anywhere
        would want to input test_ratings and expect a rmse score out
        """
        
        print("evaluating.....")

        
        cell_lines, drugs, labels = self.get_vectors(x, 0)
        hist = self.model.evaluate([np.array(cell_lines), np.array(drugs)],  # input
                                      np.array(labels),  # labels
                                      batch_size=self.batch_size, verbose=0,return_dict=True)

       
        # benchmark with randomness
        #predictions = np.repeat(predictions[0], len(predictions))
        print( "KERAS RMSE: ", hist['root_mean_squared_error'])
        return hist['root_mean_squared_error']
    
    def plot(self, dataset,pdf):
        """ For plotting to check fit in train and test curves
        parameters. There is also a notebook that does the same thing.
        -------------------------------------------------------
        dataset: initial dataset object 
        pdf: pdf to write figures to
        
        return 
        --------------------------------------------------------
        train test curves
        """
        fig, axs = plt.subplots(4, figsize=(8,20))
        plot_counter = 0
        for k in dataset.datasets:
            self.train_rmses = []
            self.test_rmses = []

            for epoch in range(self.num_epochs):
                print("epoch : " , epoch)
                t1 = time()
                # Generate training instances
                train_x = dataset.trainRatings[k]
                cell_lines, drugs, labels = self.get_vectors(train_x, 0)
                test_x =  dataset.testRatings[k]
                test_cell_lines, test_drugs, test_labels = self.get_vectors(test_x, 0)
                #labels = x['binarized'].values
                train_hist = self.model.fit({'user_input': np.array(cell_lines), 'item_input': np.array(drugs)},  # input
                                      np.array(labels),  # labels
                                      batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
                test_hist = self.model.evaluate({'user_input': np.array(test_cell_lines), 'item_input': np.array(test_drugs)},  # input
                                      np.array(test_labels),  # labels
                                      batch_size=self.batch_size, verbose=0,return_dict=True)
                t2 = time()
                self.train_rmses.append(train_hist.history['root_mean_squared_error'])
                self.test_rmses.append(test_hist['root_mean_squared_error'])
                print("train: ",train_hist.history['root_mean_squared_error'], "test: ", test_hist['root_mean_squared_error'])
                if epoch > 10 and np.max(self.train_rmses[epoch-10:epoch] - np.min(self.train_rmses[epoch-10:epoch])) < .008:
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
            print('min train err: ',min(self.train_rmses), "min test err: ", min(self.test_rmses) )
            
        plt.tight_layout()
        pdf.savefig(fig)
        
             
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
