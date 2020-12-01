import os, sys
import optuna
from cross_validation_score import cross_validation_score
from UTILS.utils import datasetParams2str
import numpy as np
import keras
from keras.backend import clear_session
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
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
import tensorflow as tf
from methods.matrix_factorization.CustomInputNCF import Neural_Collaborative_Filtering
from methods.matrix_factorization.FeaturizedNCF import Neural_Collaborative_Filtering_Features
import multiprocessing
import sqlite3

def init_normal(shape, dtype=None, name=None):
    #return initializations.normal(shape, scale=0.01, name=name)
    return tf.random.normal(
    shape, mean=0.0, stddev=1, dtype=dtype, name=name)

class Objective(object):
    def __init__(self, model, db):
        self.model = model
        self.db = db
        self.hp = self.model.get_hyper_params()

    def __call__(self, trial):

        params = {}
        for p in self.hp.keys():

            if self.hp[p]['type'] == 'categorical':
                params[p] = trial.suggest_categorical(p, self.hp[p]['values'])
            elif self.hp[p]['type'] == 'loguniform':
                params[p] = trial.suggest_loguniform(p, self.hp[p]['values'][0],
                                                     self.hp[p]['values'][1])
            elif self.hp[p]['type'] == 'integer':
                params[p] = trial.suggest_int(p, self.hp[p]['values'][0],
                                              self.hp[p]['values'][1])
            elif self.hp[p]['type'] == 'uniform':
                params[p] = trial.suggest_uniform(p, self.hp[p]['values'][0],
                                                  self.hp[p]['values'][1])
            else:
                raise ValueError('Unknown hyper-parameter type: {}'.format(self.hp[p]['type']))
        # print(params)
        self.model.set_hyper_params(**params)
        score = cross_validation_score(self.model, self.db, num_folds=1) # change back to 5 for non GPML models

        return score


def optimize_hyper_params(model, dataset, n_trials=100, direction='minimize', NCF_arch=False):
    # nb: when the code is in place for ncdg optimization make 
    # sure to put direction to maximize

    if NCF_arch:
        objective = NCFObjective(model, dataset)
    else:
        objective = Objective(model, dataset)

    datasetName = datasetParams2str(dataset.__dict__)
    study_name = '{}_{}'.format(model.name,datasetName)
    
    if not os.path.exists('hyperparam_experiments'):
        os.makedirs('hyperparam_experiments')
        
    #prevent errors from too much database concurrency
    sqlite3.connect('hyperparam_experiments/{}.db'.format(study_name),timeout=600)
    study = optuna.create_study(study_name=study_name,
                                storage='sqlite:///hyperparam_experiments/{}.db'.format(study_name),
                                direction=direction, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner())
    
    print("using ", multiprocessing.cpu_count(), " cpus")
    study.optimize(objective, n_trials=n_trials,timeout=61200) #add timeout for 60s x 60min x 17hrs
    print(study.best_trial)
    
    return study


class NCFObjective(Objective):
    """ 
    Custom objective for NCF where we can substitute
    model getter and setter hyparam methods with suggests in get_model call
    basically custom suggestions for NCF
    """
    def __init__(self, model, db):
        self.name = model.name       
        self.paradigm = 'stl'     # can change for MTL
        super().__init__(model,db)

    def __call__(self, trial):

        # Clear clutter from previous Keras session graphs.
        clear_session()
        # route model to either features or no features class automatically
        if 'Feature' in self.name:
            model = NCFFeaturescustomized(self.name, trial)
        else:
            model = NCFcustomized(self.name,trial) # get a custom model with suggest parameters for Keras for this trial
                
        score = cross_validation_score(model, self.db, num_folds=5) #FIX

        return score
    
    
class NCFFeaturescustomized(Neural_Collaborative_Filtering_Features):
    """ custom feature model with suggestions"""
    def __init__(self,name,trial):
        #this isn't used, make sure rates set to 0 though!
        hyperparams = {'batch_size': 0, 'epochs': 0, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': 0, 'mf_pretrain': '', 'mlp_pretrain': '', \
                   'num_factors': 0, 'num_neg': 0, 'out': 1, 'path': 'Data/', \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf': 0, 'verbose': 1} # needs to be true to avoid reinitialzing
        super().__init__(hyperparams, name,'feature_based',warm_start=True) 
        self.num_epochs = trial.suggest_int("epochs",low=1,high=200,log=True)
        self.batch_size = trial.suggest_int("batch size",low=24,high=72, step=12)
        self.trial = trial
        self.model =  self.get_model(trial)
        
    
    def get_model(self,trial):
        """
        Model suggestion for hyperparameter optimization on Keras implementation of NCF. Same initialization except using suggest for:
        1. MLP layer sizes and regularizers (done)
        2. Matrix factorization dimension and regularizer (done)
        3. Dropout and dropout rate (done)
        4. Optimizer (done)
        5. Learn rate (done)
        6. possibly kernel regularizer (TODO)
        """
        num_layer = trial.suggest_int("number of MLP layers",0,12)
        # Input variables
        user_input = Input(shape=(10,), dtype='float32', name='user_input')
        item_input = Input(shape=(10,), dtype='float32', name='item_input')

        # MF part
        mf_user_latent = Flatten()(user_input)
        mf_item_latent = Flatten()(item_input)
        mf_vector = Multiply()([mf_user_latent, mf_item_latent]) 

        # MLP part
        mlp_user_latent = Flatten()(user_input)
        mlp_item_latent = Flatten()(item_input)
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        for idx in range(1, num_layer):
            layer = Dense(trial.suggest_int('layer ' +  str(idx) + 'dim',low=2, high=128, step=2, log=True), kernel_regularizer=l1( trial.suggest_loguniform('mlp regualarizer at ' +  str(idx) + 'dim',0.00001,.1)), activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer, dropout not in use because there is more data in this version, currently underfitting not overfitting 
        predict_vector_dropout = Dropout(trial.suggest_float("dropout rate",low=0.001,high=.5))(predict_vector)
        prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name="prediction")(predict_vector_dropout)

        model = Model(inputs={'user_input':user_input,'item_input':item_input},
                      outputs=prediction)
#         model = tf.keras.Model(inputs=[user_input, item_input],
#                               output=prediction)

        # not to use binary crossentropy with regression
        learning_rate = trial.suggest_float('learning_rate_init',
                                             1e-5, 1e-3, log=True)
        learner = trial.suggest_categorical("learner",['adagrad','rmsprop', 'adam','sgd'])
        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        else:
            momentum = trial.suggest_float('momentum', 0.0, 1.0)
            model.compile(optimizer=SGD(lr=learning_rate, momentum=momentum), loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
        print("FEATURES",learning_rate, learner,self.batch_size, self.num_epochs, "check for suggested params")

        return model
    
class NCFcustomized(Neural_Collaborative_Filtering):
    """ custom ratings matrix model with suggestions"""
    def __init__(self,name,trial):
        # these params will be overwritten, they are just here to initialize
        hyperparams = {'batch_size': 0, 'epochs': 0, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': 0, 'mf_pretrain': '', 'mlp_pretrain': '', \
                   'num_factors': 0, 'num_neg': 0, 'out': 0, 'path': 'Data/', \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf': 0, 'verbose': 1,}
        super().__init__(hyperparams, name,'feature_based',warm_start=True)
        self.num_epochs = trial.suggest_int("epochs",low=1,high=200,log=True)
        self.batch_size = trial.suggest_int("batch size",low=24,high=72, step=12)
        self.model =  self.get_model(trial)
        self.trial = trial
    
    def get_model(self,trial):
        """
        Model suggestion for hyperparameter optimization on Keras implementation of NCF. Same initialization except using suggest for:
        1. MLP layer sizes and regularizers (done)
        2. Matrix factorization dimension and regularizer (done)
        3. Dropout and dropout rate (done)
        4. Optimizer (done)
        5. Learn rate (done)
        6. possibly kernel regularizer (TODO)
        """

        num_layer = trial.suggest_int("number of MLP layers", 0, 12)
        # Input variables
        user_input = Input(shape=(1,), dtype='float32', name='user_input')
        item_input = Input(shape=(1,), dtype='float32', name='item_input')
        mf_dim = trial.suggest_int("mf dim",0,100)
        reg_mf = trial.suggest_loguniform("reg_mf",0.00001,.1)
        first_layer = trial.suggest_int('first mlp layer dim',low=2, high=128, step=2, log=True) # try even layer dimensions form 2->128
        # but favoring the lower end of the distribution
        first_reg = trial.suggest_loguniform('first_reg',0.00001,.1)
        
        MF_Embedding_User = Embedding(input_dim=1000, output_dim=mf_dim, name='mf_embedding_user',
                                      embeddings_initializer=init_normal, embeddings_regularizer=l1(reg_mf), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=1000, output_dim=mf_dim, name='mf_embedding_item',
                                      embeddings_initializer=init_normal, embeddings_regularizer=l1(reg_mf), input_length=1)

        MLP_Embedding_User = Embedding(input_dim=1000, output_dim=int(first_layer / 2), name="mlp_embedding_user",
                                       embeddings_initializer=init_normal, embeddings_regularizer=l1(first_reg), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim=1000, output_dim=int(first_layer / 2), name='mlp_embedding_item',
                                       embeddings_initializer=init_normal, embeddings_regularizer=l1(first_reg), input_length=1)

  
        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        mf_vector = Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

        # MLP part
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        # MF part
        mf_user_latent = Flatten()(user_input)
        mf_item_latent = Flatten()(item_input)
        mf_vector = Multiply()([mf_user_latent, mf_item_latent]) 

        # MLP part
        mlp_user_latent = Flatten()(user_input)
        mlp_item_latent = Flatten()(item_input)
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        for idx in range(1, num_layer):
            if idx != 1:
                layer = Dense(trial.suggest_int('layer ' +  str(idx) + 'dim', low=2, high=128, step=2, log=True), kernel_regularizer=l1( trial.suggest_loguniform("MLP reg for layer " + str(idx), 0.00001,.1)), activation='relu', name="layer%d" % idx)
                mlp_vector = layer(mlp_vector)
            else:
                layer = Dense(first_layer, kernel_regularizer=l1(first_reg), activation='relu', name="layer%d" % idx)
                mlp_vector = layer(mlp_vector)

        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer, dropout not in use because there is more data in this version, currently underfitting not overfitting 
        predict_vector_dropout = Dropout(trial.suggest_float("suggested dropout",0.0,0.5))(predict_vector)
        prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name="prediction")(predict_vector_dropout)

        model = Model(inputs={'user_input':user_input,'item_input':item_input},
                      outputs=prediction)
#         model = tf.keras.Model(inputs=[user_input, item_input],
#                               output=prediction)

        # not to use binary crossentropy with regression
        learning_rate = trial.suggest_float('learning_rate_init',
                                             1e-5, 1e-3, log=True)
        learner = trial.suggest_categorical('learning method',['adagrad','rmsprop', 'adam','sgd'])
        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        else:
            momentum = trial.suggest_float('momentum', 0.0, 1.0)
            model.compile(optimizer=SGD(lr=learning_rate, momentum=momentum), loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
            
        print(learning_rate, learner, mf_dim,self.batch_size, self.num_epochs, "check for suggested params")

        return model




