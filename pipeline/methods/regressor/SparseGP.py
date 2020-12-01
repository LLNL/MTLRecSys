#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:22:38 2020

Sparse GP Model. Uses inducing points for low-rank approximation. 
Uses GPytorch. 
Dependency is GPModels, a module full of classes defining models and kernels. 

@author: soper3
"""


import torch
import gpytorch
import numpy as np
from ..base import BaseOwnSTLEstimator
from .GPModels import SparseGPModel
from .GPModels import SparseGPCompositeKernelModel



class SparseGPRegression(BaseOwnSTLEstimator):
    """
    Sparse GP, Gaussian Process evaluated only at N inducing points sampled from training points
    
    Args:
        :attr:`name` (optional, string):
            model name
        :attr:`num_iters` (int):
            number of iterations for Gaussian Process
        :attr:`learning_rate` (int):
            learning rate for conjugate gradient. recommended around .1 or .01
        :attr:`noise_covar` (float):
            hyperparamter, noise assumed in the data
       :attr:`lengthscale` (float):
            hyperparameter, magnitude relative to assumed correlation in data
       :attr:`output_scale` (optional, float):
            scaling parameter
       :attr:`n_inducing_points` (optional, int):
            number of training points to sample from for Gaussian Process
            
            
    """

    def __init__(self, name='SparseGP', num_iters=50, learning_rate=1e-1,
                 noise_covar=1.0, length_scale=100.0, output_scale=1.0, n_inducing_points=500,use_initial=True):
        super().__init__(name, 'feature_based', 'array')
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(.01))
        self.training_iter = num_iters
        self.learning_rate = learning_rate # HP
        self.noise_covar = noise_covar
        self.length_scale = length_scale
        self.output_scale = output_scale 
        self.n_inducing_points = n_inducing_points
        self.use_initial = use_initial

    def _find_initial_conditions(self, x, y, n_restarts, n_iters, n_inducing_points):
        self.n_restarts = n_restarts
        self.sparse_model = SparseGPModel(x, y, self.likelihood, n_inducing_points)
        # initialize hyperparameters
        min_loss = np.infty
        for k in range(n_restarts):
            noise = 2*np.random.random()
            lengthscale = 100*np.random.random()
            outputscale = 100*np.random.random()
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(noise),
                'covar_module.base_kernel.base_kernel.lengthscale': torch.tensor(lengthscale),
                'covar_module.base_kernel.outputscale': torch.tensor(outputscale),
            }
            self.sparse_model.initialize(**hypers)
            # put in train mode
            self.sparse_model.train()
            self.likelihood.train()

            # set optimizer
            optimizer = torch.optim.Adam([{'params': self.sparse_model.parameters()},], lr=0.1)
     
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.sparse_model)
            
            for i in range(n_iters):
                # zero gradients from previous step
                optimizer.zero_grad()
                # output from model
                output = self.sparse_model(x)
                # calc loss and backprop gradients
                loss = -mll(output, y)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f lengthscale: %.3f outputscale: %.3f noise: %.3f' % (
                    i+1, n_iters, 
                    loss.item(),
                    self.sparse_model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                    self.sparse_model.covar_module.base_kernel.outputscale.item(),      
                    self.sparse_model.likelihood.noise.item()      
                ))
                optimizer.step()
                
            output = self.sparse_model(x)
            loss = -mll(output, y)
            if loss < min_loss:
                min_loss = loss
                min_hypers = {
                'likelihood.noise_covar.noise': self.sparse_model.likelihood.noise.item(),
                'covar_module.base_kernel.base_kernel.lengthscale': self.sparse_model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                'covar_module.base_kernel.outputscale': self.sparse_model.covar_module.base_kernel.outputscale.item(),
                }
                
        return min_loss, min_hypers    

    def _fit(self, x, **kwargs):
#         Fit hyperparameters using Maximum Likelihood Estimation

        y = kwargs['y']  # output variable
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        perm = torch.randperm(x.size(0))
        idx = perm[:10000] #max size for inducing point kernel before memory error
        x = x[idx,:]
        y = y[idx]


        self.model = SparseGPModel(x, y, self.likelihood, self.n_inducing_points)
        
        """
        TO DO:
        These next lines were how I was initializing the initial conditions of the hyperparameters. 
        I was passing **hypers as a kwarg to _fit and then passing it directly to model.initialize.
        That does not work in the pipeline though.
        I'm not sure how we should do it with the new pipeline. 
        """
        # initialize hyperparameters
#        if len(hypers)>0:
#            self.model.initialize(**hypers)
        hypers = {
                'likelihood.noise_covar.noise': self.noise_covar, #torch.tensor(self.noise_covar),
                'covar_module.base_kernel.base_kernel.lengthscale': self.length_scale, #torch.tensor(self.length_scale),
                'covar_module.base_kernel.outputscale': self.output_scale #torch.tensor(self.output_scale),
                }
    
        if self.use_initial:
            self.model.initialize(**hypers)
        
        # put in train mode
        self.model.train()
        self.likelihood.train()

        # set optimizer
        optimizer = torch.optim.Adam([{'params': self.model.parameters()},], lr=self.learning_rate)
 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        with gpytorch.settings.fast_pred_var():

            for i in range(self.training_iter):
                # zero gradients from previous step
                optimizer.zero_grad()
                # output from model
                output = self.model(x)
                # calc loss and backprop gradients
                loss = -mll(output, y)
                loss.backward()

                print('Iter %d/%d - Loss: %.3f lengthscale: %.3f outputscale: %.3f noise: %.3f' % (
                i+1, self.training_iter, 
                loss.item(),
                self.model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                self.model.covar_module.base_kernel.outputscale.item(),      
                self.model.likelihood.noise.item()      
                ))

                optimizer.step()

    def _predict(self, x, **kwargs):

        return_std = False
#         if 'return_std' in kwargs.keys():
#             return_std = kwargs['return_std']
        
        # put in eval mode
        self.model.eval()
        self.likelihood.eval()
        self.model.recording = False

        # make predictions using likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(x, dtype=torch.float32)
            observed_pred = self.likelihood(self.model(test_x))
            pred_y = observed_pred.mean

        if return_std:
            lower, upper = observed_pred.confidence_region()
            return pred_y, lower, upper
        else:
            return pred_y
        
    # for use with optuna
    def get_hyper_params(self):
        hparams = {'noise_covar': {'type': 'loguniform', 'values': [.1,2]},
                  'learning_rate': {'type': 'uniform', 'values':  [.0001,.1]},
                  'length_scale': {'type': 'uniform', 'values': [1,100]}, # don't use 60 for synthetic data,
                  'output_scale': {'type': 'loguniform', 'values': [.1,2]},
                  'num_iters': {'type': 'integer', 'values': [10, 200]}
                  }
        return hparams
    
    # for use with optuna
    def set_hyper_params(self, **kwargs):
        self.noise_covar = kwargs['noise_covar']
        self.learning_rate = kwargs['learning_rate']
        self.length_scale = kwargs['length_scale']
        self.output_scale = kwargs['output_scale']
        self.training_iter = kwargs['num_iters']        
        
    


