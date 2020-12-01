#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:37:35 2020

Exact GP Regression model. 
Uses GPytorch. 
Dependency is GPModels, a module full of classes defining models and kernels. 

@author: soper3
"""
import torch
import gpytorch
import numpy as np
from ..base import BaseOwnSTLEstimator
from .GPModels import ExactGPModel, SparseGPModel


class ExactGPRegression(BaseOwnSTLEstimator):
    """
    Exact GP, Gaussian Process evaluated at all training points
    
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
            
    """

    def __init__(self, name='ExactGP', num_iters=50, learning_rate=1e-1,
                 noise_covar=1.0, length_scale=100.0, output_scale=1.0):
        super().__init__(name, 'feature_based', 'array')
        
        """
        Flags
        ------------------------------------------------------------------
        num_iters, learning_rate: These are training parameters that decide how long model should train, at what rate.
        noise_covar, lenght_scale, outputscale: direct model "hyper" parameters that get passed into kernel. 
        """

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.training_iter = num_iters
        self.learning_rate = learning_rate
        self.noise_covar = noise_covar
        self.length_scale = length_scale
        self.output_scale = output_scale

    def _find_initial_conditions(self, x, y, n_restarts, n_iters, n_inducing_points):
        """
        Code to find initial conditions that work well for GP model so that hyperparameters can be intialized 
        from nice values.
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in 2d array form (nsamples,nfeatures)
        y: training data in list form, (nsameples) for each response
        n_restarts: number of trials to test different intial conditions
        n_iters: number of iterations to train for each trial
        n_inducing_points: number of points to sample for runs
        
        outputs
        ------------------------------------------------------------------------------------
        min_loss: best (lowest) loss achieved
        min_hypers: dictionary with values for best parameters
        """
        self.n_restarts = n_restarts
        self.sparse_model = SparseGPModel(x, y, self.likelihood, n_inducing_points)
        # initialize hyperparameters
        min_loss = np.infty
        for k in range(n_restarts):
            noise = 2*np.random.random()
            lengthscale = 100*np.random.random()
            outputscale = 10*np.random.random()
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
            optimizer = torch.optim.Adam([{'params': self.sparse_model.parameters()}, ],
                                         lr=self.learning_rate)
     
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.sparse_model)
    
            for i in range(n_iters):
                # zero gradients from previous step
                optimizer.zero_grad()
                # output from model
                output = self.sparse_model(x)
                # calc loss and backprop gradients
                loss = -mll(output, y)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (i+1, n_iters, loss.item() ))
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

    def _fit(self, x, **params):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in 2d array form (nsamples,nfeatures)
        y: training data in list form, (nsameples) for each response
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        """

        y = params['y']  # output variable

        cat_point = params['cat_point']  # feature concatenation point
        # cell_feat = x[:, :cat_point]
        # drug_feat = x[:, cat_point:]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        
        self.model = ExactGPModel(x, y, self.likelihood)

        hypers = {
                'likelihood.noise_covar.noise': self.noise_covar, #torch.tensor(self.noise_covar),
                'covar_module.base_kernel.lengthscale': self.length_scale, #torch.tensor(self.length_scale),
                'covar_module.outputscale': self.output_scale, #torch.tensor(self.output_scale),
                }

        self.model.initialize(**hypers)

        # put in train mode
        self.model.train()
        self.likelihood.train()

        # set optimizer
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}, ],
                                     lr=self.learning_rate)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.training_iter):
            # zero gradients from previous step
            optimizer.zero_grad()
            # output from model
            output = self.model(x)
            # calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
        
            print('Iter %d/%d - Loss: %.3f' % (i+1, self.training_iter, loss.item()))
        
            optimizer.step()

    def _predict(self, x, **kwargs):
        """ 
        Predict using trained model.
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in 2d array form (nsamples,nfeatures)
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        """
        return_std = False
        if 'return_std' in kwargs.keys():
            return_std = kwargs['return_std']

        # put in eval mode
        self.model.eval()
        self.likelihood.eval()

        # make predictions using likelihood
        with torch.no_grad():
            test_x = torch.tensor(x, dtype=torch.float32)
            observed_pred = self.likelihood(self.model(test_x))
            pred_y = observed_pred.mean

        if return_std:
            lower, upper = observed_pred.confidence_region()
            return pred_y, lower, upper
        else:
            return pred_y
