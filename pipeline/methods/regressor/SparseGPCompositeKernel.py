#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:37:52 2020

Sparse GP Model. Uses inducing points for low-rank approximation. 
Uses GPytorch. 
Dependency is GPModels, a module full of classes defining models and kernels. 

@author: soper3
"""

import torch
import gpytorch
import numpy as np
from ..base import BaseOwnSTLEstimator
from .GPModels import SparseGPCompositeKernelModel


class SparseGPCompositeKernelRegression(BaseOwnSTLEstimator):
    """
    Sparse GP where seperate Kernels are evaluated for drugs and cells and then muliplied or added to make a shared kernel, Gaussian Process evaluated at only n_inducing training points
    
    Args:
        :attr:`name` (optional, string):
            model name
        :attr:`num_iters` (int):
            number of iterations for Gaussian Process
        :attr:`learning_rate` (int):
            learning rate for conjugate gradient. recommended around .1 or .01
        :attr:`noise_covar` (float):
            hyperparamter, noise assumed in the data
       :attr:`length_scale_cell` (float):
            hyperparameter, magnitude relative to assumed correlation in **cell** data
       :attr:`length_scale_drug` (float):
            hyperparameter, magnitude relative to assumed correlation in **drug** data
       :attr:`output_scale_drug` (optional, float):
            scaling parameter for **drug** data
       :attr:`output_scale_cell` (optional, float):
            scaling parameter for **cell** data
       :attr:`n_inducing_points` (optional, int):
            number of training points to sample from for Gaussian Process
            
            
    """

    def __init__(self, name='SparseGPCompositeKernel', num_iters=50, learning_rate=1e-1, noise_covar=1.0, 
                 length_scale_cell=100.0, output_scale_cell=1.0, 
                 length_scale_drug=100.0, output_scale_drug=1.0, n_inducing_points=500):
        
        """
        Flags
        ------------------------------------------------------------------
        num_iters, learning_rate: These are training parameters that decide how long model should train, at what rate.
        noise_covar, lenght_scale, outputscale: direct model "hyper" parameters that get passed into kernel. the ones 
        ending in "cell" and "drug" go into those respective kernels for composite model. Noise covar same for both
        n_inducing points: number of points to sample from training, anything over 2000 samples will be a bit slow
        """
        
        super().__init__(name,type_met='feature_based', output_shape='array')

        self.training_iter = num_iters
        self.learning_rate = learning_rate
        self.noise_covar   = noise_covar

        self.length_scale_cell  = length_scale_cell
        self.output_scale_cell  = output_scale_cell
        self.length_scale_drug  = length_scale_drug 
        self.output_scale_drug  = output_scale_drug 

        self.n_inducing_points = n_inducing_points

        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(.01))

    def _find_initial_conditions(self, x, y, cell_drug_split, n_inducing_points, n_restarts, n_iters):
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
        self.sparse_model = SparseGPCompositeKernelModel(x, y, self.likelihood, cell_drug_split, n_inducing_points)
        # initialize hyperparameters
        min_loss = np.infty
        for k in range(n_restarts):
            print('Restart %d/%d' % (k+1, n_restarts ))

            noise = 2*np.random.random()
            cell_lengthscale = 100*np.random.random()
            cell_outputscale = 100*np.random.random()            
            drug_lengthscale = 100*np.random.random()
            drug_outputscale = 100*np.random.random()
            hypers = {
                     'likelihood.noise_covar.noise': torch.tensor(noise),
                     'cell_covar_module.base_kernel.lengthscale': torch.tensor(cell_lengthscale),
                     'cell_covar_module.outputscale': torch.tensor(cell_outputscale),
                     'drug_covar_module.base_kernel.lengthscale': torch.tensor(drug_lengthscale),
                     'drug_covar_module.outputscale': torch.tensor(drug_outputscale)
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
                print('Iter %d/%d - Loss: %.3f' % (i+1, n_iters, loss.item() ))
                optimizer.step()
                
            output = self.sparse_model(x)
            loss = -mll(output, y)
            if loss < min_loss:
                min_loss = loss
                min_hypers = {
                              'likelihood.noise_covar.noise': self.sparse_model.likelihood.noise.item(),
                              'cell_covar_module.base_kernel.lengthscale': self.sparse_model.cell_covar_module.base_kernel.lengthscale.item(),
                              'cell_covar_module.outputscale': self.sparse_model.cell_covar_module.outputscale.item(),
                              'drug_covar_module.base_kernel.lengthscale': self.sparse_model.drug_covar_module.base_kernel.lengthscale.item(),
                              'drug_covar_module.outputscale': self.sparse_model.drug_covar_module.outputscale.item()
                              }
             
        return min_loss, min_hypers    

    def _fit(self, x, **kwargs):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in 2d array form (nsamples,nfeatures)
        y(kwarg): training data in list form, (nsameples) for each response
        cell_drug_split(kwarg): splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        
        """
        
        y = kwargs['y']  # output variable
#         x = torch.tensor(x, dtype=torch.float32)
#         y = torch.tensor(y, dtype=torch.float32)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        perm = torch.randperm(x.size(0))
        idx = perm[:10000] # max size for induc. kernel before error
        x = x[idx,:]
        y = y[idx]
        print(y.shape)
     

        """ 
        TO DO:  include cell_drug_split to params. 
        This is the index that splits cell features from drug features. 
        """
        cell_drug_split = kwargs['cat_point']

        self.model = SparseGPCompositeKernelModel(x, y, self.likelihood, cell_drug_split, self.n_inducing_points)
       
        hypers = {
                      'likelihood.noise_covar.noise': self.noise_covar,
                      'cell_covar_module.base_kernel.lengthscale': self.length_scale_cell,
                      'cell_covar_module.outputscale': self.output_scale_cell,
                      'drug_covar_module.base_kernel.lengthscale': self.length_scale_drug,
                      'drug_covar_module.outputscale': self.output_scale_drug
                 }
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

                print('Iter %d/%d - Loss: %.3f' % (i+1, self.training_iter, loss.item() ), "covariance noise:", self.model.likelihood.noise.item()  , \
                     "lengthscale :",  self.model.drug_covar_module.base_kernel.lengthscale.item())

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
        with torch.no_grad(),  gpytorch.settings.fast_pred_var():
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
                  'length_scale_drug': {'type': 'uniform', 'values': [1,100]}, # don't use 60 for synthetic data,
                  'output_scale_drug': {'type': 'loguniform', 'values': [.1,10]},
                  'length_scale_cell': {'type': 'uniform', 'values': [1,100]}, # don't use 60 for synthetic data,
                  'output_scale_cell': {'type': 'loguniform', 'values': [.1,10]},
                  'num_iters': {'type': 'integer', 'values': [10, 200]}
                  }
        return hparams
    
    # for use with optuna
    def set_hyper_params(self, **kwargs):
        self.noise_covar = kwargs['noise_covar']
        self.learning_rate = kwargs['learning_rate']
        self.length_scale_drug = kwargs['length_scale_drug']
        self.output_scale_drug = kwargs['output_scale_drug']
        self.length_scale_cell = kwargs['length_scale_cell']
        self.output_scale_cell = kwargs['output_scale_cell']
        self.training_iter = kwargs['num_iters']  

                
    


