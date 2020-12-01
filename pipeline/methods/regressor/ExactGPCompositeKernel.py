#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:37:51 2020

@author: soper3
"""

import torch
import gpytorch
import numpy as np
from ..base import BaseOwnSTLEstimator
from .GPModels import ExactGPCompositeKernelModel, SparseGPCompositeKernelModel


class ExactGPCompositeKernelRegression(BaseOwnSTLEstimator):
    """
    Exact GP where seperate Kernels are evaluated for drugs and cells and then muliplied or added to make a shared kernel, Gaussian Process evaluated at all training points
    
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
            
            
    """

    def __init__(self, name='ExactGPCompositeKernel', num_iters=50, learning_rate=1e-1, noise_covar=1.0, 
                 length_scale_cell=100.0, output_scale_cell=1.0, 
                 length_scale_drug=100.0, output_scale_drug=1.0):
        
        """
        Flags
        ------------------------------------------------------------------
        num_iters, learning_rate: These are training parameters that decide how long model should train, at what rate.
        noise_covar, length_scale, outputscale: direct model "hyper" parameters that get passed into kernel respectively for 
        drug and cell kernels. 
        """
        super().__init__(name,'feature_based', 'array')
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.training_iter = num_iters
        self.learning_rate = learning_rate
        self.noise_covar   = noise_covar

        self.length_scale_cell  = length_scale_cell
        self.output_scale_cell  = output_scale_cell
        self.length_scale_drug  = length_scale_drug 
        self.output_scale_drug  = output_scale_drug 

    def _find_initial_conditions(self, x, y, cell_drug_split, n_inducing_points, n_restarts, n_iters):
        """
        Code to find initial conditions that work well for GP model so that hyperparameters can be intialized 
        from nice values.
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in 2d array form (nsamples,nfeatures)
        y: training data in list of nsamples responses
        n_restarts: number of trials to test different intial conditinos
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

    def _fit(self, x, **params):
        """ Fit hyperparameters using Maximum Likelihood Estimation 
        inputs
        ------------------------------------------------------------------------------------
        x: training data in 2d array form (nsamples,nfeatures)
        y: training data in list form, (nsameples) for each response
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        
        """
        
        y = params['y']  # output variable
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        """ 
        TO DO:  include cell_drug_split to params. This is the index that splits cell features from drug features. 

        """
        cell_drug_split = params['cat_point']

        self.model = ExactGPCompositeKernelModel(x, y, self.likelihood, cell_drug_split)

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

        for i in range(self.training_iter):
            # zero gradients from previous step
            optimizer.zero_grad()
            # output from model
            output = self.model(x)
            # calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
        
            print('Iter %d/%d - Loss: %.3f' % (i+1, self.training_iter, loss.item() ))
        
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
        
        
    


