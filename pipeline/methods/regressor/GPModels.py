#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:46:45 2020

This module contains several classes to be used for constructing models using GPytorch.
Any custom kernel or custom GPytorch regression model can be defined as a class here. 

To Do: Allow user to set the kernels, not just RBF. The mean can stay constant for now. 
To Do: Consider making one model that can be either exact or sparse, and arbitrary kernel. This would simplify all this down to one class. 



@author: soper3
"""

import gpytorch
import numpy as np

class CellDrugKernel(gpytorch.kernels.Kernel):
    """ 
    kernel to combine cell drug kernels respectively to make composite kernel with product or adding
    """
    
    def __init__(self, kernel_cell, kernel_drug, cell_drug_split, operation='add'):
        """
        inputs
        --------------------------------------------------------------------------
        kernel_cell: Gpy kernel object corresponding to rbf kernel on cell features
        kernel_drug: Gpy kernel object corresponding to rbf kernel on drug features
        cell drug split: index in which the two sets of features are split in training data
        operation: string 'add to add or multiply kernels
        """
        super(CellDrugKernel, self).__init__()
        self.kernel_cell = kernel_cell
        self.kernel_drug = kernel_drug
        self.cell_drug_split = cell_drug_split
        self.operation = operation
        
    def forward(self, x1, x2, last_dim_is_batch,diag=True):
        """
        inputs
        --------------------------------------------------------------------------
        x1, x2: feature data arrays
        """
        cell_forward = self.kernel_cell.forward(x1[:,:self.cell_drug_split], x2[:,:self.cell_drug_split])
        drug_forward = self.kernel_drug.forward(x1[:,self.cell_drug_split:], x2[:,self.cell_drug_split:])
        
        if self.operation=='product':
            out = cell_forward * drug_forward
        else:
            out = cell_forward + drug_forward
            
        return out
    
    

class ExactGPModel(gpytorch.models.ExactGP):
    """
    Nonsamping version of GP, uses every training point in kernel
    """
    def __init__(self, train_x, train_y, likelihood):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPModel(gpytorch.models.ExactGP):
    """
    Sampling GP model, only defines process at n inducing points
    """
    def __init__(self, train_x, train_y, likelihood, n_inducing_points):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        n_inducing_points: number of samples to take from data
        """
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.n_inducing_points = n_inducing_points
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(0.1))) #lengthscale_constraint=gpytorch.constraints.GreaterThan(0.1)
        n,m=train_x.shape
        inducing_points = np.random.choice(n, size=n_inducing_points, replace=False)
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=train_x[inducing_points,:],likelihood=likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.recording = True

    def forward(self, x):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if self.recording:
            self.covar_x = covar_x
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    # getters
    def getTrainY(self):
        return self.train_y
    
    def getTrainX(self):
        return self.train_x # second dimension is just task numbers
    
    def getCovar(self):
        return self.covar_x

    

class ExactGPCompositeKernelModel(gpytorch.models.ExactGP):
    """
    Nonsamping version of GP, uses every training point and splits features into two kernels, one for drug and one for cell line
    """
    def __init__(self, train_x, train_y, likelihood, cell_drug_split):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        cell_drug_split: index at which we should split our features to go into different kernels
        """
        super(ExactGPCompositeKernelModel, self).__init__(train_x, train_y, likelihood)
        self.cell_drug_split = cell_drug_split
        n,m=train_x.shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.cell_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.drug_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = CellDrugKernel(self.cell_covar_module, self.drug_covar_module, cell_drug_split)#, n-cell_drug_split)
        
    def forward(self, x):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SparseGPCompositeKernelModel(gpytorch.models.ExactGP):
    """
    Sampling GP model, only defines process at n inducing points. Uses two seperate kernels for drug and cell features    
    """
    def __init__(self, train_x, train_y, likelihood, cell_drug_split, n_inducing_points):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        cell_drug_split: index at which we should split our features to go into different kernels
        n_inducing: points where GP is defined (samples from training set)
        """
        super(SparseGPCompositeKernelModel, self).__init__(train_x, train_y, likelihood)
        self.cell_drug_split = cell_drug_split
        self.n_inducing_points = n_inducing_points
        n,m=train_x.shape
        inducing_points = np.random.choice(n, size=n_inducing_points, replace=False)
        self.mean_module = gpytorch.means.ConstantMean()
        self.cell_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.drug_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.base_covar_module = CellDrugKernel(self.cell_covar_module, self.drug_covar_module, cell_drug_split, n-cell_drug_split)
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=train_x[inducing_points,:],likelihood=likelihood)

    def forward(self, x):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

