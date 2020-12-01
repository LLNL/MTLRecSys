#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the actual GP Models and Kernel defintions

@author: soper3
@author: ladd12
"""


import gpytorch
import numpy as np
import torch
from ..base import BaseMTLEstimator
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.lazy import KroneckerProductLazyTensor, lazify
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.kernel import Kernel
from copy import deepcopy
from torch.nn import ModuleList
from gpytorch.means import Mean
import torch

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
    


class MTLGPCompositeKernelModel(gpytorch.models.ExactGP):
    """
    Composite kernel for MTL, uses CellDrugKernel above
    """
    def __init__(self, train_x, train_y, likelihood, cell_drug_split,num_tasks=1,multitask_kernel=True):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        cell_drug_split: index in which to split training data features between cell and drug
        num_tasks: number of tasks
        multitask_kernel: whether to actualyl use multitask kernel in forward. DEPRECATED. should always use multitask kern.
        """
        super(MTLGPCompositeKernelModel, self).__init__(train_x, train_y, likelihood)
        self.cell_drug_split = cell_drug_split
        n,m=train_x[0].shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.cell_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.drug_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = CellDrugKernel(self.cell_covar_module, self.drug_covar_module, cell_drug_split, n-cell_drug_split)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1)
        self.multitask_kernel = multitask_kernel
        self.train_x = train_x
        self.train_y = train_y
        self.covar_recording = True
        
    def forward(self,x,i):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        i: task mask. same length as training data, but integers ranging from 0-ntasks for each sample
        """
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)  
        # Get task-task covariance
        # Multiply the two together to get the covariance we want
        if not self.multitask_kernel:
            covar_i = torch.ones_like(covar_i.evaluate())
        if self.covar_recording:
            self.covar_x = covar_x
            self.covar_i = covar_i
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    # getters below for visualization
       
    def getTrainY(self):
        return self.train_y
    
    def getTrainX(self):
        return self.train_x[0] # second dimension is just task numbers
    
    def getCovar(self):
        return self.covar_x
    
    def getTaskCovar(self):
        return self.covar_i
    
class SparseGPModel(gpytorch.models.ExactGP):
    """
    Not in use. Model that is used for non-MTL GP that uses inducing kernel instead of sampling
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
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        n,m=train_x.shape
        inducing_points = np.random.choice(n, size=n_inducing_points, replace=False)
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=train_x[inducing_points,:],likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
      
    
class HadamardGPModel(gpytorch.models.ExactGP):
    """
    Hadamard MultiTask learning model for GP Regression
    """
    def __init__(self, train_x, train_y, likelihood, n_inducing_points,num_tasks):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        n_inducing_points: number of samples to take from data
        num_tasks = number of tasks
        """
        super(HadamardGPModel, self).__init__(train_x, train_y, likelihood)
        # COVARIANCE / MEAN module choices commented out
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.covar_module = gpytorch.kernels.MaternKernel()#lengthscale_constraint=gpytorch.constraints.Interval(.001,3)
        #self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(.4))
        self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(.4))

        #self.mean_module = gpytorch.means.ZeroMean()
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = MyMultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=num_tasks)  
        
        self.train_x = train_x
        self.train_y = train_y
        self.covar_recording = True # stores covariance so we can visualize it whenever
        self.num_tasks = num_tasks
        
    def addBias(self, mean_x, i):
        """
        inputs
        --------------------------------------------------------------------------
        mean_x: current mean vector for each task
        i: task mask. same length as training data, but integers ranging from 0-ntasks for each sample
        """
        task_mask = torch.flatten(i) # mask
        mean_x_bias = mean_x.clone()
        for idx in range(self.num_tasks):
            task_inds = task_mask == idx #get task specific indices
            mean_x_bias[task_inds] = mean_x_bias[task_inds] + self.bias[idx]
        return mean_x_bias
        
        
    def forward(self,x,i):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        i: task mask. same length as training data, but integers ranging from 0-ntasks for each sample
        """
        mean_x = self.mean_module(x)
        
#         # add bias if applicable
#         if hasattr(self,'bias'):
#             mean_x = addBias(mean_x, i)
         
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        
        #record covariance to see later
        if self.covar_recording:
            self.covar_x = covar_x
            self.covar_i = covar_i
            
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
       
    def getTrainY(self):
        return self.train_y
    
    def getTrainX(self):
        return self.train_x[0] # second dimension is just task numbers
    
    def getCovar(self):
        return self.covar_x
    
    def getTaskCovar(self):
        return self.covar_i
    
    
class GPyFullMTLModel(gpytorch.models.ExactGP):
    """
    Full multitask model with both Gpy mutlitask kernel and multitask means
    """
    def __init__(self, train_x, train_y, likelihood, n_inducing_points,num_tasks,bias_only=False):
        """
        inputs
        --------------------------------------------------------------------------
        train_x: training data for x, should be 2d array with shape (nsamples,nfeatures)
        train_y : training data vector of cell-drug response vector
        likelihood: likelihood to use in MLE for this GP model, should be marginal log likelihood
        n_inducing_points: number of samples to take from data
        num_tasks : number of tasks
        bias_only: DEPRECATED
        """
        super(GPyFullMTLModel, self).__init__(train_x, train_y, likelihood)
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.covar_module =  gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=2
        )
        self.recording = True
        
  

    def forward(self,x):
        """
        inputs
        --------------------------------------------------------------------------
        x: training data for x, should be (nsamples,nfeatures)
        """
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        if self.recording:
            self.mean_x = mean_x
            self.covar_x = covar_x
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def getMean(self):
        return self.mean_x
    
    def getCovar(self):
        return self.covar_x

    
########################################################################     
# below are some overridden methods for Multitask Kernel and Mean
# they aren't in use but could be helpful for adding customization
########################################################################   

class MyMultitaskKernel(Kernel):
    """
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks
    :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
    :param ~gpytorch.priors.Prior task_covar_prior: (default None) Prior to use for task kernel.
        See :class:`gpytorch.kernels.IndexKernel` for details.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """

    def __init__(self, data_covar_module, num_tasks, rank=1, task_covar_prior=None,bias_only=False, **kwargs):
        """
        """
        super(MyMultitaskKernel, self).__init__(**kwargs)
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks, batch_shape=self.batch_shape, rank=rank, prior=task_covar_prior
        )
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks
        self.bias_only=bias_only

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        if self.bias_only:
            covar_i = lazify(torch.ones_like(covar_i.evaluate())) # task covariance now all one so it shares covariance but still
            # as multitask mean
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLazyTensor(covar_x, covar_i)
        return res.diag() if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

class MyMultitaskMean(Mean):
    """
    Convenience :class:`gpytorch.means.Mean` implementation for defining a different mean for each task in a multitask
    model. Expects a list of `num_tasks` different mean functions, each of which is applied to the given data in
    :func:`~gpytorch.means.MultitaskMean.forward` and returned as an `n x t` matrix of means, one for each task.
    """

    def __init__(self, base_means, num_tasks):
        """
        Args:
            base_means (:obj:`list` or :obj:`gpytorch.means.Mean`): If a list, each mean is applied to the data.
                If a single mean (or a list containing a single mean), that mean is copied `t` times.
            num_tasks (int): Number of tasks. If base_means is a list, this should equal its length.
        """
        super(MyMultitaskMean, self).__init__()

        if isinstance(base_means, Mean):
            base_means = [base_means]

        if not isinstance(base_means, list) or (len(base_means) != 1 and len(base_means) != num_tasks):
            raise RuntimeError("base_means should be a list of means of length either 1 or num_tasks")

        if len(base_means) == 1:
            base_means = base_means + [deepcopy(base_means[0]) for i in range(num_tasks - 1)]

        self.base_means = ModuleList(base_means)
        self.num_tasks = num_tasks

    def forward(self, input, task_mask):
        """
        Evaluate each mean in self.base_means on the input data, and return as an `n x t` matrix of means.
        """
        multiDimMean = torch.cat([sub_mean(input).unsqueeze(-1) for sub_mean in self.base_means], dim=-1)
        print(multiDimMean.shape)
        task_mask = torch.flatten(task_mask)
        #print(task_mask)

        #print(final_mean.shape, "final mean")

        for i in range(self.num_tasks):
            task_inds = task_mask == i
            print(task_inds,i, "HI")
            print(multiDimMean[task_inds,1])
            multiDimMean[task_inds,0] = multiDimMean[task_inds,i]
        print(multiDimMean[0,0], multiDimMean[199,0])
        print(multiDimMean.shape)
        
        return multiDimMean[:,0]
    
    def __call__(self, x, t):
        # Add a last dimension
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x,t)

        return res 

    
    
    