#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes that find initial conditons, initialization, 
training and testing for multitask learning Gaussian Processes. 

Contains: hadamard GP, multitask GP, multioutputt GP from https://docs.gpytorch.ai/en/v1.1.1/index.html

Imports models from GPModels.py

@author: soper3
@author: ladd12
"""

import gpytorch
import numpy as np
import torch
from ..base import BaseMTLEstimator
from gpytorch.mlls import SumMarginalLogLikelihood
# import models from GP Models . py
from methods.mtl.GPModels import CellDrugKernel, MTLGPCompositeKernelModel, SparseGPModel, HadamardGPModel
from methods.mtl.GPModels import GPyFullMTLModel, MyMultitaskKernel, MyMultitaskMean    
    
class HadamardMTL(BaseMTLEstimator):
    """
    pipeline suited implementation of
    https://docs.gpytorch.ai/en/v1.1.1/examples/03_Multitask_Exact_GPs/Hadamard_Multitask_GP_Regression.html 
    
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
       :attr:`composite` (bool):
           whether to use composite kernel or not
       :attr:`validate` (bool):
           whether to produce validation curve data as well during training
       :attr:`bias` (bool):
           whether to add bias term for each dataset
       :attr:`stabilize` (bool):
           whether to stabilize loss at the end
       :attr:`use_initial` (bool):
           whether to even use initial parameters
          
          
    """
    
    def __init__(self, name='HadamardMTL', num_iters=50, learning_rate=1e-1,
                 noise_covar=1.0, length_scale=100.0, output_scale=1.0, n_inducing_points=500, composite=False, \
                 validate=False, bias=False, stabilize=False, use_initial=True):
        """
        Flags
        ------------------------------------------------------------------
        num_iters, learning_rate: These are training parameters that decide how long model should train, at what rate.
        noise_covar, lenght_scale, outputscale: direct model "hyper" parameters that get passed into kernel. 
        n_inducing points: number of points to sample from training, anything over 2000 samples will be a bit slow
        comoposite: boolean, whether the model should implement two seperate kernels for drug features and cell features then combine them
        to make one composite kernel
        validate: boolean, whether model should be validated with predictions, since it is a probabilistic model. I used training data, 
        could also use testing data instead to check for overfitting.
        bias: boolean, whether model should add a bias parameter for each dataset
        stabilize: boolean, whether we should sharply drop learning rate at low loss so model will converge, only works for loss < .8
        use_intial: boolean, whether we should actually initialize any hyperparameters. 
        """
        
        super().__init__(name, type_met='feature_based')
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(.01))
                                                                  #noise_constraint=gpytorch.constraints.Interval(.001,1.0)
        self.training_iter = num_iters
        self.learning_rate = learning_rate # HP
        self.noise_covar = noise_covar
        self.length_scale = length_scale 
        self.output_scale = output_scale 
        self.n_inducing_points = n_inducing_points
        self.composite= composite
        self.validate = validate
        self.validations = []
        self.stabilize = stabilize
        self.bias = bias
        self.use_initial = use_initial
        

    def fit(self, x, y, cell_drug_split=10, **params):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in dictionary form, each key corresponding to a task/dataset
        y: training data in dictionary form, each key corresponding to a task/dataset
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        """
        assert isinstance(x, dict)
        
        full_train_x = torch.empty(0)
        full_train_idxs = torch.empty(0, dtype=torch.long)
        full_train_y = torch.empty(0)
        task_num = 0
        for dataset in list(x.keys()):
            curr_x = torch.tensor(x[dataset].astype(np.float32))
            print(curr_x.shape, "sampling from: ", dataset)
            curr_y = torch.tensor(y[dataset].astype(np.float32))
            perm = torch.randperm(curr_x.size(0))
            idx = perm[:self.n_inducing_points] # sample from input data
            x_samples = curr_x[idx]
            y_samples = curr_y[idx]
            full_train_x = torch.cat([full_train_x, x_samples])
            print(full_train_x.shape, "full train shape evolve")
            train_idxs = torch.full_like(x_samples[:,0], dtype=torch.long, fill_value=task_num) #torch.from_numpy(np.repeat(task_num,len(x_samples)))
            full_train_idxs = torch.cat([full_train_idxs, train_idxs])
            full_train_y = torch.cat([full_train_y, y_samples])
            task_num += 1
        
        # define model from GP Models. py
        if self.composite:
            self.model = MTLGPCompositeKernelModel((full_train_x, full_train_idxs), full_train_y, self.likelihood, self.n_inducing_points, num_tasks=task_num, multitask_kernel=self.multitask_kernel)
        else:
            self.model = HadamardGPModel((full_train_x, full_train_idxs), full_train_y, self.likelihood, self.n_inducing_points, num_tasks=task_num)
        
        # define and intialize hyperparameters
        hypers = {
               'likelihood.noise_covar.noise': self.noise_covar, #torch.tensor(self.noise_covar),
               'covar_module.lengthscale': self.length_scale, #torch.tensor(self.length_scale),
                }
        
        if self.use_initial:
            self.model.initialize(**hypers)
        
        # decide to use bias or not
        if self.bias:
            self.model.register_parameter("bias", torch.nn.parameter.Parameter(torch.zeros(task_num), requires_grad=True))

        # put in train mode
        self.model.train()
        self.likelihood.train()

        # set optimizer
        optimizer = torch.optim.Adam([{'params': self.model.parameters()},], lr=self.learning_rate)
        #optimizer = torch.optim.SGD([{'params': self.model.parameters()},], lr=self.learning_rate, momentum=.9)
        #optimizer = torch.optim.RMSprop([{'params': self.model.parameters()},], lr=self.learning_rate)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # set some custom conjuagte gradient settings for increased stability
        gpytorch.settings.cg_tolerance(4)._set_value(.5)
        gpytorch.settings.max_cg_iterations(0)._set_value(2000)
        gpytorch.settings.eval_cg_tolerance(4)._set_value(.001)
        
        # context manager for training, fast computation, uses LOVE approximation refrenced in GPytorch docs
        #with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_preconditioner_size(5):
        with gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True):
            for i in range(self.training_iter):
                
                # zero gradients from previous step
                optimizer.zero_grad()
                
                # output from model
                output =  self.model(full_train_x, full_train_idxs)
                
                # calc loss and backprop gradients
                loss = -mll(output,full_train_y)
                loss.backward()
                
                #handle printouts
                for param in list(self.model.named_parameters()):
                    #print(param[0], param[1])
                    if param[0] == 'likelihood.noise_covar.raw_noise':
                        noise_covar = param[1].data.numpy()[0]
                    if param[0] == 'covar_module.lengthscale':
                        lenscale = param[1].data.numpy()[0][0]
                    if param[0] == 'bias':
                        print(param[1], "bias")

                print('Iter %d/%d - Loss: %.3f' % (i+1, self.training_iter, loss.item() ), "covariance noise:", self.model.likelihood.noise.item()  , \
                     "lengthscale :",  self.model.covar_module.lengthscale.item())
                
                optimizer.step()
                
                
                # record validations
                if i % 10 == 0:
                    last_ten_loss = []
                    frozen_models = []
                    if self.validate and i % 10 == 0:
                        with torch.no_grad():
                            self.model.eval()
                            self.likelihood.eval()

                            pred = self.likelihood(self.model(full_train_x, full_train_idxs)) # replace with full_test_x
                            pred = pred.mean
                            rmse = np.sqrt(np.sum(((pred.numpy() - full_train_y.numpy()) ** 2) / len(full_train_y.numpy())))
                            self.validations.append(rmse)

                # shrinks learning rate with really low loss if stabilize is true                 
                if loss.item() < .85 and self.stabilize:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = .0001
                        print(param_group['lr'])
                        milestone_hit = True
                    

    def predict(self, x, **kwargs):
        """ 
        Predict using trained model.
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in dictionary form, each key corresponding to a task/dataset
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        """
        assert isinstance(x, dict)
        
        # put in eval mode
        self.model.eval()
        # we don't want to store prediction covariances with getter/setter
        self.model.covar_recording = False
        self.likelihood.eval()
        
        task_num = 0
        res = {} # return dictionary where each key corresponds to the predictions for each dataset
        
        #settings for nice gradients
        gpytorch.settings.cg_tolerance(4)._set_value(1)
        gpytorch.settings.max_cg_iterations(0)._set_value(1000)
        gpytorch.settings.eval_cg_tolerance(4)._set_value(.01)
       
        #gpytorch.settings.fast_computations(), gpytorch.settings.max_preconditioner_size(20)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(),  gpytorch.settings.max_preconditioner_size(10):
            for dataset in list(x.keys()):
                curr_x = torch.tensor(x[dataset].astype(np.float32))
                train_idxs = torch.full_like(curr_x[:,0], dtype=torch.long, fill_value=task_num) 
                observed_pred = self.likelihood(self.model(curr_x, train_idxs))
                res[dataset] = observed_pred.mean # predictions are a distribution of predicted function outputs, we'll take that mean
                task_num += 1
        return res
    
    def getValidations(self):
        return self.validations
    
     # for use with optuna
    def get_hyper_params(self):
        hparams = {'noise_covar': {'type': 'loguniform', 'values': [.1,2]},
                  'learning_rate': {'type': 'uniform', 'values':  [.0001,.1]},
                  'length_scale': {'type': 'uniform', 'values': [1,60]}, # don't use 60 for synthetic data
                  'use_bias': {'type': 'uniform', 'values':  [0.1,.99]},
                  'use_stabilizer': {'type': 'uniform', 'values':   [0.1,.99]},
                  'num_iters': {'type': 'integer', 'values': [10, 200]}
                  }
        return hparams
    
    # for use with optuna
    def set_hyper_params(self, **kwargs):
        self.noise_covar = kwargs['noise_covar']
        self.learning_rate = kwargs['learning_rate']
        self.length_scale = kwargs['length_scale']
        self.training_iter = kwargs['num_iters']

        if kwargs['use_bias'] < .10:
            self.bias = True
        if kwargs['use_stabilizer'] < .34:
            self.stabilizer = True
            

    def _find_initial_conditions(self, x, y, n_restarts, n_iters, n_inducing_points):
        """
        Code to find initial conditions that work well for GP model so that hyperparameters can be intialized 
        from nice values.
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in dictionary form, each key corresponding to a task/dataset
        y: training data in dictionary form, each key corresponding to a task/dataset
        n_restarts: number of trials to test different intial conditinos
        n_iters: number of iterations to train for each trial
        n_inducing_points: number of points to sample for runs
        
        outputs
        ------------------------------------------------------------------------------------
        min_loss: best (lowest) loss achieved
        min_hypers: dictionary with values for best parameters
        """
        assert isinstance(x, dict)
        assert isinstance(y, dict)

        full_train_x = torch.empty(0)
        full_train_idxs = torch.empty(0, dtype=torch.long)
        full_train_y = torch.empty(0)
        task_num = 0
        for dataset in list(x.keys()):
            curr_x = torch.tensor(x[dataset].astype(np.float32))
            print(curr_x.shape)
            curr_y = torch.tensor(y[dataset].astype(np.float32))
            perm = torch.randperm(curr_x.size(0))
            idx = perm[:self.n_inducing_points]
            x_samples = curr_x[idx]
            y_samples = curr_y[idx]
            full_train_x = torch.cat([full_train_x, x_samples])
            train_idxs = torch.full_like(x_samples[:,0], dtype=torch.long, fill_value=task_num) #torch.from_numpy(np.repeat(task_num,len(x_samples)))
            full_train_idxs = torch.cat([full_train_idxs, train_idxs])
            full_train_y = torch.cat([full_train_y, y_samples])           
            task_num += 1
        
        self.n_restarts = n_restarts
        self.sparse_model = MTLHadamardGP((full_train_x, full_train_idxs), full_train_y, self.likelihood, self.n_inducing_points, num_tasks=task_num)
        
        # initialize hyperparameters
        min_loss = np.infty
        for k in range(n_restarts):
            noise = 2*np.random.random() # choosing random params
            lengthscale = 100*np.random.random()
            outputscale = 100*np.random.random()
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(noise),
                'covar_module.lengthscale': torch.tensor(lengthscale),
                #'covar_module.outputscale': torch.tensor(outputscale),
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
                output = self.sparse_model(full_train_x, full_train_idxs)
                # calc loss and backprop gradients
                loss = -mll(output, full_train_y)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f lengthscale: %.3f and noise: %.3f' % (
                    i+1, n_iters, 
                    loss.item(),
                    self.sparse_model.covar_module.lengthscale.item(),
                    self.sparse_model.likelihood.noise.item()      
                ))
                optimizer.step()
                
            output = self.sparse_model(full_train_x,full_train_idxs)
            loss = -mll(output, full_train_y)
            if loss < min_loss:
                min_loss = loss
                min_hypers = {
                'likelihood.noise_covar.noise': self.sparse_model.likelihood.noise.item(),
                'covar_module.lengthscale': self.sparse_model.covar_module.lengthscale.item(),
                #'covar_module.base_kernel.outputscale': self.sparse_model.covar_module.base_kernel.outputscale.item(),
                }
                
        return min_loss, min_hypers 
                

    
    
    
    
    
class GPyFullMTL(BaseMTLEstimator):
    """
    Adapted of Full MultiTask GP model from GpyTorch
    
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
       :attr:`bias_only` (bool):
           deprecated. Do not use.
       :attr:`num_tasks` (int):
           number of tasks you are giving model, should be equal to number of datasets

          
    """
    def __init__(self, name='fullGP', num_iters=50, learning_rate=1e-1,
                 noise_covar=1.0, length_scale=100.0, output_scale=1.0, n_inducing_points=500, use_initial=True, num_tasks=1, validate=False):
        """
        Flags
        ------------------------------------------------------------------
        num_iters, learning_rate: These are training parameters that decide how long model should train, at what rate.
        noise_covar, lenght_scale, outputscale: direct model "hyper" parameters that get passed into kernel. 
        n_inducing points: number of points to sample from training, anything over 2000 samples will be a bit slow
        comoposite: boolean, whether the model should implement two seperate kernels for drug features and cell features then combine them
        to make one composite kernel
        validate: boolean, whether model should be validated with predictions, since it is a probabilistic model. I used training data, 
        could also use testing data instead to check for overfitting.
        bias: boolean, whether model should add a bias parameter for each dataset
        num_tasks: number of tasks being passed in
        """
        super().__init__(name, type_met='feature_based')
        
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
        self.training_iter = num_iters
        self.learning_rate = learning_rate # HP
        self.noise_covar = noise_covar
        self.length_scale = length_scale 
        self.output_scale = output_scale 
        self.n_inducing_points = n_inducing_points
        self.use_initial = use_initial
        self.num_tasks = num_tasks
        self.validate = validate
        self.validations = []
        

    def fit(self, x, y, cat_point):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in dictionary form, each key corresponding to a task/dataset
        y: training data in dictionary form, each key corresponding to a task/dataset
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        """
        assert isinstance(x, dict)
        cell_drug_split = cat_point
        
        full_train_x = torch.empty(0)
        y_stack = []
        task_num = 0
        for dataset in list(x.keys()):
            curr_x = torch.tensor(x[dataset].astype(np.float32))
            curr_y = torch.tensor(y[dataset].astype(np.float32))
            perm = torch.randperm(curr_x.size(0))
            idx = perm[:self.n_inducing_points]
            y_idx = perm[:self.n_inducing_points*self.num_tasks]
            x_samples = curr_x[idx]
            y_samples = curr_y[y_idx]
            full_train_x = torch.cat([full_train_x, x_samples])
            y_stack.append(y_samples)
            task_num += 1
        print(task_num,  "tasks")
        # TODO: add cell_drug_split later for composite kernel            
        full_train_y = torch.stack(y_stack, -1)
        
        print("full train and then y shape:",full_train_x.shape, full_train_y.shape, "task_num", task_num)
        self.model = GPyFullMTLModel(full_train_x, full_train_y, self.likelihood, self.n_inducing_points, num_tasks=task_num)
        

        hypers = {
               'likelihood.noise_covar.noise': self.noise_covar, #torch.tensor(self.noise_covar),
               'covar_module.data_covar_module.lengthscale': self.length_scale, #torch.tensor(self.length_scale),
               #'covar_module.data_covar_module.outputscale': self.output_scale #torch.tensor(self.output_scale),

                }
        if self.use_initial:
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
            output =  self.model(full_train_x)
            # calc loss and backprop gradients
            #print(np.mean(self.likelihood(output)))
            loss = -mll(output,full_train_y)
            
            #loss = - torch.distributions.Normal(output.mean, output.variance.sqrt()).log_prob(full_train_y).mean()
            print("#### ACTUAL LOSS ###", -torch.distributions.Normal(output.mean, output.variance.sqrt()).log_prob(full_train_y).mean())    
            loss.backward()
            
            means_list = []
            # handle printouts
            for param in list(self.model.named_parameters()):
                if param[0] == 'likelihood.noise_covar.raw_noise':
                    noise_covar = param[1].data.numpy()[0]
                if param[0] == 'covar_module.data_covar_module.raw_lengthscale':
                    lenscale = param[1].data.numpy()[0][0]
                for j in range(task_num):
                    if param[0] == 'mean_module.base_means.' + str(j) + '.constant':
                        means_list.append(param[1].data.numpy()[0])
        
            print('Iter %d/%d - Loss: %.3f' % (i+1, self.training_iter, loss.item() ), "covariance noise:", self.model.likelihood.noise.item(), \
                 "lengthscale :", lenscale, "bias terms", means_list)
        
            optimizer.step()
            
            # get validations
            if self.validate and i % 10 == 0:
                with torch.no_grad():
                    # evaluation mode
                    self.model.eval()
                    self.likelihood.eval()

                    pred = self.likelihood(self.model(full_train_x))
                    pred = pred.mean
                    rmse = np.sqrt(np.sum(((pred.numpy() - full_train_y.numpy()) ** 2) / len(full_train_y.numpy())))
                    self.validations.append(rmse)
                    
                    # put back in train mode
                    self.model.train()
                    self.likelihood.train()
                    
    def getValidations(self):
        return self.validations

    def _predict(self, x, **kwargs):
        """ 
        Predict using trained model.
        
        inputs
        ------------------------------------------------------------------------------------
        x: training data in dictionary form, each key corresponding to a task/dataset
        cell_drug_split: splits features if we are using composite, ie: 10 features for cell and 10 for drug -> split at idx 10
        """
        assert isinstance(x, dict)
        return_std = False
        if 'return_std' in kwargs.keys():
            return_std = kwargs['return_std']
        
        # put in eval mode
        self.model.eval()
        self.likelihood.eval()
        self.recording = False
        
        task_num = 0
        res = {}
        i = 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for dataset in list(x.keys()):
                curr_x = torch.tensor(x[dataset].astype(np.float32))
                #curr_y = torch.tensor(y[dataset].astype(np.float32))
                observed_pred = self.likelihood(self.model(curr_x))
                res[dataset] = observed_pred.mean[:,i]
                i += 1 
        return res
    
    
     # for use with optuna
    def get_hyper_params(self):
        hparams = {'noise_covar': {'type': 'loguniform', 'values': [.1,2]},
                  'learning_rate': {'type': 'uniform', 'values':  [.0001,.1]},
                  'length_scale': {'type': 'uniform', 'values': [1,60]}, # don't use 60 for synthetic data,
                  #'output_scale': {'type': 'loguniform', 'values': [.1,2]},
                  'num_iters': {'type': 'integer', 'values': [10, 200]}
                  }
        return hparams
    
    # for use with optuna
    def set_hyper_params(self, **kwargs):
        self.noise_covar = kwargs['noise_covar']
        self.learning_rate = kwargs['learning_rate']
        self.length_scale = kwargs['length_scale']
        #self.output_scale = kwargs['output_scale']
        self.training_iter = kwargs['num_iters']
        
        

        
    
    
