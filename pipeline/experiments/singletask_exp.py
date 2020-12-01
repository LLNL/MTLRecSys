
import sys

sys.path.append('../')
from design import ModelTraining
from methods.mtl.MF_MTL import MF_MTL
from methods.matrix_factorization.MF_STL import MF_STL
import matplotlib.pyplot as plt
from methods.regressor.FFNN import FeedForwardNN
from methods.regressor.ExactGP import ExactGPRegression
from methods.regressor.SparseGPCompositeKernel import SparseGPCompositeKernelRegression
from methods.regressor.SparseGP import SparseGPRegression
from methods.matrix_factorization.MF import SVD_MF, NonNegative_MF
from methods.matrix_factorization.CustomInputNCF import Neural_Collaborative_Filtering
from methods.matrix_factorization.FeaturizedNCF import Neural_Collaborative_Filtering_Features
from methods.knn.KNN import KNN_Normalized
from datasets import SyntheticData as SD
import os
import numpy as np
import matplotlib

if __name__ == '__main__':
    dataset = SD.SyntheticDataCreator(num_tasks=3,cellsPerTask=400, drugsPerTask=10, function="cosine",
                 normalize=False, noise=1, graph=False, test_split=0.3)
    
    dataset.prepare_data()

    
    hyperparams_feats = {'batch_size': 64, 'epochs': 150, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': 0.001, 'mf_pretrain': '', 'mlp_pretrain': '', \
                   'num_factors': 10, 'num_neg': 4, 'out': 1, 'path': 'Data/', \
                   'reg_layers': '[0,0,0,0]', 'reg_mf': 0, 'verbose': 1}
    
    hyperparams = {'batch_size': 32, 'epochs': 300, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': .001, 'mf_pretrain': '', 'mlp_pretrain': '', \
                   'num_factors': 8, 'num_neg': 4, 'out': 1, 'path': 'Data/', \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf': 0.01, 'verbose': 1}


    methods = [
               Neural_Collaborative_Filtering_Features(hyperparams_feats,'NCF', 'feature_based',warm_start=False),
               KNN_Normalized(k=10),
               FeedForwardNN([25, 25], 'relu', epochs=60, lr=1e-3),
               SVD_MF(n_factors=10),
               SparseGPRegression(num_iters=57, length_scale=28.99850556026648, noise_covar=0.880495306431355, \
                                  n_inducing_points=500,learning_rate=0.08750861518081232,output_scale=0.2726750961954937),
               SparseGPCompositeKernelRegression(num_iters=55, length_scale_cell=23.909358694255733, length_scale_drug=25.35428771496125, \
                                                 output_scale_cell= 0.23155460333191216, output_scale_drug=2.3750260726401704, \
                                                 noise_covar=2, n_inducing_points=500, learning_rate= 0.009494776750100815),
               #ExactGPRegression(num_iters=10, length_scale=50, noise_covar=1.5) this one is very slow
                Neural_Collaborative_Filtering(hyperparams, 'Ratings matrix NCF','non_feature_based',warm_start=False)
            ]
    metrics = ['rmse','mae']

    exp_folder = __file__.strip('.py')
    exp = ModelTraining(exp_folder)
    exp.execute(dataset, methods, metrics, nruns=1)  # delete after testing
    exp.generate_report()




    
    
    
    
    
