"""
Smoke test to make sure every model is working correctly.
Should check these models:
1. KNN
2. MF and NNMF
3. All GPs
4. Both NCFs
5. Vanilla NN
"""
import sys
from design import ModelTraining
from methods.regressor.ExactGP import ExactGPRegression
from methods.regressor.SparseGPCompositeKernel import SparseGPCompositeKernelRegression
from methods.regressor.ExactGPCompositeKernel import ExactGPCompositeKernelRegression
from methods.matrix_factorization.CustomInputNCF import Neural_Collaborative_Filtering
from methods.matrix_factorization.FeaturizedNCF import Neural_Collaborative_Filtering_Features
from methods.regressor.SparseGP import SparseGPRegression
from methods.regressor.FFNN import FeedForwardNN
from methods.matrix_factorization.MF import SVD_MF, NonNegative_MF
from methods.knn.KNN import KNN_Normalized, KNN_Basic
from datasets import SyntheticData as SD
import methods.mtl.MTL_GP as MtlGP
import methods.mtl.NCF_MTL as NCF_MTL


sys.path.append('experiments/')
import os
os.chdir("/g/g16/ladd12/mtl4c_drugresponse/pipeline/experiments/")

import pytest


metrics = ['rmse','mae']
dataset = SD.SyntheticDataCreator(num_tasks=3,cellsPerTask=400, drugsPerTask=10, function="cosine",
                 normalize=False, noise=1, graph=False, test_split=0.3)
dataset.prepare_data()


class TestKNN:
    def test_KNNNormalized(self):
        exp_folder = "knnnorm_test"
        exp = ModelTraining(exp_folder)
        methods = [KNN_Normalized(k=10)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction
            
    def test_KNNBasic(self):
        exp_folder = "knnbasic_test"
        exp = ModelTraining(exp_folder)
        methods = [KNN_Basic(k=10)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction
       
    
class TestMF:
    def test_SVDMF(self):
        exp_folder = "svdmf_test"
        exp = ModelTraining(exp_folder)
        methods = [SVD_MF(n_factors=30)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction
            
    def test_NonNegative_MF(self):
        exp_folder = "svdnnmf_test"
        exp = ModelTraining(exp_folder)
        methods = [NonNegative_MF(n_factors=30)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction
            
class TestGPs:
    def test_ExactGP(self):
        exp_folder = "exactGP_test"
        exp = ModelTraining(exp_folder)
        methods = [ExactGPRegression(num_iters=10, length_scale=50, noise_covar=1.5)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction
            
    def test_SparseGP(self):
        exp_folder = "sparseGP_test"
        exp = ModelTraining(exp_folder)
        methods = [SparseGPRegression(num_iters=10, length_scale=50, noise_covar=1.5, n_inducing_points=250)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction 
            
    def test_HadamardGP(self):
        exp_folder = "HadamardGP_test"
        exp = ModelTraining(exp_folder)
        methods = [MtlGP.HadamardMTL(num_iters=50, length_scale=20, noise_covar=.9, n_inducing_points=500, \
                                composite=False, learning_rate=.1, validate=False)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction 
            
#TODO: fix this method            
    def test_FullMTL(self):
        metrics = ['rmse']
        ddataset = SD.SyntheticDataCreator(num_tasks=3,cellsPerTask=500, drugsPerTask=10, function="gauss",
             normalize=False, noise=1, graph=False, test_split=0.3)
        dataset.prepare_data()
        exp_folder = "fullMtlGP_test"
        exp = ModelTraining(exp_folder)
        methods = [MtlGP.GPyFullMTL(num_iters=50, length_scale=20, noise_covar=.9, n_inducing_points=500, num_tasks=3)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction       
            
    def test_ExactCompGP(self):
        exp_folder = "exactCompGP_test"
        exp = ModelTraining(exp_folder)
        methods = [ExactGPCompositeKernelRegression(num_iters=35, learning_rate=1e-1, noise_covar=1.0, 
                 length_scale_cell=30.0, output_scale_cell=1.0, 
                 length_scale_drug=30.0, output_scale_drug=1.0)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction'
            

# TODO: method works but return shape is wrong becuase of inducing points     
    def test_SparseCompGP(self):
        exp_folder = "sparseCompGP_test"
        exp = ModelTraining(exp_folder)
        methods = [SparseGPCompositeKernelRegression(num_iters=15, learning_rate=1e-1, noise_covar=1.0, 
                 length_scale_cell=30.0, output_scale_cell=1.0, 
                 length_scale_drug=30.0, output_scale_drug=1.0)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction 

class TestNCF:
    def test_NCFFeat(self):
        exp_folder = "ncfFeat_test"
        exp = ModelTraining(exp_folder)
        hyperparams_feats = {'batch_size': 64, 'epochs': 150, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': 0.001, 'mf_pretrain': '', 'mlp_pretrain': '', \
                   'num_factors': 8, 'num_neg': 4, 'out': 1, 'path': 'Data/', \
                   'reg_layers': '[0,0,0,0]', 'reg_mf': 0, 'verbose': 1}
        methods = [Neural_Collaborative_Filtering_Features(hyperparams_feats,'Neural Collaborative Filtering', 'feature_based',warm_start=True)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 2 # arbitrary threshold for reasonable prediction
            
    def test_NCF(self):
        hyperparams = {'batch_size': 32, 'epochs': 200, 'layers': '[64,32,16,8]', \
                   'learner': 'rmsprop', 'lr': 0.001, 'num_factors': 8, 'num_neg': 4, \
                   'reg_layers': '[0,0,0,0]', 'reg_mf': 0.0, 'verbose': 1, 'warm_start':False}
        exp_folder = "ncf_test"
        exp = ModelTraining(exp_folder)
        methods = [Neural_Collaborative_Filtering(hyperparams, 'Ratings matrix NCF','non_feature_based',warm_start=False)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 2 # arbitrary threshold for reasonable prediction 
     
    def test_NCF_MTL_MLP(self):
        hyperparams_mtlmlp = {'batch_size': 64, 'epochs': 150, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': .001,'mlp_lr': .001, 'num_factors': 10, \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf': 0.01, 'verbose': 1}
        methods = [NCF_MTL.Neural_Collaborative_Filtering_FeaturesMTLMLP(hyperparams_mtlmlp,'MTL NCF MLP', 'feature_based')]
        exp_folder = "ncf_test_MLP"
        exp = ModelTraining(exp_folder)
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 2 # arbitrary threshold for reasonable prediction 
        
    def test_NCF_MTL_MF(self):
        hyperparams_mtlmf = {'batch_size': 64, 'epochs': 150, 'layers': '[64,32,16,8]', \
               'learner': 'adam', 'lr': .001,'mf_lr': .001, 'num_factors': 10, \
               'reg_layers': '[0,0,0,.01]', 'reg_mf': 0.01, 'verbose': 1}
        methods = [NCF_MTL.Neural_Collaborative_Filtering_FeaturesMTLMF(hyperparams_mtlmf,'NCF_MTL_MF', 'feature_based')]
        exp_folder = "ncf_test_MF"
        exp = ModelTraining(exp_folder)
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 2 # arbitrary threshold for reasonable prediction 

            
class TestNN:
    def test_NN(self):
        ddataset = SD.SyntheticDataCreator(num_tasks=3,cellsPerTask=400, drugsPerTask=10, function="cosine",
             normalize=True, noise=1, graph=False, test_split=0.3)
        dataset.prepare_data()
        exp_folder = "NN_test"
        exp = ModelTraining(exp_folder)
        methods = [FeedForwardNN([25, 25], 'relu', epochs=60, lr=1e-3)]
        exp.execute(dataset, methods, metrics, nruns=1)
        df = exp.getResultsWrapper()
        rmses = df['Value'].values
        for rmse in rmses:
            assert rmse < 1.5 # arbitrary threshold for reasonable prediction 
        
             
            
