import sys
sys.path.append('../')
from design import ModelTraining
from methods.regressor.FFNN import FeedForwardNN
import methods.mtl.MTL_GP as MtlGP
from methods.knn.KNN import KNN_Normalized
from datasets import SyntheticData as SD
import methods.mtl.NCF_MTL as NCF_MTL
from methods.matrix_factorization.MF import SVD_MF, NonNegative_MF



if __name__ == '__main__':
    dataset = SD.SyntheticDataCreator(num_tasks=3,cellsPerTask=400, drugsPerTask=10, function="cosine",
                 normalize=False, noise=.5, graph=False, test_split=0.3)
    dataset.prepare_data()

    hyperparams_mtlmlp = {'batch_size': 64, 'epochs': 150, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': .001,'mlp_lr': .001, 'num_factors': 10, \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf': 0.01, 'verbose': 1}
    hyperparams_mtlmf = {'batch_size': 64, 'epochs': 150, 'layers': '[64,32,16,8]', \
                   'learner': 'adam', 'lr': .001,'mf_lr': .001, 'num_factors': 10, \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf': 0.01, 'verbose': 1}


    methods = [MtlGP.HadamardMTL(num_iters=150, length_scale=57, noise_covar=.24, n_inducing_points=1000, \
                                composite=False, learning_rate=.07, validate=False,bias=False,stabilize=True),
               MtlGP.GPyFullMTL(num_iters=72, length_scale=58.828, noise_covar=0.31587, n_inducing_points=500,  num_tasks=3, learning_rate=0.02729),
               NCF_MTL.Neural_Collaborative_Filtering_FeaturesMTLMLP(hyperparams_mtlmlp,'MTL NCF MLP', 'feature_based'),
               NCF_MTL.Neural_Collaborative_Filtering_FeaturesMTLMF(hyperparams_mtlmf,'NCF_MTL_MF', 'feature_based'),
               SVD_MF(n_factors=10),
               KNN_Normalized(k=10)
              ]

    metrics = ['rmse','mae']

    exp_folder = __file__.strip('.py')
    exp = ModelTraining(exp_folder)
    exp.execute(dataset, methods, metrics, nruns=1) #increase n runs for more accurate error
    exp.generate_report()

    
    
"""

frozen hyperparams-- sometimes they do worse


    hyperparams_mtlmf = {'batch_size': 64, 'epochs': 227, 'layers': '[64,32,16,8]', \
                   'learner': 'sgd', 'lr': 1.00293510662245e-05,'mf_lr': 0.000111324, 'num_factors': 100, \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf':  0.009970084324087263, 'verbose': 1}
    hyperparams_mtlmlp = {'batch_size': 64, 'epochs': 238, 'layers': '[64,32,16,8]', \
                   'learner': 'sgd', 'lr': 0.00042715,'mlp_lr': .001, 'num_factors': 84, \
                   'reg_layers': '[0,0,0,.01]', 'reg_mf':0.0028382, 'verbose': 1}


    methods = [MtlGP.HadamardMTL(num_iters=20, length_scale=59.811557976423494, noise_covar=1.9424144884041885, n_inducing_points=1000, \
                                composite=False, learning_rate=0.05837504582780572, validate=False,bias=False,stabilize=True),
               MtlGP.GPyFullMTL(num_iters=97, length_scale=31.010654440111225, noise_covar= 0.4934115870229582, n_inducing_points=1000,  num_tasks=3, learning_rate= 0.030302689041614293),
               NCF_MTL.Neural_Collaborative_Filtering_FeaturesMTLMLP(hyperparams_mtlmlp,'MTL NCF MLP', 'feature_based'),
               NCF_MTL.Neural_Collaborative_Filtering_FeaturesMTLMF(hyperparams_mtlmf,'NCF_MTL_MF', 'feature_based'),
               SVD_MF(n_factors=10)
              ]

"""
    