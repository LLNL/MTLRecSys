""" 
Wrapper class for surpise methods for collaborative filtering

SVD MF and NNMF

TODO: add formulas here? or in docs?

n_factors: rank of approximation for singular value decomposition
n_epochs: ?
"""
from surprise import SVD, NMF
from ..base import BaseSurpriseSTLEstimator  

class SVD_MF(BaseSurpriseSTLEstimator):
    """
    Matrix Factorization 
    
    Args:
        :attr:`n_factors` (int): 
            number of latent vectors/factors for matrix factorization
        :attr:`n_epochs` (int): 
            Integer, The number of iteration of the SGD procedure. Default is 20
    
    see https://surprise.readthedocs.io/en/stable/matrix_factorization.html for more info
    """
    def __init__(self, n_factors, n_epochs=50, name='SVD_MF'):
        super().__init__(name, 'non_feature_based')
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs)

    def _fit(self, x):
        self.model.fit(x)

    def _predict(self, x):
        return self.model.test(x)
    
    def get_hyper_params(self):
        hparams = {'n_factors': {'type': 'integer', 'values':  [2, 150]}, 'n_epochs': {'type': 'integer', 'values':  [2, 150]}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.n_factors = kwargs['n_factors']

class NonNegative_MF(BaseSurpriseSTLEstimator):
    """
    Nonnegative Matrix Factorization
    
    Args:
        :attr:`n_factors` (int): 
            number of latent vectors/factors for matrix factorization
        :attr:`n_epochs` (int): 
            Integer, The number of iteration of the SGD procedure. Default is 20
    
    see https://surprise.readthedocs.io/en/stable/matrix_factorization.html for more info
    """

    def __init__(self, n_factors, n_epochs=50, name='NonNegative_MF'):
        super().__init__(name, 'non_feature_based')
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.model = NMF(n_factors=self.n_factors, n_epochs=self.n_epochs)

    def _fit(self, x):
        self.model.fit(x)

    def _predict(self, x):
        return self.model.test(x)
    
    def get_hyper_params(self):
        hparams = {'n_factors': {'type': 'integer', 'values':  [2, 150]}, 'n_epochs': {'type': 'integer', 'values':  [2, 150]}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.n_factors = kwargs['n_factors']