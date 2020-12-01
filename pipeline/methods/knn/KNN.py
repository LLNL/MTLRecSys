"""
surprise wrapper for K nearest neighbors
"""

from surprise import KNNWithZScore, KNNBasic
from ..base import BaseSurpriseSTLEstimator
import numpy as np

class KNN_Normalized(BaseSurpriseSTLEstimator):

    def __init__(self, k, name='KNN_Normalized'):
        super().__init__(name, 'non_feature_based')
        self.k = k
        self.model = KNNWithZScore(k=self.k, verbose=False)

    def _fit(self, x):
        self.model.fit(x)

    def _predict(self, x):
        return self.model.test(x)
    
    def get_hyper_params(self):
        hparams = {'k': {'type': 'integer', 'values':  [2, 13]}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.k = kwargs['k']

    def similarity_matrix(self):
        return self.model.compute_similarities()


class KNN_Basic(BaseSurpriseSTLEstimator):
    """
    Args:
        :attr:`k` (int):
            number of neighbors
        :attr:`sim_options` (optional):
            option from surprise for a similarity metric
    
    """

    def __init__(self, k, name='KNN_Basic', sim_options=None):
        super().__init__(name, 'non_feature_based')
        self.k = k
        if sim_options is not None:
            self.model = KNNBasic(k=self.k, verbose=False, sim_options=sim_options)
        else:
            self.model = KNNBasic(k=self.k, verbose=False)

    def _fit(self, x):
        self.model.fit(x)

    def _predict(self, x):
        return self.model.test(x)

    def get_hyper_params(self):
        hparams = {'k': {'type': 'integer', 'values':  [2, 13]}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.k = kwargs['k']

    def similarity_matrix(self):
        return self.model.compute_similarities()
