"""
pipeline wrapper for really bad predict normal benchmark
considering deleteing
"""

from abc import ABC
from surprise import NormalPredictor
from ..base import BaseSurpriseSTLEstimator  


class NormalPredictorX(BaseSurpriseSTLEstimator, ABC):

    def __init__(self, name="Normal Prediction"):
        super().__init__(name, 'non_feature_based')
        self.model = NormalPredictor()

    def _fit(self, x):
        self.model.fit(x)

    def _predict(self, x):
        return self.model.test(x)

