import numpy as np
import torch
import torch.nn as nn
from ..base import BaseOwnSTLEstimator  


def my_loss_stl(pred, obs, **kwargs):
    """ STL loss function """
    rho1 = kwargs['rho1']
    rho2 = kwargs['rho2']
    params = kwargs['params']
    p = params['p']
    q = params['q']

    inds = obs == obs  # this finds the non-nans
    mse = (pred[inds] - obs[inds]).pow(2).mean()

    loss = mse + rho1 * torch.norm(p) + rho2 * torch.norm(q)
    return loss


class PQ(nn.Module):
    def __init__(self, p_dim, q_dim, num_factors):
        super(PQ, self).__init__()
        # initialize parameters
        self.p = nn.Parameter(torch.randn(p_dim, num_factors, dtype=torch.float32))
        self.q = nn.Parameter(torch.randn(q_dim, num_factors, dtype=torch.float32))

    def forward(self):
        return self.p.mm(self.q.t())


class MF_STL(BaseOwnSTLEstimator):

    def __init__(self, num_factors, rho1, rho2,
                 learning_rate=1e-2, num_epochs=1000, name='MF_STL'):
        super().__init__(name, 'non_feature_based')
        self.num_factors = num_factors
        self.rho1 = rho1
        self.rho2 = rho2
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None

    def _fit(self, ratings):
        """ Fit MTL Matrix Factorization model. """
        ratings = torch.from_numpy(ratings.astype(np.float32))

        self.model = PQ(ratings.shape[0], ratings.shape[1], self.num_factors)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        for t in range(self.num_epochs):
            optimizer.zero_grad()
            y_pred = self.model()
            loss = my_loss_stl(y_pred, ratings, params=self.model.state_dict(),
                               rho1=self.rho1, rho2=self.rho2)
            loss.backward()
            optimizer.step()
            # scheduler.step()

    def _predict(self, _):
        return self.model()

    def get_hyper_params(self):
        hparams = {'num_factors': {'type': 'integer', 'values': [2, 10]},
                   'rho1': {'type': 'loguniform', 'values': [1e-3, 100]},
                   'rho2': {'type': 'loguniform', 'values': [1e-3, 100]}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.num_factors = kwargs['num_factors']
        self.rho1 = kwargs['rho1']
        self.rho2 = kwargs['rho2']
