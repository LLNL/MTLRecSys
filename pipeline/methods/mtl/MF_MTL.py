import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from ..base import BaseMTLEstimator  


def my_loss_mtl(pred, obs, **kwargs):
    """ Loss function for the MTL model. """
    rho1 = kwargs['rho1']
    rho2 = kwargs['rho2']
    params = kwargs['params']

    ps = dict()
    for k in params.keys():
        if k.startswith('ps'):
            ps[k.split('.')[-1]] = params[k]
    q = params['q']
    mse = 0
    p_norms = 0
    for k in obs.keys():
        inds = obs[k] == obs[k]  # this finds the non-nans
        mse += (pred[k][inds] - obs[k][inds]).pow(2).mean()
        p_norms += torch.norm(ps[k])
    return mse + rho1 * p_norms + rho2 * torch.norm(q)


class PSharedQ(nn.Module):
    def __init__(self, tasks, ps_dim, q_dim, num_factors):
        super(PSharedQ, self).__init__()
        self.tasks = tasks
        # initialize parameters
        ps = dict()
        for k in self.tasks:
            ps[k] = nn.Parameter(torch.randn(ps_dim[k], num_factors, dtype=torch.float32))
        self.ps = nn.ParameterDict(ps)
        self.q = nn.Parameter(torch.randn(q_dim, num_factors, dtype=torch.float32))

    def forward(self):
        y_pred = dict()
        for k in self.tasks:
            y_pred[k] = self.ps[k].mm(self.q.t())
        return y_pred


class MF_MTL(BaseMTLEstimator):

    def __init__(self, num_factors, rho1, rho2,
                 learning_rate=1e-2, num_epochs=1000, name='MF_MTL'):
        super().__init__(name, 'non_feature_based')
        self.num_factors = num_factors
        self.rho1 = rho1
        self.rho2 = rho2
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None 

    def _fit(self, ratings):
        """ Fit MTL Matrix Factorization model. """
        ps_dim = dict()
        for k in ratings.keys() :
            ps_dim[k] = ratings[k].shape[0]
            ratings[k] = torch.from_numpy(ratings[k].astype(np.float32))
        q_dim = ratings[k].shape[1]

        self.model = PSharedQ(list(ratings.keys()), ps_dim, q_dim, self.num_factors)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        for t in range(self.num_epochs):
            optimizer.zero_grad()
            y_pred = self.model()
            loss = my_loss_mtl(y_pred, ratings, params=self.model.state_dict(),
                               rho1=self.rho1, rho2=self.rho2)
            loss.backward()
            optimizer.step()
            # scheduler.step()

    def _predict(self, _):
        return self.model()
