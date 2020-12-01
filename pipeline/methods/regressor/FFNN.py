import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_sc
from ..base import BaseOwnSTLEstimator


activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}


class NpDataset(Dataset):
    """ Torch wrapper for a numpy array. This is to 
    make possible to use two numpy arrays (x and y) as 
    an iterator for a Pytorch DataLoader. """
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        if len(y.shape) < 2:
            y = y[:, None]
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __add__(self, ds):
        return torch.utils.data.TensorDataset(torch.from_numpy(self.x + ds))

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class NN(nn.Module):
    """
    Vanilla Neural Network implementation
    
    Args:
        :attr:`input` (int): 
            dimension of input data
        :attr:`arch` (list):
            list specifying architecture for each layer
        :attr:`activation` (string):
            string specifying what activation to use ie: "ReLU" or "Sigmoid" or "TanH" 

    """

    def __init__(self, input_dim, arch, activation):
        super(NN, self).__init__()
        assert activation in activations
        arch = [input_dim] + arch + [1]
        self.linears = nn.ModuleList([nn.Linear(arch[i - 1], arch[i]) for i in range(1, len(arch))])
        self.activation = activations[activation]

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.linears[-1](x)
        return x


def train(model, device, train_loader, optimizer,
          epoch, verbose, log_interval=10):
    """ Train model for one epoch. """
    model.train()
    loss_epoch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    if verbose:
        print('Epoch {}: {}'.format(epoch, loss_epoch))



# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


class FeedForwardNN(BaseOwnSTLEstimator):

    def __init__(self, arch, activation='relu',
                 batch_size=128, epochs=50, lr=1e-3,
                 name='Vanilla Neural Net', verbose=False):
        super().__init__(name, 'feature_based', output_shape='array')
        self.arch = arch
        self.activation = activation
        self.net = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fit(self, x, **kwargs):

        self.net = NN(x.shape[1], self.arch, self.activation)

        y = kwargs['y']  # output variable        

        train_dataset = NpDataset(x, y)
        # kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        milestones = (np.arange(0.1, 1, step=0.2) * self.epochs).astype(int)
        scheduler = lr_sc.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        for epoch in range(1, self.epochs + 1):
            train(self.net, self.device, train_loader, optimizer, epoch, self.verbose)
            # test(self.net, device, test_loader)  # TODO: create validation set
            scheduler.step()

    def _predict(self, x):
        self.net.eval()
        x = torch.from_numpy(x).type(torch.float).to(self.device)
        with torch.no_grad():
            #pred = self.net(torch.from_numpy(x.astype(np.float32)))
            pred = self.net(x)
        return pred
    
    def get_hyper_params(self):
        hparams = {'activation': {'type': 'categorical', 'values': ['relu','sigmoid','tanh']},
                   'batch_size': {'type': 'integer', 'values': [2, 128]},
                   'lr': {'type': 'loguniform', 'values': [1e-4, 1e-2]}}
        return hparams

    def set_hyper_params(self, **kwargs):
        self.activation = kwargs['activation']
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['lr']
