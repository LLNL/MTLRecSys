
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import torch
from surprise import Dataset, Reader
import sys
sys.path.append('../../')
from UTILS.Logger import Logger


def dataset_to_df(ratings):
    rows = list()
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[1]):
            if np.isfinite(ratings[i, j]):
                rows.append([i, j, ratings[i, j]])
    df = pd.DataFrame(rows, columns=['userID', 'itemID', 'rating'])
    return df

def check_dataframe(x):
    """ 
    check if input x is a dataframe if so
        convert it to numpy array
        
    """
    ntasks = len(x)
    if isinstance(x[0], pd.DataFrame):  # if x is a pandas dataframe
        for t in range(ntasks):
            x[t] = x[t].values.astype(float)
    return x


class BaseEstimator(object):
    """ 
    Abstract class representing a generic STL Method. 
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, type_met, paradigm, output_shape):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
            :attr:`type_met` (string):
            whether the method is "feature_based" or "non_feature_based"
            :attr:`paradigm` (string):
            whether the method is "stl" or "mtl"
            :attr:`paradigm` (string):
            
        """
        self.name = name
        self.type = type_met
        self.paradigm = paradigm
        self.output_shape = output_shape  # array or matrix
        self.output_directory = ''
        self.logger = Logger()

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self):
        """
        fit model parameters 
        
        Args
            :attr:`x` (np.array):
            np.array w/shape (nsamples,nfeatures)
            :attr:`y` (np.array):
            np.array w/shape (nsamples,1)
            
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Perform prediction.
        
        Args
            :attr:`x` (np.array):
            np.array w/shape (nsamples,nfeatures)
     
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Perform prediction.
        
        Args
            :attr:`x` (np.array):
            np.array w/shape (nsamples,nfeatures)
            :attr:`y` (np.array):
            np.array w/shape (nsamples,1)
            
        Return
            :attr:`results` (np.array):
            np.array of errors
            
        """
        pass

    @abstractmethod
    def set_params(self):
        """
        Set method's parameters for optuna
            
        """
        pass

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        
        Args:
            :attr:`output_dir` (str): 
            path to output directory.
            
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())


class BaseSurpriseSTLEstimator(BaseEstimator):

    def __init__(self, name, type_met):
        super().__init__(name, type_met, 'stl', 'matrix')
        self.rating_scale = ()

    def fit(self, ratings, **kwargs):
        df_ratings = dataset_to_df(ratings)
        self.rating_scale = (df_ratings['rating'].min(), df_ratings['rating'].max())
        reader = Reader(rating_scale=self.rating_scale)
        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(df_ratings, reader)
        raw_trainset = [data.raw_ratings[i] for i in range(data.df.shape[0])]
        trainset = data.construct_trainset(raw_trainset)
        self._fit(trainset, **kwargs)

    def predict(self, ratings, **kwargs):
        pred = np.zeros_like(ratings) * np.nan
        for i in range(ratings.shape[0]):  # items (cell lines)
            for u in range(ratings.shape[1]):  # users (drugs)
                if np.isfinite(ratings[i, u]):  # need prediction
                    p = self.model.predict(i, u)
                    pred[i, u] = p.est
        return pred


class BaseOwnSTLEstimator(BaseEstimator):

    def __init__(self, name, type_met, output_shape='matrix'):
        super().__init__(name, type_met, 'stl', output_shape)
        self.nb_dims = -1        

    def fit(self, x, **kwargs):
        """ 
        
        fit model parameters
        
        """
        self._fit(x, **kwargs)  # call child's class specific fit

    def predict(self, x, **kwargs):
        yhat = self._predict(x)  # call child's class specific predict
        if type(yhat) is torch.Tensor:
            return yhat.detach().cpu().numpy()
        else:
            return yhat


class BaseMTLEstimator(BaseEstimator):
    """
    Base class for multitask learning estimators
    
    """

    def __init__(self, name, type_met):
        """
        Class initialization.
        
        Args
            :attr:`name` (string):
            the name attribute of the method.
            :attr:`type_met` (string):
            whether the method is "feature_based" or "non_feature_based"
            :attr:`paradigm` (string):
            whether the method is "stl" or "mtl"
            :attr:`paradigm` (string):
         """   
        super().__init__(name, type_met, 'mtl', 'matrix')
        self.nb_dims = -1

    def fit(self, x, **kwargs):
        """
        fit model parameters 
        
        Args
            :attr:`x` (dict):
            dictionary with keys corresponding to feature vectors for each task eg: {"CCLE": np.array w/shape (nsamples,nfeatures)}
            :attr:`y` (dict):
            dictionary with keys corresponding to output vectors for each task eg: {"CCLE": np.array w/shape (nsamples,1)}
            
        """
        assert isinstance(x, dict)
        self._fit(x, **kwargs)  # call child's class specific fit

    def predict(self, x, **kwargs):
        """
        predict model parameters 
        
        Args
            :attr:`x` (dict):
            dictionary with keys corresponding to feature vectors for each task eg: {"CCLE": np.array w/shape (nsamples,nfeatures)}
            
        """
        assert isinstance(x, dict)
        yhat = self._predict(x)  # call child's class specific predict
        for k in yhat.keys():
            if type(yhat[k]) is torch.Tensor:
                yhat[k] = yhat[k].detach().numpy()
        return yhat