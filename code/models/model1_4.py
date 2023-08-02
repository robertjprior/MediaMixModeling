'''
Google 2.6 in model_googleresearch
'''

from curses import echo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import pandas as pd
import numpy as np
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
class Saturation(BaseEstimator, TransformerMixin): #saturation 3
    def __init__(self, S=1., K = 0.5, beta = 1.5):
        self.S = S
        self.K = K
        self.beta = beta
        
    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True) # from BaseEstimator
        return self
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False) # from BaseEstimator
        #return 1 - np.exp(-self.a*X)
        return self.beta - (self.K**self.S*self.beta)/(X**self.S + self.K**self.S) 
        
#def beta_hill(x, S, K, beta):
#     return beta - (K**S*beta)/(x**S+K**S)
#beta_hill(X, self.S, self.K, self.beta)
# %%
#preprocessing - lagged effect
from scipy.signal import convolve2d
import numpy as np
class Carryover(BaseEstimator, TransformerMixin): #carryover2
    def __init__(self, strength=0.5, length=1, sigma = 0.5):
        self.strength = strength #number between 0 and 1 FLOAT
        self.length = length # number between 0 and whatever INT
        self.sigma = sigma #number between 0 and length INT
    def fit(self, X, y=None):
        X = check_array(X) #checks that it is non zero and finite
        self._check_n_features(X, reset=True) #sets the number of features there should be
        self.sliding_window_ = np.array(
            [self.strength **((i - self.sigma)**2) for i in range(self.length+1)]
        ).reshape(-1, 1)
        self.norm_denom = sum([self.strength **((i - self.sigma)**2) for i in range(self.length+1)])
        return self
    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_) #test out what this does with convolve2d(X.loc[:,['TV']], (0.52*np.arange(5)).reshape(-1,1)). the second array is applied reverse order starting with 0 position on array 1 position 0 EX: convolve2d(np.array([[1,2,3,4]]), np.array([[5,4]]))
        convolution = convolution/self.norm_denom
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution

class Hyperparams():
    def __init__(self, params = {
        'adstock__tv_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__tv_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__tv_pipe__carryover__sigma': IntUniformDistribution(0, 6),
        'adstock__tv_pipe__saturation__S': UniformDistribution(0, 1),
        'adstock__tv_pipe__saturation__K': UniformDistribution(0, 1),
        'adstock__tv_pipe__saturation__beta': UniformDistribution(0, 5),
        'adstock__radio_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__radio_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__radio_pipe__carryover__sigma': IntUniformDistribution(0, 6),
        'adstock__radio_pipe__saturation__S': UniformDistribution(0, 1),
        'adstock__radio_pipe__saturation__K': UniformDistribution(0, 1),
        'adstock__radio_pipe__saturation__beta': UniformDistribution(0, 5),
        'adstock__banners_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__banners_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__banners_pipe__carryover__sigma': IntUniformDistribution(0, 6),
        'adstock__banners_pipe__saturation__S': UniformDistribution(0, 1),
        'adstock__banners_pipe__saturation__K': UniformDistribution(0, 1),
        'adstock__banners_pipe__saturation__beta': UniformDistribution(0, 5),
    }):
        self.params = params