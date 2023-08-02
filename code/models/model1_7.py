from curses import echo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import pandas as pd
import numpy as np
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
class Saturation(BaseEstimator, TransformerMixin):
    def __init__(self, a=1.):
        self.a = a
        
    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True) # from BaseEstimator
        return self
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False) # from BaseEstimator
        return 1 - np.exp(-self.a*X)
# %%
#preprocessing - lagged effect
from scipy.signal import convolve2d
import numpy as np
class Carryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.5, length=1):
        self.strength = strength
        self.length = length
    def fit(self, X, y=None):
        X = check_array(X) #checks that it is non zero and finite
        self._check_n_features(X, reset=True) #sets the number of features there should be
        self.sliding_window_ = (
            self.strength ** np.arange(self.length + 1)
        ).reshape(-1, 1)
        return self
    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_) #test out what this does with convolve2d(X.loc[:,['TV']], (0.52*np.arange(5)).reshape(-1,1))
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution

class Hyperparams():
    def __init__(self, params = {
        'adstock__tv_pipe__carryover__strength': UniformDistribution(0, 2),
        'adstock__tv_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__tv_pipe__saturation__a': UniformDistribution(0, 1.0),
        'adstock__radio_pipe__carryover__strength': UniformDistribution(0, 2),
        'adstock__radio_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__radio_pipe__saturation__a': UniformDistribution(0, 1.0),
        'adstock__banners_pipe__carryover__strength': UniformDistribution(0, 2),
        'adstock__banners_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__banners_pipe__saturation__a': UniformDistribution(0, 1.0),
    }):
        self.params = params