# %%
#imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import pandas as pd

#%%
#read in data
data = pd.read_csv(
    'https://raw.githubusercontent.com/Garve/datasets/4576d323bf2b66c906d5130d686245ad205505cf/mmm.csv',
    parse_dates=['Date'],
    index_col='Date'
)

#%%
data.head()
# %%
X = data.drop(columns=['Sales'])
y = data['Sales']
lr = LinearRegression()
print(cross_val_score(lr, X, y, cv=TimeSeriesSplit())) #time series split can be: (35,) (68,) (101,) (134,) (167,)
#if n_splits is 5 and test size =2 and X is 200 obs, then each 5 splits will be 190, 192, 194, 196, 198 with those 2 intervals
# %%
'''
tscv = TimeSeriesSplit(n_splits=5,  test_size=2)
for train_index, test_index in tscv.split(X):
    print(train_index.shape)
    print(test_index.shape)
'''
# %%
lr.fit(X,y)
lr

#import statsmodels.api as sm
#X = sm.add_constant(X.ravel())
#results = sm.OLS(y,x).fit()
#results.summary()  
# %%
print('Contribution Plot')
weights = pd.Series(
    lr.coef_,
    index=X.columns
)
base = lr.intercept_
unadj_contributions = X.mul(weights).assign(Base=base)
adj_contributions = (unadj_contributions
                     .div(unadj_contributions.sum(axis=1), axis=0)
                     .mul(y, axis=0)
                    ) # contains all contributions for each day
ax = (adj_contributions[['Base', 'Banners', 'Radio', 'TV']]
      .plot( #.area could go after plot and before (
          figsize=(16, 10),
          linewidth=1,
          title='Predicted Sales and Breakdown',
          ylabel='Sales',
          xlabel='Date')
     )
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1], labels[::-1],
    title='Channels', loc="center left",
    bbox_to_anchor=(1.01, 0.5)
)
# %%
error_df = pd.DataFrame(lr.predict(X), columns=['pred'])
print(error_df.shape)
error_df['actual'] = y.tolist()
error_df['error'] = error_df['pred'] - error_df['actual']
error_df.index = X.index

error_df.head()
# %%
error_df['error'].plot(ylabel ='Sales', xlabel='Date')
error_df.plot(ylabel ='Error', xlabel='Date')
# %%
#preprocessing - saturation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
class ExponentialSaturation(BaseEstimator, TransformerMixin):
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


def beta_hill(x, S, K, beta):
    return beta - (K**S*beta)/(x**S+K**S)
class GoogleSaturation(BaseEstimator, TransformerMixin):
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
        return beta_hill(X, self.S, self.K, self.beta)
# %%
#preprocessing - lagged effect
from scipy.signal import convolve2d
import numpy as np
class ExponentialCarryover(BaseEstimator, TransformerMixin):
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

#%%
'''
Option to create lags by editing data - would remove our saturation effect though
class InsertLags(BaseEstimator, TransformerMixin):
    """
    Automatically Insert Lags
    """
    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        original_cols=list(range(len(X[0,:])))
        for lag in self.lags:
            X_lagged=pd.DataFrame(X[:,original_cols]).shift(lag).as_matrix()
            X=np.concatenate((X,X_lagged), axis=1)
        return X
'''
# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
adstock = ColumnTransformer(
    [
     ('tv_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['TV']),
     ('radio_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['Radio']),
     ('banners_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['Banners']),
    ],
    remainder='passthrough'
)
model = Pipeline([
                  ('adstock', adstock),
                  ('regression', LinearRegression())
])

#%%
model.fit(X, y)
print(cross_val_score(model, X, y, cv=TimeSeriesSplit()).mean())

#%%
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution
tuned_model = OptunaSearchCV(
    estimator=model,
    param_distributions={
        'adstock__tv_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__tv_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__tv_pipe__saturation__a': UniformDistribution(0, 0.01),
        'adstock__radio_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__radio_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__radio_pipe__saturation__a': UniformDistribution(0, 0.01),
        'adstock__banners_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__banners_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__banners_pipe__saturation__a': UniformDistribution(0, 0.01),
    },
    n_trials=100, #1000,
    cv=TimeSeriesSplit(),
    random_state=0
)
# %%
print(cross_val_score(tuned_model, X, y, cv=TimeSeriesSplit()))

#%%
tuned_model.fit(X, y)

#%%
tuned_model.best_trial_

#%%
trials_comp = tuned_model.trials_dataframe()

#%%
print(tuned_model.best_params_)
print(tuned_model.best_estimator_.named_steps['regression'].coef_)
print(tuned_model.best_estimator_.named_steps['regression'].intercept_)
# %%
#what the data looks like after transformers and before going into LR
adstock_data = pd.DataFrame(
    tuned_model.best_estimator_.named_steps['adstock'].transform(X),
    columns=X.columns,
    index=X.index
)

#%%

# %%
