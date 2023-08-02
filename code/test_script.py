#%%
from curses import echo
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution
import os
import importlib
import numpy as np

n_trials = 300
results_comp = []
model_name_comp = []
hyperparams_comp = []
best_score_comp = []
pred = []


#%%
#read in data
data = pd.read_csv(
    'https://raw.githubusercontent.com/Garve/datasets/4576d323bf2b66c906d5130d686245ad205505cf/mmm.csv',
    parse_dates=['Date'],
    index_col='Date'
)


data.head()

X = data.drop(columns=['Sales'])
y = data['Sales']

#%%
#grabbing the models built

#Python Script Version
#cwd = os.getcwd()
#dir_list = os.listdir(cwd + '/code/models')

#notebook version
cwd = os.getcwd()
dir_list = os.listdir(cwd + '/models')
dir_list.sort()
#from models.model1_1 import HyperParams, Saturation, Carryover


#%%
for i in dir_list:
    if 'model' in i:
        #log the model used
        model_name_comp.append(i)
        model_library = importlib.import_module("models."+ i[:-3]) #sub libraries: HyperParams, Saturation, Carryover
        Carryover = model_library.Carryover()
        Saturation = model_library.Saturation()
        Hyperparams = model_library.Hyperparams().params
        print('models loaded')
        #setup the pipeline
        adstock = ColumnTransformer(
            [
            ('tv_pipe', Pipeline([
                                ('carryover', Carryover),
                                ('saturation', Saturation)
            ]), ['TV']),
            ('radio_pipe', Pipeline([
                                ('carryover', Carryover),
                                ('saturation', Saturation)
            ]), ['Radio']),
            ('banners_pipe', Pipeline([
                                ('carryover', Carryover),
                                ('saturation', Saturation)
            ]), ['Banners']),
            ],
            remainder='passthrough'
        )
        model = Pipeline([
                        ('adstock', adstock),
                        ('regression', LinearRegression())
        ])

        print('starting cv')
        #CV Setup
        tuned_model = OptunaSearchCV(
            estimator=model,
            param_distributions=Hyperparams,
            n_trials=n_trials,
            cv=TimeSeriesSplit(8),
            random_state=0,
            return_train_score=True, 
        )


        #cv_results = cross_val_score(tuned_model, X, y, cv=TimeSeriesSplit(n_splits=10), n_jobs=-1)
        #print(cv_results)
        #results_comp.append(cv_results)

        tuned_model.fit(X,y)
        trials_df = tuned_model.trials_dataframe()
        row = np.argmax(trials_df['user_attrs_mean_test_score'])

        print('completed cv')
        results_comp.append(dict(trials_df.loc[row,:]))
        best_score_comp.append(tuned_model.best_score_)
        hyperparams_comp.append(tuned_model.best_params_)
        pred.append(tuned_model.predict(X))
results_comp = pd.DataFrame(results_comp)
pred = pd.DataFrame(pred).transpose()
pred['target'] = y

#%%
cols = ['user_attrs_mean_test_score', 'user_attrs_std_test_score',
       'user_attrs_mean_train_score', 'user_attrs_std_train_score', 'user_attrs_split0_test_score',
       'user_attrs_split0_train_score', 'user_attrs_split1_test_score',
       'user_attrs_split1_train_score', 'user_attrs_split2_test_score',
       'user_attrs_split2_train_score', 'user_attrs_split3_test_score',
       'user_attrs_split3_train_score', 'user_attrs_split4_test_score',
       'user_attrs_split4_train_score', 'user_attrs_split5_test_score',
       'user_attrs_split5_train_score', 'user_attrs_split6_test_score',
       'user_attrs_split6_train_score', 'user_attrs_split7_test_score',
       'user_attrs_split7_train_score',
       ]
results_comp[cols]

# %%
model_name_comp
# %%
best_score_comp
#%%
hyperparams_comp
# %%
import matplotlib.pyplot as plt
#plt.plot(results_comp[['user_attrs_mean_test_score', 'user_attrs_std_test_score',
#       'user_attrs_mean_train_score', 'user_attrs_std_train_score']])
results_comp[['user_attrs_mean_test_score', 'user_attrs_mean_train_score','user_attrs_std_test_score',
       'user_attrs_std_train_score']].plot()
plt.legend()
plt.xlabel("Model #")
plt.ylabel("R^2")
plt.title("Variance vs Bias Insight Across Models")
plt.show()

#%%
best_models = [6,8,'target']
pred[best_models].plot(alpha=0.4)
#%%
'''
look for model 1 compared to models 7 and 8 now for improvement
'''