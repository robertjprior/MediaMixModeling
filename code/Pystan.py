'''
Now that the dataset has been simulated it is time to fit the model. The paper uses STAN, however, I use Python/PyMC3.
'''

import arviz as az
import pymc3 as pm
with pm.Model() as m:
    alpha = pm.Beta('alpha'          , 3 , 3  , shape=3)
    theta = pm.Uniform('theta'       , 0 , 12 , shape=3)
    k     = pm.Beta('k'              , 2 , 2  , shape=3)
    s     = pm.Gamma('s'             , 3 , 1 , shape=3)
    beta  = pm.HalfNormal('beta'     , 1      , shape=3)
    ru    = pm.HalfNormal('intercept', 5) 
    lamb  = pm.Normal('lamb'         , 0 , 1) 
    noise = pm.InverseGamma('noise'  , 0.05, 0.0005) 
    
    transpose_m1 = [beta_hill(x, s[0], k[0], beta[0]) for x in carryover(media_1, alpha[0], L, theta = theta[0], func='delayed')]
    transpose_m2 = [beta_hill(x, s[1], k[1], beta[1]) for x in carryover(media_2, alpha[1], L, theta = theta[1], func='delayed')]
    transpose_m3 = [beta_hill(x, s[2], k[2], beta[2]) for x in carryover(media_3, alpha[2], L, theta = theta[2], func='delayed')]
    
    
    y_hat = pm.Normal('y_hat', mu=ru + transpose_m1 + transpose_m2 + transpose_m3 + lamb * price_variable,
                  sigma=noise, 
                  observed=y)
trace = pm.fit(method='svgd')