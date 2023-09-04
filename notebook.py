#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from scipy import stats


import linearmodels as lm


# loading tables


aid = pd.read_csv('reg_aid.csv')
distance = pd.read_csv('reg_distance.csv')
population = pd.read_csv('reg_pop.csv')
undernourished = pd.read_csv('reg_undernourishment.csv')
water = pd.read_csv('reg_water.csv')
stability = pd.read_csv('reg_stability.csv')
risk = pd.read_csv('reg_risk.csv')


# introducing lags (optional)


#aid['TIME']=aid['TIME']+1
#aid['TIME']=aid['TIME']+2
#aid['TIME']=aid['TIME']+3


# getting rid of outliers (optional)
#aid=aid[(np.abs(stats.zscore(aid['AID'])) < 3)]


# looking at specifictions (optional)
#aid = aid[aid['AID'] > 10000000]


# merging dataframes into one
df0 = pd.merge(aid, distance, on=['NAME', 'TIME'], how='outer')
df1 = pd.merge(df0, population, on=['NAME', 'TIME'], how='outer')
df2 = pd.merge(df1, undernourished, on=['NAME', 'TIME'], how='outer')
df3 = pd.merge(df2, water, on=['NAME', 'TIME'], how='outer')
df4 = pd.merge(df3, stability, on=['NAME', 'TIME'], how='outer')
df = pd.merge(df4, risk, on=['NAME', 'TIME'], how='outer')


# grouping data
df = df.groupby(['NAME','TIME'], dropna=False, as_index=False).mean()


# saving a consolidated table
df.to_csv('consolidated.csv', index=False)


# reading file and preparing for analysis
df = pd.read_csv('consolidated.csv', index_col = ['NAME', 'TIME'])
#df['YEAR'] = pd.Categorical(df.index.get_level_values('TIME').to_list())
df['YEAR'] = df.index.get_level_values('TIME').to_list()


# transforming into ln where needed
df['AID'] = np.log(df['AID'])
df['POPULATION'] = np.log(df['POPULATION'])
df['DISTANCE'] = np.log(df['DISTANCE'])


# select a year for a linear regression
new_df = df[df['YEAR'] == 2019]


new_df = new_df.dropna() # dropping missing values


# running a multiple linear regression
x = new_df[['DISTANCE', 'POPULATION', 'UNDERNOURISHED', 'WATER', 'STABILITY', 'RISK']] # independent
y = new_df['AID'] # dependent


x = sm.add_constant(x) # taking care of a constant


model = sm.OLS(y, x).fit()
predictions = model.predict(x)
print_model = model.summary()
print(print_model)


print(model.summary().as_latex())


# running a miltiple linear regression with robust errors


model = sm.RLM(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)


print(model.summary().as_latex())


# Pooled OLS


exog = sm.tools.tools.add_constant(df[['POPULATION', 'DISTANCE', 'RISK']])
aid = df['AID']


model = lm.PooledOLS(aid, exog)
pooledOLS_res = model.fit(cov_type='clustered', cluster_entity=True)
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids


# Heteroscedacity test

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue', s=1)
ax.axhline(0, color = 'orange', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 9)
ax.set_ylabel('Residuals', fontsize = 9)
ax.set_title('Homoskedasticity Test', fontsize = 11)

plt.show()


# Fixed Effect and Random Effect panel data regression analysis


risk = sm.add_constant(df['RISK'])
aid = df['AID']

# random effects model
model_re = lm.RandomEffects(aid, risk) 
re_res = model_re.fit() 

# fixed effects model
model_fe = lm.PanelOLS(aid, risk, entity_effects = True) 
fe_res = model_fe.fit()


print(fe_res)
print(re_res)
print(fe_res.summary.as_latex())
print(re_res.summary.as_latex())

# Hausman Test
# as in https://towardsdatascience.com/ \\
# a-guide-to-panel-data-regression-theoretics \\
# -and-implementation-with-python-4c84c5055cf8

import numpy.linalg as la
from scipy import stats
import numpy as np

def hausman(fe, re):
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov
    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval

hausman_results = hausman(fe_res, re_res)

print('chi-Squared: ' + str(hausman_results[0]))
print('degrees of freedom: ' + str(hausman_results[1]))
print('p-Value: ' + str(hausman_results[2]))


# 2-Stage Least Squares Estimation


# Verify that the selected instruments satisfy the relevance condition
x = new_df[['WATER', 'STABILITY', 'UNDERNOURISHED']] #independent
y = new_df['RISK'] #dependent


x = sm.add_constant(x) # taking care of a constant


model = sm.RLM(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)
print(model.summary().as_latex())


# IV2SLS with three instruments - WATER, STABILITY, UNDERNOURISHED


model_iv2sls = lm.IV2SLS(aid, exog, None, df[['WATER', 'STABILITY', 'UNDERNOURISHED']]).fit(cov_type="unadjusted")
print(model_iv2sls)
print(model_iv2sls.summary.as_latex())
