import numpy as np
import statsmodels.api as sm

x,y = np.loadtxt('data/xy-002.csv',delimiter=',',unpack=True,skiprows=1)
X_plus_one = np.stack( (np.ones(x.size),x), axis=-1)
ols = sm.OLS(y, X_plus_one)
ols_result = ols.fit()
print(ols_result.summary())