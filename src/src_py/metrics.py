import numpy as np
import statsmodels.api as sm

x,y = np.loadtxt('data/xy-002.csv',delimiter=',',unpack=True,skiprows=1)
X_plus_one = np.stack( (np.ones(x.size),x), axis=-1)
ols = sm.OLS(y, X_plus_one)
ols_result = ols.fit()
print(ols_result.summary())


import numpy as np
import matplotlib.pyplot as plt

# Dane
x = np.linspace(0, 10, 100)
const1 = 3423.916
const2 = 4824.28
lower_x1 = -19.771
upper_x1 = 29.505

# Skrajne przebiegi linii regresji
lower_y = const1 + lower_x1 * x
upper_y = const2 + upper_x1 * x

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(x, lower_y, color='red', label='dolny przebieg x1')
plt.plot(x, upper_y, color='blue', label='górny przebieg x1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Skrajne przebiegi linii regresji dla współczynnika x1')
plt.legend()
plt.grid(True)
plt.show()