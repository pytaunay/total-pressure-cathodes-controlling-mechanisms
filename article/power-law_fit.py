# MIT License
# 
# Copyright (c) 2021- Pierre-Yves Taunay 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
File: power-law_fit.py
Author: Pierre-Yves Taunay
Date: August, 2021
Description: performs the power law fit.

Recreates Table II and Fig 5a (w/o error bars) in "Total pressure in thermionic orificed hollow 
cathodes"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_hdf("cathode_database.h5",key="data")
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6','PI7']].dropna()

PI1 = np.array(pidata['PI1'])
Y = np.log10(PI1)
X = (np.array(pidata[['PI2','PI3','PI4','PI5','PI6','PI7']]))

X0 = np.ones(len(Y))
X1 = np.log10(X[:,0]) # PI2
X2 = np.log10(X[:,1]) # PI3
X3 = np.log10(X[:,2]) # PI4 
X4 = np.log10(X[:,3]) # PI5 
X5 = np.log10(X[:,4]) # PI6
X6 = np.log10(X[:,5]) # PI7

Xreg = np.array([X0,X1,X2,X3,X4,X5,X6]).T

### Perform regression
reg = LinearRegression()
reg.fit(Xreg,Y)

# Store data
coef = np.copy(reg.coef_)
coef[0] = np.copy(reg.intercept_)


### 95% Confidence intervals
Yp = reg.predict(Xreg) # Predicted Y
mse = mean_squared_error(Y,Yp)
r2 = r2_score(Y,Yp)

# Covariance matrix
Ainv = mse * np.linalg.inv(Xreg.T@Xreg)

# Compute interval
# see, e.g., G. James, et al., "An Introduction to Statistical Learning," 2017. p.66
# or https://stats.stackexchange.com/questions/136157/general-mathematics-for-confidence-interval-in-multiple-linear-regression
vals = 1.96 * np.sqrt(np.diag(Ainv))

### Printouts
# Recreate Table II
print("Pi-product","Lower bound","Value","Upper bound")
idx = 0
for c, e in zip(coef, vals):
    if idx == 0:
        pi_str = "C"
        idx = 2
    else:
        pi_str = "Pi" + str(idx)
        idx = idx + 1
    
    print(pi_str,":",f'{c-e:.3}',f'{c:.3}',f'{c+e:.3}')
        
### Plot the regression w/o error bars
plt.loglog(10**Yp,10**Y,'ko')
plt.legend(["Power law fit"])
plt.xlabel("$\Gamma (\Pi)$")
plt.ylabel("$\Pi_1$")

# Perfect correlation
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')

