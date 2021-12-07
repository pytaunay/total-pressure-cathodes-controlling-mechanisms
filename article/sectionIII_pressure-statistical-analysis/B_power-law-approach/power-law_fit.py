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

Recreates Table II, Figs 5a (w/o error bars or gray scale), 5b, 6, 7 in "Total pressure in 
thermionic orificed hollow cathodes"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

########################################
############# GET DATA #################
########################################
data = pd.read_hdf("../../../data/cathode_database.h5",key="data")
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

########################################
############ REGRESSION ################
########################################
### Perform regression
reg = LinearRegression()
reg.fit(Xreg,Y)
Yp = reg.predict(Xreg) # Predicted Y

# Store data
coef = np.copy(reg.coef_)
coef[0] = np.copy(reg.intercept_)

### Plot by cathode: Fig. 5b
plt.figure()
for name in np.unique(data[['cathode']]):
    databycathode = data[['cathode','totalPressure']].dropna()
    lPI1 = pidata[['PI1']][databycathode.cathode==name]
    mask = databycathode.cathode==name
    llsq = 10**Yp[mask]
    
    # Get the style
    if name == 'AR3':
        color = 'k'
        marker = '>'
    elif name == 'EK6':
        color = 'tab:olive'
        marker = '<' 
    elif name == 'SC012':
        color = 'tab:cyan'
        marker = 'p' 
    elif name == 'Friedly':
        color = 'tab:pink'
        marker = 'o' 
    elif name == 'JPL-1.5cm' or name == 'JPL-1.5cm-3mm' or name == 'JPL-1.5cm-5mm':
        marker = 'v'
        color = 'tab:cyan'
    elif name == 'NEXIS':
        color = 'tab:orange'
        marker = '^' 
    elif name == 'NSTAR':
        color = 'tab:blue'
        marker = 'o' 
    elif name == 'PLHC':
        color = 'k'
        marker = 'o'
    elif name == 'Salhi-Ar-0.76' or name == 'Salhi-Ar-1.21':
        color = 'tab:red'
        marker = '*' 
    elif name == 'Salhi-Xe':
        color = 'tab:purple'
        marker = 'd' 
    elif name == 'Siegfried':
        color = 'tab:brown'
        marker = 'o' 
    elif name == 'Siegfried-NG':
        color = 'tab:green'
        marker = 'o'
    elif name == 'T6':
        color = 'tab:gray'
        marker = 's' 
    
    plt.loglog(llsq,lPI1,markerfacecolor=color,marker=marker,markeredgecolor='k',linestyle='')
    
plt.legend(["AR3","EK6","Friedly","JPL 1.5cm","","","NEXIS","NSTAR","PLHC","SC012",
            "Salhi-Ar","","Salhi-Xe","Siegfried (Hg)","Siegfried (Ar, Xe)","T6"])
plt.xlabel("$\Gamma (\Pi)$")
plt.ylabel("$\Pi_1$")
plt.title("Figure 5b: Proposed power law (Equation 37) applied to the entire data set")
    
# Perfect correlation
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')


#### Plot the regression w/o error bars or colors 
plt.figure()
plt.loglog(10**Yp,10**Y,'ko',markerfacecolor='none')
plt.legend(["Power law fit"])
plt.xlabel("$\Gamma (\Pi)$")
plt.ylabel("$\Pi_1$")

# Perfect correlation
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')

########################################
######### COEFFICIENT ERROR ############
########################################
### 95% Confidence intervals
mse = mean_squared_error(Y,Yp)
r2 = r2_score(Y,Yp)

# Covariance matrix
Ainv = mse * np.linalg.inv(Xreg.T@Xreg)

# Compute interval
# see, e.g., G. James, et al., "An Introduction to Statistical Learning," 2017. p.66
# or https://stats.stackexchange.com/questions/136157/general-mathematics-for-confidence-interval-in-multiple-linear-regression
vals = 1.96 * np.sqrt(np.diag(Ainv))

### Coefficient error: Table II
print("---------------")
print("Pi-product::Lower bound::Value::Upper bound")
idx = 0
for c, e in zip(coef, vals):
    if idx == 0:
        pi_str = "C"
        idx = 2
    else:
        pi_str = "Pi" + str(idx)
        idx = idx + 1
    
    print(pi_str,"::",f'{c-e:.3}',"::",f'{c:.3}',"::",f'{c+e:.3}')

########################################
########## ERROR HISTOGRAM #############
########################################
### Pressure error
vec_err = np.abs(10**Yp - 10**Y)/10**Y * 100

print("---------------")
print("STATISTICS: R2 AND AVERAGE ERROR")
print("Model\t R^2 \t Average error (%)")
print("Power law:", r2, np.mean(np.abs(10**Yp - 10**Y)/10**Y * 100))
print("---------------")

### Kernel density of pressure error
# Calculate best kernel density bandwidth
bandwidths = 10 ** np.linspace(0, 1, 200)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=5,
                    verbose = 1)
grid.fit(vec_err[:,None])

print('Best params:',grid.best_params_)

# Instantiate and fit the KDE model
print("Instantiate and fit the KDE model")
kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], 
                    kernel='gaussian')
kde.fit(vec_err[:,None])
# Score_samples returns the log of the probability density
x_d = np.linspace(0,100,1000)
logprob = kde.score_samples(x_d[:,None])

# Plot: Figure 6
plt.figure()
plt.plot(x_d,np.exp(logprob),'k-')
plt.fill_between(x_d,np.exp(logprob),color='tab:gray')
_ = plt.hist(vec_err,bins=40,density=True,histtype='step',color='k')

plt.title("Figure 6: Pressure error histogram and KDE")
plt.xlabel("Pressure error (%)")
plt.ylabel("Counts (a.u.)")  

########################################
##### PRINCIPAL COMPONENT ANALYSIS #####
########################################
# Based on PI2 through PI7
# Use log10 to decrease variation (e.g. PI4 ~ 1e14 vs. PI7 ~ 1e0)
X_train = np.array(np.log10(pidata))[:,1::]
pca = PCA(n_components=6)
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)

# Plot: Figure 7
plt.figure()
plt.plot(np.arange(1,7,1),cumsum,'ko')
plt.title("Figure 7: Principal Component Analysis")
plt.xlabel("Dimensions")
plt.ylabel("Explained variance")

plt.show()
