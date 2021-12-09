# MIT License
# 
# Copyright (c) 2021 Pierre-Yves Taunay 
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
'''
File: theoretical-fit_xvalidated-selection.py
Date: August, 2021
Author: Pierre-Yves Taunay

Description: perform a grid-search on all possible combination of Pi-products to find the one
that has the best AIC score. The candidate functions are derived from theory and take the following
form:
    1/4 - log(PI2) + c0 * PI5 + 
    c1 * (1/PI2**2-1) * PI2**c2 * PI3**c3 * PI4**c4 * PI5**c5 * PI6**c6

The grid search is cross-validated with a k-fold approach with 10 folds and multiple initial
conditions that are pre-computed.

'''
import numpy as np
import pandas as pd

from itertools import product

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y
from sklearn.metrics import r2_score


from lmfit import fit_report

from theoretical_model_estimator import TheoreticalModelEstimator

def build_initial_conditions(pidata,nc):
    m2list = [True,False]
    m3list = [True,False]
    m4list = [True,False]
    m5list = [True,False]
    m6list = [True,False]
    
    Ytrain = pidata['PI1']
    Xtrain = pidata[['PI2','PI3','PI4','PI5','PI6']]
    ctest = np.linspace(cmin,cmax,nc)

    
    icdf = pd.DataFrame(columns=['m2','m3','m4','m5','m6','p0','p1','p2','p3','p4','p5','p6'])
    nparam = 7
    print("=========================")
    print("BUILDING INITIAL CONDITIONS")
    for x in product(m2list,m3list,m4list,m5list,m6list):
        mask = list(x)
        
        for z in range(nc):
            rdv = np.zeros(nparam)
       
            ### Grab the C to test            
            Ctheory = ctest[z]
            
            ### Figure out the initial point
            # 1. Compute PI1 - (1/4 - log(PI2) - eta_0 * PI5)
            ylin = Ytrain - (1/4 - np.log(Xtrain['PI2']) + Ctheory * Xtrain['PI5'])
            ylin = np.abs(ylin) / (Xtrain['PI2']**(-2)-1)
    
    
            if(any(ylin < 0.0)):
                raise
            
            # 2. Perform linear regression to get an idea of the initial values
            linreg = LinearRegression()
            Xlin, ylin = check_X_y(Xtrain, ylin, accept_sparse=True,copy=True)
            Xlin = Xlin[:,mask]
    
    
            
            if Xlin.shape[1] > 0:
                linreg.fit(np.log10(Xlin),np.log10(ylin))
       
            # 3. Fill the initial values
            rdv[0] = Ctheory
            if(any(mask)):
                rdv[1] = 10**linreg.intercept_
                
                idxcoeff = 0
                
                for kk,v in enumerate(mask):
                    if v:
                        rdv[kk+2] = linreg.coef_[idxcoeff]
                        idxcoeff = idxcoeff + 1
                    else:
                        rdv[kk+2] = 0.0
            
            if(any(Xtrain['PI2']) < 0.0):
                raise
            
            # Insert into the initial condition dataframe        
            icdf.loc[-1] = [*mask,*rdv]  # adding a row
            icdf.index = icdf.index + 1  # shifting index
            icdf = icdf.sort_index()  # sorting by index
    
    print("DONE.")
    return icdf

########################################
############# GET DATA #################
########################################
data = pd.read_hdf("../../data/cathode_database.h5",key="data")

### Grab the Pi products
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6','PI7']].dropna()

### Rescale the Pi products
lnpidata = np.log10(pidata)

Ytrain = pidata['PI1']
Xtrain = pidata[['PI2','PI3','PI4','PI5','PI6']]


########################################
##### CROSS-VALIDATION PARAMETERS ######
########################################
# Get the scaling factor as computed from theory
scaling_factor = pd.DataFrame()
try:
    scaling_factor = pd.read_csv("cvec.csv")
except:
    print("Exception when reading cvec.csv. Have you run average_theoretical_constant.py?")
    raise

cmin = scaling_factor.min()['scaling_factor']
cmax = scaling_factor.max()['scaling_factor']

nc = 20

icdf = build_initial_conditions(pidata,nc)

print("=========================")
print("CROSS VALIDATION")
parameters = {
        'm2':[True,False],
        'm3':[True,False],
        'm4':[True,False],
        'm5':[True,False],
        'm6':[True,False],
        'icidx': np.arange(nc)
        }


# Grid search
clf = GridSearchCV(
        TheoreticalModelEstimator(icdf),
        parameters,
        n_jobs=4,
        cv=KFold(n_splits=10,shuffle=True),
        verbose=10,
        error_score='raise'
        )

clf.fit(Xtrain,Ytrain)

print("Best AIC score:", clf.best_score_)
print("Best parameters:",clf.best_params_)
# Note: clf.best_estimator.params_ does not correspond to the "best" parameters as determined 
# by LMFIT; rather, it is the last tested set of parameters.

print("=========================")
print("FIT REPORT")
warn_str = "/!\ The error bounds on the variables are not to be trusted! "
warn_str += "LMFIT computes them in an unreliable way for non-linear fits. "
warn_str += "Use the bootstrap method to evaluate the bounds of those parameters instead. /!\ "
print(warn_str)
print(fit_report(clf.best_estimator_.fit_))

########################################
############ OUTPUT INFO ###############
########################################
# Redo regression using the best estimator
X, y = check_X_y(Xtrain,np.log10(Ytrain), accept_sparse=True,copy=True)
Yp = clf.best_estimator_.predict(X)
print("=========================")
print("STATISTICS: R2 AND AVERAGE ERROR")
print("R^2 \t Average error (%)")
print("MSE, R2")
print(r2_score(y,Yp),np.mean(np.abs(10**Yp-10**y)/10**y)*100.)

# Plot
plt.loglog(10**Yp,10**y,'ko',markerfacecolor='none')
plt.xlabel("$\Gamma (\Pi)$")
plt.ylabel("$\Pi_1$")
plt.title("Figure 9b: Theoretically derived expressions applied to the entire experimental dataset.")

onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')

### Gather info about all of the cross-validated results
cvdf = pd.DataFrame(clf.cv_results_)
cvdf = cvdf.sort_values(by=['rank_test_score'])

