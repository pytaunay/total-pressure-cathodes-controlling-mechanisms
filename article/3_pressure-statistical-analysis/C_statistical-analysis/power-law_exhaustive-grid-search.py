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
'''
File: power-law_exhaustive-grid-search.py
Date: August, 2021
Author: Pierre-Yves Taunay

Deescription: Performs an exhaustive grid search to find the Pi products find that do not matter.
Each regression is scored using the AIC criteration. See Section III.C.3 in the associated paper.
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score

class ExhaustiveGridSearchEstimator(BaseEstimator):
    def __init__(self, m2=True, m3=True, m4=True, m5=True, m6=True, m7=True):
        super().__init__()
        self.params_ = None
        self.mse_ = -1
        self.n_parameters = 0
        self.mask_ = []
        
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        self.m6 = m6
        self.m7 = m7
        
    
    def fit(self, X, y):
        # Check X and Y
        X, y = check_X_y(X, y, accept_sparse=True)
        # Apply mask on X
        self.mask_ = [self.m2,self.m3,self.m4,self.m5,self.m6,self.m7]
        X = X[:,self.mask_]
        
        # Count number of parameters. Add 1 for the y intercept
        self.n_parameters = sum(self.mask_) + 1        

        # Perform linear regression
        self.reg_ = LinearRegression()
        if X.shape[1] > 0:
            self.reg_.fit(X,y)
    
        self.is_fitted_ = True
        return self        
  
    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        X = X[:,self.mask_]
        if X.shape[1] > 0 and X.shape[1] < 4:
            Yp = self.reg_.predict(X)
        else:
            Yp = np.zeros(X.shape[0])
        
        return Yp

    def score(self, X,y):
        X, y = check_X_y(X, y, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
                
        nsamples = X.shape[0]

        # No need to apply the mask here, it is applied in the predict(X) method         
        if X.shape[1] > 0:
            # Predicted y
            ypred = self.predict(X)
            
            self.mse_ = mean_squared_error(y,ypred)
            
            # AIC score            
            aic = 2*self.n_parameters + nsamples * np.log(self.mse_)            
            
        else:
            aic = -1e10
        
        return -aic

########################################
############# GET DATA #################
########################################
data = pd.read_hdf("../../../data/cathode_database.h5",key="data")

### Grab the Pi products
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6','PI7']].dropna()
lnpidata = np.log10(pidata)

Ytrain = lnpidata['PI1']
Xtrain = lnpidata[['PI2','PI3','PI4','PI5','PI6','PI7']]


########################################
######## SETUP GRID SEARCH  ############
########################################
### Set parameters by cross-validation
# Each Pi-product may / may not be included
parameters = {
        'm2':[True,False],
        'm3':[True,False],
        'm4':[True,False],
        'm5':[True,False],
        'm6':[True,False],
        'm7':[True,False]
        }

# Include k-fold with k = 10
clf = GridSearchCV(
        ExhaustiveGridSearchEstimator(),
        parameters,
        cv=KFold(n_splits=10,shuffle=True)
        )

########################################
####### PERFORM GRID SEARCH  ###########
########################################
clf.fit(Xtrain,Ytrain)

########################################
########### OUTPUT INFO  ###############
########################################
print("---------------")
print("Score and parameters")
print(clf.best_score_,clf.best_params_)

# Get the mask
mask = [value for key, value in clf.best_params_.items()]

# Redo linear regression with the best mask
reg = LinearRegression()
X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True)
reg.fit(X[:,mask],y)

Yp = reg.predict(X[:,mask])

### Output statistics
print("---------------")
print("Best MSE, R2")
print(mean_squared_error(y,Yp),r2_score(y,Yp))
print("---------------")
print("STATISTICS: R2 AND AVERAGE ERROR")
print("R^2 \t Average error (%)")
print(r2_score(y,Yp), np.mean(np.abs(10**Yp - 10**y)/10**y * 100))

### Plot correlation
plt.figure()
plt.loglog(10**Yp,10**y,'ko')
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')
plt.show()

print("---------------")
results_df = pd.DataFrame(clf.cv_results_)
results_df = results_df.sort_values(by=['rank_test_score'])
print(results_df[['rank_test_score','mean_test_score','std_test_score']])
