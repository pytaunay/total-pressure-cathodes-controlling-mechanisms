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
File: power-law_limited-parameters.py
Date: July, 2021
Author: Pierre-Yves Taunay

Description: Similar to "power-law_exhaustive-grid-search.py", but we now limit the maximum number 
of parameters that can be used for the power law.

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score


class PowerLawEstimator(BaseEstimator):
    '''
    A class that wraps a linear regression and scoring function for a power law.
    
    Attributes
    ----------        
    mse : double
        Mean squared error of the fit
    n_parameters : int
        Number of parameters used in the linear regression (i.e. number of variables + 1)
    mask : list
        A mask to pick the relevant Pi-products out of the input training data
    m2,m3,m4,m5,m6,m7 : boolean
        Should the given Pi-product be used in the linear regression?
    npi_max : int 
        How many predictors should be used?
    '''
    def __init__(self, m2=True, m3=True, m4=True, m5=True, m6=True, m7=True,npi_max=6):
        super().__init__()
        self.mse_ = -1
        self.n_parameters = 0
        self.mask_ = []
        
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        self.m6 = m6
        self.m7 = m7
        self.npi_max = npi_max
        
    
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
        
        # Reject the case where no Pi-products are chosen (i.e. all mk's are false)
        # Also reject the case where we try to use more parameters than we should
        if X.shape[1] > 0 and X.shape[1] <= self.npi_max:
            Yp = self.reg_.predict(X)
        else:
            Yp = np.zeros(X.shape[0])
        
        return Yp

    def score(self, X,y):
        X, y = check_X_y(X, y, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
                
        nsamples = X.shape[0]

        # No need to apply the mask here, it is applied in the predict(X) method
        # Sanity check
        if X.shape[1] > 0:
            # Predicted y
            ypred = self.predict(X)
            
            # Get MSE
            self.mse_ = mean_squared_error(y,ypred)
            
            # Get score
            aic = 2*self.n_parameters + nsamples * np.log(self.mse_)

        else:
            aic = -1e10    
        return -aic


########################################
############# GET DATA #################
########################################
data = pd.read_hdf("cathode_database.h5",key="data")

### Grab the Pi products
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6','PI7']].dropna()

### Rescaling the Pi products can also be done
lnpidata = np.log10(pidata)

### Training data
Ytrain = lnpidata['PI1']
Xtrain = lnpidata[['PI2','PI3','PI4','PI5','PI6','PI7']]

########################################
### PERFORM EXHAUSTIVE GRID SEARCH #####
########################################

for npi_max in [1,2,3,4]:
    print("================")
    print("Maximum number of Pi products:", npi_max)
    
    ### Set parameters by cross-validation
    ### We try all combinations of Pi-products
    parameters = {
            'm2':[True,False],
            'm3':[True,False],
            'm4':[True,False],
            'm5':[True,False],
            'm6':[True,False],
            'm7':[True,False]
            }

    clf = GridSearchCV(
            PowerLawEstimator(npi_max=npi_max),
            parameters, # Parameters to try
            cv=KFold(n_splits=10,shuffle=True) # K-Fold with 10 folds, with randomized data
            )
    
    clf.fit(Xtrain,Ytrain)
    
    ### Print results
    npi_opt = 0
    for p in clf.best_params_:
        if type(clf.best_params_[p]) == bool:    
            if clf.best_params_[p]:
                npi_opt = npi_opt+1
                
    print("Optimal number of pi-products:", npi_opt)
    print("AIC Score:", clf.best_score_) 
    print("Pi-products to include", clf.best_params_)

    ## Get the mask
    mask = [value for key, value in clf.best_params_.items()]
    
    # Redo linear regression
    reg = LinearRegression()
    X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True)
    reg.fit(X[:,mask],y)
    
    Yp = reg.predict(X[:,mask])
    
    Yp = 10**Yp
    y = 10**y
    print("---------------")
    print("STATISTICS: R2 AND AVERAGE ERROR")
    print("R^2 \t Average error (%)")
    print(r2_score(y,Yp),np.mean( np.abs(Yp-y)/y)*100.)
    print("---------------")
    print("Power law coefficients")
    print("Pi-product","Value")
    coef = np.zeros_like(mask,dtype=np.float64)
    
    idx = 0
    for kk, v in enumerate(clf.best_params_):
        if clf.best_params_[v]:
            coef[kk] = reg.coef_[idx]
            idx = idx + 1
    
    all_coef = np.zeros(len(coef)+1)
    all_coef[0] = reg.intercept_
    all_coef[1:] = np.copy(coef)
    
    for c in all_coef:
        if idx == 0:
            pi_str = "C"
            idx = 2
        else:
            pi_str = "Pi" + str(idx)
            idx = idx + 1
        
        print(pi_str,":",f'{c:.3}')
        
#    plt.loglog(Yp,y,'o')
    