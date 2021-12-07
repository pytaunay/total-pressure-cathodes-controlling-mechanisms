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
File: theory_fit_bootstrap.py
Date: July, 2021
Author: Pierre-Yves Taunay
Description: Performs a bootstrap analysis to find the 95% confidence intervals for:
    1. each predicted value, and
    2. each coefficient.
Reproduces Table V in the associated publication.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import choices

# Sklearn
from sklearn.utils.validation import check_X_y
from sklearn.metrics import mean_squared_error, r2_score

from theoretical-fit_xvalidated-selection import TheoryModelEstimator 

########################################
############# GET DATA #################
########################################
data = pd.read_hdf("cathode_database.h5",key="data")

### Grab the Pi products
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6']].dropna()

### Define training data
Ytrain = pidata['PI1']
Xtrain = pidata[['PI2','PI3','PI4','PI5','PI6']]


########################################
##### CROSS-VALIDATION PARAMETERS ######
########################################
icdf_single = pd.DataFrame(columns=['m2','m3','m4','m5','m6','p0','p1','p2','p3','p4','p5','p6'])
mask = [True,True,True,False,True]


# Initial values as determined from the "best" fit
rdv = np.array([])
try:
    # Note: must run theoretical-fit_xvalidated-selection.py to obtain those values
    parameters = clf.best_estimator_.fit_.params
    rdv = np.array([parameters['a0'].value,
                    parameters['a1'].value,
                    parameters['a2'].value,
                    parameters['a3'].value,
                    parameters['a4'].value,
                    0.0, # No PI5
                    parameters['a6'].value
                    ])
except:
    # If the script has not run, then try those values that we obtained
    rdv = np.array([2.45386892,
                    2.8587e12 ,
                    3.56248969,
                    -1.01288268,
                    -0.20601033,
                    0.0, # No PI5
                    0.79316430])

icdf_single.loc[-1] = [*mask,*rdv]  # adding a row
icdf_single.index = icdf_single.index + 1  # shifting index
icdf_single = icdf_single.sort_index()  # sorting by index
        
main_reg = TheoryModelEstimator(icdf_single,*mask,icidx=0)

X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True)
main_reg.fit(Xtrain,Ytrain)


main_coeff = np.array([
        main_reg.fit_.params['a0'].value,
        main_reg.fit_.params['a1'].value,
        main_reg.fit_.params['a2'].value,
        main_reg.fit_.params['a3'].value,
        main_reg.fit_.params['a4'].value,
        main_reg.fit_.params['a6'].value,
        ])

# Predicted values of Y
Yp_main = main_reg.predict(Xtrain)


#########################################
############## BOOTSTRAP ################
#########################################
nsamples = len(Ytrain)
ntry = 10000

# Placeholders for the coefficients to be found and the predicted y values
all_coeff = np.zeros((ntry,len(main_coeff)))

# Same, but for all errors
r2_all = np.zeros(ntry)
mse_all = np.zeros_like(r2_all)
ave_err_all = np.zeros_like(r2_all)

# Redefine the regression
reg = TheoryModelEstimator(icdf_single,*mask,icidx=0)


# Dataframe to store each result
df = pd.DataFrame(columns=range(len(Ytrain)))

Yoriginal = np.copy(Ytrain)

# Do bootstrap
ldic = []
for k in range(ntry):
    # Random choices with replacement for the number of samples in the dataset
    randomized = np.array(
            choices(np.array(pidata),k=nsamples))
    
    # Extract training data
    Xtrain = randomized[:,1:]
    Ytrain = randomized[:,0]
 
    # Perform fit
    reg.fit(Xtrain,Ytrain)
    
    # Store coefficients
    all_coeff[k,:] = np.array([
        reg.fit_.params['a0'].value,
        reg.fit_.params['a1'].value,
        reg.fit_.params['a2'].value,
        reg.fit_.params['a3'].value,
        reg.fit_.params['a4'].value,
        reg.fit_.params['a6'].value,
        ])

    # Predict y and get R2, MSE, average error
    Yp = reg.predict(Xtrain)
    
    Ytrue = np.log10(Ytrain)
    r2_all[k] = r2_score(Ytrue,Yp)
    mse_all[k] = mean_squared_error(Ytrue,Yp)
    ave_err_all[k] = np.mean(100.0*np.abs( (Ytrue-Yp)/Ytrue))
    
    if k%100 == 0:
        print(k)
    
    # Store each y prediction in list of dictionaries
    data_dict = {}
    for ii,yy in enumerate(Ytrain):
        idx = np.where(Yoriginal==yy)[0][0]
        
        data_dict[idx] = 10**Yp[ii]
        
    ldic.append(data_dict)

# Create a dataframe out of the list of dictionaries
df = pd.DataFrame(ldic,columns=range(len(Ytrain)))

### Plot data with bootstrapped error bounds on the Y prediction
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')

desc = df.describe(percentiles=[0.05,0.95])
sdesc = desc.sort_values(by='mean',axis=1)

plt.errorbar(desc.loc['mean'],
             pidata['PI1'],
                  xerr =np.array([
                          desc.loc['mean']-desc.loc['5%'],
                          desc.loc['95%']-desc.loc['mean']]),fmt='ko')

### Get the bootstraped error for the coefficients
df = pd.DataFrame(all_coeff)

# Standard error
se_arr = np.zeros(len(main_coeff))
for k in range(ntry):
    ser = all_coeff[k,:] - 1/ntry * np.sum(all_coeff,axis=0)
    se_arr += ser**2
se_arr /= ntry-1
se_arr = np.sqrt(se_arr)

print(se_arr)
print(main_coeff + se_arr)
print(main_coeff - se_arr)

