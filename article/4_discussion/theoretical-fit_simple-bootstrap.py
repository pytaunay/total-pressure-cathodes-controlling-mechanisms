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
File: theoretical-fit_simple-bootstrap.py
Date: July, 2021
Author: Pierre-Yves Taunay
Description: Performs a bootstrap analysis to find the 95% confidence intervals for the constant
that appears in the fit PI1 = C * PI5. The values reported in the associated publication are
4.03 +/- 0.22. The values obtained with this script may differ a little because of the 
random numbers.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import choices

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y
from sklearn.metrics import mean_squared_error, r2_score

########################################
############# GET DATA #################
########################################
data = pd.read_hdf("../../data/cathode_database.h5",key="data")

### Grab the Pi products
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6']].dropna()

### Define training data
Ytrain = pidata['PI1']
Xtrain = pidata['PI5']

Xtrain = np.array(Xtrain).reshape(-1,1)
Ytrain = np.array(Ytrain).reshape(-1,1)

########################################
##### CROSS-VALIDATION PARAMETERS ######
########################################
# Linear regression with all data
main_reg = LinearRegression(fit_intercept=False)

X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True)
main_reg.fit(X,y)

main_coeff = [*main_reg.coef_]
main_coeff = np.array(main_coeff)

# Predicted values of Y
Yp_main = main_reg.predict(Xtrain)
Y_main = np.copy(Ytrain)
X_main = np.copy(Xtrain)

########################################
############# BOOTSTRAP ################
########################################
nsamples = len(Ytrain)
ntry = 50000

# Placeholders for the coefficients to be found and the predicted y values
all_coeff = np.zeros((ntry,len(main_coeff)))

# Same, but for all errors
r2_all = np.zeros(ntry)
mse_all = np.zeros_like(r2_all)
ave_err_all = np.zeros_like(r2_all)

# Redefine the regression
reg = LinearRegression(fit_intercept=False)

# Dataframe to store each result
df = pd.DataFrame(columns=range(len(Ytrain)))

Yoriginal = np.copy(Ytrain)

# Do bootstrap
ldic = []
for k in range(ntry):
    # Random choices with replacement for the number of samples in the dataset
    randomized = np.array(
            choices(np.array(pidata[['PI1','PI5']]),k=nsamples))
    
    # Extract training data
    Xtrain = randomized[:,1:]
    Ytrain = randomized[:,0]
 
    # Perform fit
    reg.fit(Xtrain,Ytrain)
    
    # Store coefficients
    c = [*reg.coef_]
    all_coeff[k,:] = np.array(c)

    # Predict y and get R2, MSE, average error
    Yp = reg.predict(Xtrain)
    
    Ytrue = np.copy(Ytrain)
    r2_all[k] = r2_score(Ytrue,Yp)
    mse_all[k] = mean_squared_error(Ytrue,Yp)
    ave_err_all[k] = np.mean(100.0*np.abs( (Ytrue-Yp)/Ytrue))
    
    if k%1000 == 0:
        print(k)
    
    # Store each y prediction in list of dictionaries
    data_dict = {}
    for ii,yy in enumerate(Ytrain):
        idx = np.where(Yoriginal==yy)[0][0]
        
        data_dict[idx] = Yp[ii]
        
    ldic.append(data_dict)

# Create a dataframe out of the list of dictionaries
df = pd.DataFrame(ldic,columns=range(len(Ytrain)))

########################################
############ OUTPUT INFO ###############
########################################
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
desc = df.describe(percentiles=[0.05,0.95])

print("---------------")
print("Boostrapped coefficient C in PI1 = C PI5")
print("Lower bound::Value::Upper bound::Min error::Max error")
Cexp = desc.loc['mean'][0]
Cmin = desc.loc['5%'][0]
Cmax = desc.loc['95%'][0]
print(f'{Cmin:.3}',f'{Cexp:.3}',f'{Cmax:.3}',f'{Cexp-Cmin:.3}',f'{Cmax-Cexp:.3}')

print("---------------")
print("STATISTICS: R2 AND AVERAGE ERROR")
print("R^2 \t Average error (%)")
print(np.mean(r2_all),np.mean(ave_err_all))

plt.show()
