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
File: .py
Author: Pierre-Yves Taunay
Date: December, 2021
Description: performs the Backward stepwise selection with randomized Pi-products

Recreates Table III, "Total pressure in thermionic orificed hollow cathodes"
'''
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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

########################################
####### RANDOMIZED REGRESSION ##########
########################################
### Iteration 1
rand_results = np.zeros((6,2))

NITER=1000
for niter in range(NITER):    
    for idx in np.arange(1,7):        
        Xlsq = np.array([X0,X1,X2,X3,X4,X5,X6])

        
        Xidx = Xlsq[idx,:]
        np.random.shuffle(Xidx)
        Xlsq[idx,:] = np.copy(Xidx)
        Xlsq = Xlsq.T
        
        reg = LinearRegression()
        reg.fit(Xlsq,Y)
        Yp = reg.predict(Xlsq) # Predicted Y        
        
        R2 = r2_score(Y,Yp)
        
        ## Average error
        # Least squares
        P_xp = np.array(data[['totalPressure_SI']].dropna())[:,0]
        P_model = data[['totalPressure_SI','magneticPressure']].dropna()
        P_model = np.array(P_model[['magneticPressure']])[:,0]
        
        P_model *= 10**Yp
        
        vec_err =  np.abs((P_xp-P_model)/P_xp)* 100
        
        ave_err = np.average(vec_err) 
        
        rand_results[idx-1,0] += R2
        rand_results[idx-1,1] += ave_err
    
# Recreate 'Iteration 1' in Table III
print("ITERATION 1")
print("Reference",0.98,22.7) 
print("Perturbed Pi-product,R^2,Average error (%)")
for idx in np.arange(1,7):
    print(idx+1,f'{rand_results[idx-1,0]/NITER:.3}',f'{rand_results[idx-1,1]/NITER:.3}')    

#### Iteration 2: REMOVE PI3 AND PI6
print("-----")
print("Remove PI3 and PI6")
X = (np.array(pidata[['PI2','PI4','PI5','PI7']]))
X0 = np.ones(len(Y))
X1 = np.log10(X[:,0]) # PI2
X3 = np.log10(X[:,1]) # PI4 
X4 = np.log10(X[:,2]) # PI5 
X6 = np.log10(X[:,3]) # PI7

# REFERENCE
Xlsq = np.array([X0,X1,X3,X4,X6]).T

reg = LinearRegression()
reg.fit(Xlsq,Y)
Yp = reg.predict(Xlsq) # Predicted Y     

P_xp = np.array(data[['totalPressure_SI']].dropna())[:,0]
P_model = data[['totalPressure_SI','magneticPressure']].dropna()
P_model = np.array(P_model[['magneticPressure']])[:,0]

P_model *= 10**Yp

vec_err =  np.abs((P_xp-P_model)/P_xp)* 100
print("ITERATION 2")
print("New R-squared and average error (%)")
print("Reference",f'{r2_score(Y,Yp):.3}',f'{np.mean(vec_err):.3}')

# Iteration 2
rand_results = np.zeros((4,2))
NITER=1000
for niter in range(NITER):    
    for idx in np.arange(1,5):     
        Xlsq = np.array([X0,X1,X3,X4,X6]) 

        Xidx = Xlsq[idx,:]
        np.random.shuffle(Xidx)
        Xlsq[idx,:] = np.copy(Xidx)
        Xlsq = Xlsq.T
        
        reg = LinearRegression()
        reg.fit(Xlsq,Y)
        Yp = reg.predict(Xlsq) # Predicted Y        
        
        R2 = r2_score(Y,Yp)
        
        ## Average error
        # Least squares
        P_xp = np.array(data[['totalPressure_SI']].dropna())[:,0]
        P_model = data[['totalPressure_SI','magneticPressure']].dropna()
        P_model = np.array(P_model[['magneticPressure']])[:,0]
        
        P_model *= 10**Yp
        
        vec_err =  np.abs((P_xp-P_model)/P_xp)* 100
        
        ave_err = np.average(vec_err) 
        
        rand_results[idx-1,0] += R2
        rand_results[idx-1,1] += ave_err#for niter in range(NITER):     


pilist = ["PI2","PI4","PI5","PI7"]
print("Perturbed Pi-product,R^2,Average error (%)")
for idx in np.arange(0,4):
    print(pilist[idx],f'{rand_results[idx,0]/NITER:.3}',f'{rand_results[idx,1]/NITER:.3}')
