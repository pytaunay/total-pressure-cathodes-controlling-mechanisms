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
File: theoretical_fits.py
Author: Pierre-Yves Taunay
Date: August, 2021
Description: evaluates the Poiseuille flow and isentropic flow models 
    
This generates Fig. 4 in "Total pressure in thermionic orificed hollow cathodes"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

data = pd.read_hdf("cathode_database.h5",key="data")
pidata = data[['PI1','PI2','PI3','PI4','PI5','PI6','PI7']].dropna()

gam = 5./3.

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

R2 = []
ave_err = []
model_list = ["Poiseuille","Isentropic"]

### Poiseuille
# Pi coefficents
C_poiseuille = 4/np.sqrt(gam)
b_vec_poiseuille = np.array([np.log10(C_poiseuille),
                             0.0,-0.5,0.0,1.0,0.0,-0.5])

# Predicted Y
Yp = np.log10(C_poiseuille) + np.sum(np.log10(X**b_vec_poiseuille[1:]) , axis =1)

# Error metrics
R2.append(r2_score(10**Y,10**Yp))
err = np.mean( np.abs((10**Yp-10**Y)/10**Y)*100)
ave_err.append(err)

# Plot
plt.loglog(10**Yp,10**Y,'kd')


### Isentropic
# Pi coefficents
C_iso = 1/gam * ((gam+1)/2)**((gam)/(gam-1))
b_vec_iso = np.array([np.log10(C_iso),0.0,0.0,0.0,1.0,0.0,0.0])

# Error metrics
Yp = np.log10(C_iso) + np.sum(np.log10(X**b_vec_iso[1:]) , axis =1)
R2.append(r2_score(10**Y,10**Yp))
err = np.mean( np.abs((10**Yp-10**Y)/10**Y)*100)
ave_err.append(err)

# Plot
plt.loglog(10**Yp,10**Y,'k^',markerfacecolor='none')


print("---------------")
print("STATISTICS: R2 AND AVERAGE ERROR")
print("Model\t R^2 \t Average error (%)")
for r,e,m in zip(R2,ave_err,model_list):
    print(m,r,e)

    
########################################
############## PLOT INFO ###############
########################################
plt.legend(["Poiseuille","Isentropic"])
plt.xlabel("$\Gamma (\Pi)$")
plt.ylabel("$\Pi_1$")

# Perfect correlation
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')