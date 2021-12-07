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
File: empirical_fits.py
Author: Pierre-Yves Taunay
Date: August, 2021
Description: compares empirical models by
    - D. E. Siegfried and P. J. Wilbur, "A model for mercury orificed hollow cathodes: theory and
    experiment," AIAA Journal, 22, 10 (1984).
    - M. Capacci et al, "Simple numerical model describing discharge parameters in orificed hollow 
    cathode devices," 33rd AIAA/ASME/SAE/ASEE Joint Propulsion Conference & Exhibit (1997) 
    AIAA-1997-2791
    - S. W. Patterson and D. G. Fearn, "The generation of high energy ions in hollow cathode
    discharge," 26th International Electric Propulsion Conference (1999) IEPC-1999-125.
    
This generates Fig. 3 in "Total pressure in thermionic orificed hollow cathodes"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_X_y
from sklearn.metrics import mean_squared_error, r2_score

from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr

from lmfit import Model, Parameters

########################################
############# GET DATA #################
########################################
data = pd.read_hdf("../../../data/cathode_database.h5",key="data")

### Grab the Pi products
sw_data = data[['PI1','totalPressure_SI','massFlowRate','massFlowRate_SI','gasMass','orificeDiameter','dischargeCurrent','magneticPressure']].dropna()


########################################
#### FIT SIEGFRIED AND WILBUR FORM #####
########################################
### Build the functional form
# The proposed fit in Torr by Siegfried and Wilbur is 
# P = x0 / x1^2 * (a0 + a1 * x2)
# => PI1 = P/Pmag = x0 / x1^2 * (a0 + a1 * x2) / x3
# x0: mass flow rate in eqA
# x1: orifice diameter in mm
# x2: discharge current
# x3: magnetic pressure in Pa
fstr_sw = 'log10( 101325./760. * x0 / x1**2 * (a0 + a1*x2)/x3)' 

### Declare coefficients and variables
lambcoeff = ['a0','a1']
lambvar = ['x0','x1','x2','x3']     

lamb = lambvar + lambcoeff

# Parse the string to create a callable function          
func_sw = lambdify(lamb,parse_expr(fstr_sw),modules='numpy')

def fun_sw(x0,x1,x2,x3,a0,a1):  
    return func_sw(x0,x1,x2,x3,a0,a1)

### Create fit parameters
params = Parameters()
# Initial parameters from Siegfried and Wilbur for mercury
# See, e.g., Siegfried, "A phenomenological model for orificed hollow cathodes," PhD thesis,
# 1982, p.43
params.add('a0', value=13.7,min=0)
params.add('a1', value=7.8,min=0)

### Create model
# The model can take invalid values for log10 (e.g. a0 = a1 = 0) in which case NaN are generated.
# Use "propagate" parameters for that case to avoid exceptions.
sw_fit_model = Model(fun_sw, independent_vars=lambvar,
                  param_names=lambcoeff, nan_policy='propagate')

### Extract data
# X = [x0,x1,x2,x3] 
Xtrain = sw_data[['massFlowRate','orificeDiameter','dischargeCurrent','magneticPressure']]
Ytrain = sw_data['PI1']
X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True,copy=True)

# Fit is in log space
y = np.log10(y)

### Perform fit
sw_fit = sw_fit_model.fit(y,params=params,
                            x0=X[:,0],
                            x1=X[:,1],
                            x2=X[:,2],
                            x3=X[:,3],
                            method='least_squares')

Yp = fun_sw(X[:,0],X[:,1],X[:,2],X[:,3],
            sw_fit.params['a0'].value,
            sw_fit.params['a1'].value)

### Output fit info and plot
print("=============S and W fit =============")
print("MSE, R2")
print(mean_squared_error(y,Yp),r2_score(y,Yp))
print("Average error (%):" ,np.mean( np.abs((10**Yp-10**y)/10**y)*100))
print("Parameters")
print(sw_fit.params)

plt.loglog(10**Yp,10**y,'ko',markerfacecolor='none')

#########################################
############# FIT CAPACCI ###############
#########################################
# The proposed fit in Torr by Capacci et al. is 
# P = x0 / x1^2 * (a0 + a1 * x2 + a2 * x2^2)
# => PI1 = P/Pmag = x0 / x1^2 * (a0 + a1 * x2 + a2^2) / x3
# x0: mass flow rate in eqA
# x1: orifice diameter in mm
# x2: discharge current
# x3: magnetic pressure in Pa
# Note: there is a missing division by the orifice diameter in the original manuscript of 
# Capacci et al. (Eqn. 1 has mdot multiplied by d0^2). 

fstr_cap = 'log10( 101325./760. * x0 / x1**2 * (a0 + a1*x2 + a2*x2**2) / x3)' 

lambcoeff = ['a0','a1','a2']
lambvar = ['x0','x1','x2','x3']     

lamb = lambvar + lambcoeff

# Parse the string to create a callable function          
func_cap = lambdify(lamb,parse_expr(fstr_cap),modules='numpy')

def fun_cap(x0,x1,x2,x3,a0,a1,a2):  
    return func_cap(x0,x1,x2,x3,a0,a1,a2)

### Create fit parameters
params = Parameters()
# Initial parameters assigned far away from 0; No info about the values
params.add('a0', value=1e10,min=0)
params.add('a1', value=1e10,min=0)
params.add('a2', value=1e10,min=0)

### Create model
cap_fit_model = Model(fun_cap, independent_vars=lambvar,
                  param_names=lambcoeff, nan_policy='propagate')

### Extract data
# X = [x0,x1,x2,x3] 
Xtrain = sw_data[['massFlowRate','orificeDiameter','dischargeCurrent','magneticPressure']]
Ytrain = sw_data['PI1']
X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True,copy=True)
# Fit is in log space
y = np.log10(y)

cap_fit = cap_fit_model.fit(y,params=params,
                            x0=X[:,0],
                            x1=X[:,1],
                            x2=X[:,2],
                            x3=X[:,3],
                            method='least_squares')


Yp = fun_cap(X[:,0],X[:,1],X[:,2],X[:,3],
             cap_fit.params['a0'].value,
             cap_fit.params['a1'].value,
             cap_fit.params['a2'].value)

### Output fit info and plot
print("=============Capacci fit =============")
print("MSE, R2")
print(mean_squared_error(y,Yp),r2_score(y,Yp))
print("Average error (%):" , np.mean( np.abs((10**Yp-10**y)/10**y)*100))
print("Parameters")
print(cap_fit.params)

plt.loglog(10**Yp,10**y,'kx')


########################################
########### FIT PATTERSON ##############
########################################
### Build the functional form
# The proposed fit in by Patterson and Fearn is
# P = a0 + a1 mdot + a2 mdot^2 + a3 mdot Id + a4 Id + a5 Id^2
# => PI1 = P/Pmag = 1/x2 * (a0 + a1 x0 + a2 x0^2 + a3 x0 x1 + a4 x1 + a5 x2^2)
# x0: mass flow rate in mg/s
# x1: discharge current
# x2: magnetic pressure in Pa
# Note: we converted the mass flow rate from kg/s to mg/s to have similar order of magnitudes
# for both the mass flow rate and the discharge current and avoid ill-conditioned fits.
fstr_pat = 'log10((1/x2 * (a0 + a1 * x0*1e6 + a2*(x0*1e6)**2 + a3*(x0*1e6)*x1 + a4*x1 + a5*x1**2) ))'


lambcoeff = ['a0','a1','a2','a3','a4','a5']
lambvar = ['x0','x1','x2']     

lamb = lambvar + lambcoeff

# Parse the string to create a callable function          
func_pat = lambdify(lamb,parse_expr(fstr_pat),modules='numpy')
def fun_pat(x0,x1,x2,a0,a1,a2,a3,a4,a5):  
    r = func_pat(x0,x1,x2,a0,a1,a2,a3,a4,a5)
#    return r
    if r.all() > 0.0:
        return r
    else:
        return np.zeros_like(r)

### Create fit parameters
params = Parameters()
# Initial parameters assigned at 0, except for the discharge current square
params.add('a0', value=0)
params.add('a1', value=0)
params.add('a2', value=0)
params.add('a3', value=0)
params.add('a4', value=0)
params.add('a5', value=1e-6)

### Create model
pat_fit_model = Model(fun_pat, independent_vars=lambvar,
                  param_names=lambcoeff, nan_policy='propagate')

### Extract data
# X = [x0,x1,x2,x3] 
Xtrain = sw_data[['massFlowRate_SI','dischargeCurrent','magneticPressure']]
Ytrain = sw_data['PI1']

X, y = check_X_y(Xtrain, Ytrain, accept_sparse=True,copy=True)

# Fit is in log space
y = np.log10(y)

pat_fit = pat_fit_model.fit(y,
                            params=params,
                            x0=X[:,0],
                            x1=X[:,1],
                            x2=X[:,2],
                            method='least_squares')

Yp = fun_pat(X[:,0],X[:,1],X[:,2],
             pat_fit.params['a0'].value,
             pat_fit.params['a1'].value,
             pat_fit.params['a2'].value,
             pat_fit.params['a3'].value,
             pat_fit.params['a4'].value,
             pat_fit.params['a5'].value)

print("=============Patterson fit =============")
print("MSE, R2")
print(mean_squared_error(y,Yp),r2_score(y,Yp))
print("Average error (%):" , np.mean( np.abs((10**Yp-10**y)/10**y)*100))

print("Parameters")
print(pat_fit.params)

plt.loglog(10**Yp,10**y,'k^')


########################################
############## PLOT INFO ###############
########################################
plt.legend(["Siegfried and Wilbur",
            "Capacci et al",
            "Patterson and Fearn"])
plt.xlabel("$\Gamma (\Pi)$")
plt.ylabel("$\Pi_1$")

# Perfect correlation
onetone = np.logspace(0,5,100)
plt.loglog(onetone,onetone,'k--')

plt.show()
