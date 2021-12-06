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
File: average_theoretical_constant.py
Date: July, 2021
Author: Pierre-Yves Taunay

Description: calculate the scaling factor obtained from theory. The parameters on which the factor
depends are sampled over a relevant range:
    - Orifice electron temperature: 2-5 eV
    - Orifice ionization fraction: 1% to 10% (0.01 - 0.1)
    - Neutral gas temperature: 2000-4000 K
    - Knudsen number: 0.1-10
The reported value in the article is 2.56 +/- [0.67,0.55].
'''
import numpy as np
import cathode.constants as cc
import matplotlib.pyplot as plt
import pandas as pd

from itertools import product

def static_term(Kn,gam):
    '''
    Calculate the factor that comes from the exit static pressure. 
    Original expression from Santeler, 1986; see Taunay et al., PSST 2021 for the pressure terms
    '''
    theta = 28. * Kn / (28. * Kn + 1)
    
    r = theta * np.sqrt(gam) / np.sqrt(2*np.pi) * ((gam+1)/2)**(1/(gam-1))
    r += (1-theta) * gam
    return 1/r

# Be careful; will generate vectors of size npoints^4 !
npoints = 40
gam = 5./3.

# Parameter space
alpha_o = np.linspace(0.01,0.1,npoints) # Orifice ionization fraction
Te_o = np.linspace(2,5,npoints) # Orifice electron temperature
Tn = np.linspace(2000,4000,npoints) * cc.Kelvin2eV # Neutral static temperature
Kn_o = np.linspace(0.1,10,npoints)

# Function of gamma and KNudsen number
Fgam = static_term(Kn_o,gam)

# Compute the scaling factor
Cvec = np.zeros(npoints**4)
for kk,val in enumerate(
        list(product(Tn,Te_o,alpha_o,Fgam))):
    lTn,lTeo,lalpha,lF = val
    
    Cvec[kk] = np.sqrt(1 + lalpha * lTeo/lTn) + lF
    
# Plot distribution of values
_ = plt.hist(Cvec,bins=50)

# Compute the mean and 5%-95% values
tdf = pd.DataFrame(Cvec).describe(percentiles=[0.05,0.95])

Ctheory = tdf.loc['mean'][0]
Cmin = tdf.loc['min'][0]
Cmax = tdf.loc['max'][0]
print("Lower bound, Theoretical value, Upper bound")
print(Cmin,Ctheory,Cmax)
print("Delta-min, Delta-max")
print(Ctheory-Cmin,Cmax-Ctheory)