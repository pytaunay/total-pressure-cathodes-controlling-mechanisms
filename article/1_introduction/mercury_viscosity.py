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
File: mercury_viscosity.py
Author: Pierre-Yves Taunay
Date: December, 2021
Description: reproduces Figure 1 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lj_transport import transport_properties 

atomic_mass = 1.66053904e-27

Tvec = np.linspace(300,4000)

epskb = 851.0 # K
sig = 2.898 *1e-10 # m
M = 200.59 

mu_lj = transport_properties(sig,epskb,M,Tvec)

plt.semilogy(Tvec,mu_lj*1e4,'k-')
plt.legend(["Lennard-Jones 12-6"])
plt.title("Figure 1: Viscosity of mercury vapor")
plt.xlabel("Temperature (K)")
plt.ylabel("Dynamic viscosity $\mu$ (x $10^4$ Pa-s)")

plt.xlim([300,4000])
plt.ylim([3e-1,4e0])

### Experimental data
# Data from:
# Tables on the thermophysical properties of liquids and gases : in normal and dissociated states 
# / by N.B. Vargaftik ; with a foreword to the English ed. by Y.S. Touloukian.
# p. 152
# Original data from 
# Vukalovich, M. P. and Fokin, R. V., Thermophysical properties of materials of mercury, Standards 
# Press, 1972 
# Khalilov, Kh, M Zhurnal Tekhnicheskoi fiziki 8(13-14), 1249, (1938) 
# Landholt-Bernstein has a table with data from that reference too (D3.1 Liquids and Gases, p. 397). 
# However, these are interpolated based on the data from their original reference! 
vargaftik = np.array([
[200,464e-7],
[220,483e-7],
[240,502e-7],
[260,522e-7],
[280,542e-7],
[300,562e-7],
[320,582e-7],
[340,602e-7],
[360,622e-7],
[380,642e-7],
[400,662e-7],
[420,682e-7],
[440,702e-7],
[460,722e-7],
[480,742e-7],
[500,762e-7],
[550,812e-7],
[600,862e-7],
[650,911e-7],
[700,961e-7],
[800,1057e-7],
])
# Temperature is in degC
plt.plot(vargaftik[:,0]+273.15,vargaftik[:,1]*1e4,'o')

# Data from:
# The viscosity of mercury up to 1520 K and 1000 bar - Tippelskirch et al - 1975
tippelskirch = np.array([
# Isochore 0.7g/cm3
[1310,0.123e-3,10.0],
[1360,0.127e-3,10.0],
[1400,0.133e-3,10.0],
[1430,0.135e-3,10.0],
[1460,0.139e-3,10.0],
[1520,0.137e-3,10.0],
# Isochore 1.05g/cm3
[1359,0.136e-3,10.0],
[1380,0.139e-3,10.0],
[1415,0.139e-3,10.0],
[1430,0.141e-3,10.0],
[1455,0.143e-3,10.0],
[1475,0.146e-3,10.0],
[1530,0.148e-3,10.0],
# Isochore 1.37g/cm3 
[1470,0.172e-3,10.0],
[1515,0.179e-3,10.0],
])
plt.errorbar(tippelskirch[:,0],tippelskirch[:,1]*1e4,yerr=tippelskirch[:,2]/100.0,fmt='o')

# Data from:
# Analysis and Correlation of New Data on the Thermophysical Properties of Mercury Vapor
# A.  I.  Ivanov, V.  E. Lyusternik, and L.  R.  Fokin
# The data plotted here is from multiple original sources. I did not include:
# Koch/Cox 1885, Braune 1928 since both are contained in Epstein's
# Khalilov 1938 since Vargaftik's data contains it 
# Tippelskirch 1975 since it is standalone
ivanov = np.array([
[422.42,720.38e-7],
[527.55,780.48e-7],
[610.16,941.12e-7],
[686.92,1020.1e-7]
])
# Temperature is in degC
plt.plot(ivanov[:,0]+273.15,ivanov[:,1]*1e4,'o')

# Data from:
# V.N. Popov "Thermal Properties of Mercury on the Basis of Model Potentials" High Temperature, 
# 2012, Vol. 50, No. 6, pp. 700-707. 
# The references cited are a 2010 conference paper from Fokin and a 2011 study by Fokin et al. 
# The latter mentions an experimental database of viscosity and conductivity but does not cite it... 
# Closest I can find is: GSSSD (State Service of Standard Reference Data) 57-83: Mercury. 
# The Viscosity, Thermal Conductivity, Self-Diffusion, and the Second Virial Coefficient in the 
# Temperature Range 400-2000 K at Low Pressures in the Gaseous State, Fokin L.R., and 
# Lyusternik V.E., Eds., Moscow: Izd. Standartov, 1985, p. 16.
popov = np.array([
[402.22,38.729e-6],
[498.7,46.716e-6],
[550.38,48.951e-6],
[553.83,53.026e-6],
[583.69,53.033e-6],
[594.03,55.073e-6],
[628.48,59.34e-6],
[649.16,63.234e-6],
[698.55,68.617e-6],
[715.77,69.917e-6],
[813.4,85.312e-6],
[835.22,88.095e-6],
[843.26,83.097e-6],
[869.68,86.807e-6],
[890.35,88.109e-6],
[892.65,92.739e-6],
[958.12,105.348e-6],
[969.6,100.906e-6],
[1004.06,101.47e-6],
[1050,108.519e-6],
[1060.34,110.558e-6],
[1507.12,151.78e-6],
])
plt.plot(popov[:,0],popov[:,1]*1e4,'o')

# Data from:
epstein_braune = np.array([
[218.0,4709.,0.03],
[219.5,4672.,0.03],
[223.5,4689.,0.03],
[281.0,5310.,0.03],
[300.0,5501.,0.03],
[330.0,5831.,0.03],
[421.0,6856.,0.03],
[439.5,7029.,0.03],
[496.0,7610.,0.03],
[565.0,8343.,0.03],
[588.5,8632.,0.03],
[607.0,8766.,0.03],
[610.0,8802.,0.03],
])
plt.errorbar(epstein_braune[:,0]+273.15,epstein_braune[:,1]*1e-8*1e4,
        yerr=epstein_braune[:,2],fmt='o')

# The error is obtained from Ivanov et al., Table 1 ("Assumed error")
epstein_cox = np.array([
[273.0,4940.,0.05],
[301.0,5320.,0.05],
[352.0,6078.,0.05],
[380.0,6540.,0.05],
])
plt.errorbar(epstein_cox[:,0]+273.15,epstein_cox[:,1]*1e-8*1e4,
        yerr=epstein_cox[:,2],fmt='o')

plt.show()
