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
File: feed-system_loss-estimate.py
Date: December, 2021
Author: Pierre-Yves Taunay

Description: approximates the feed-system loss as a percentage of the measured upstream pressure.
The corresponding statement in the paper is in Section II.A, last paragraph: "the feed-system loss 
is estimated to be, on average, 3%"
Our calculation assumes: 
    - a Poiseuille flow model upstream (away from) the cathode active zone,
    - a gas temperature of 1000 K (worst case scenario since it increases viscosity).
The Poiseuille flow model is from D. Goebel and I. Katz, "Fundamentals of Electric Propulsion", 2008.
The viscosity is computed in Poise using 
    - for Xe: the fit proposed by Goebel and Katz
    - for Ar: Stiel and Thodos (1961)
'''
from scipy.optimize import root
import pandas as pd

from flow_model import poiseuille_flow

import numpy as np

atomic_mass = 1.66053904e-27
kB = 1.38064852e-23

def dynamic_viscosity(TnK,species):
    '''
    Computes the viscosity of either Xe or Ar, in Poise.
    The viscosity is given by 
        - for Xe: D. Goebel and I. Katz, "Fundamentals of electric propulsion," 2008, p.465
        - for Ar: L. I. Stiel and G. Thodos, "The viscosity of nonpolar gases at normal pressures," 1961, Eqn. 10
    '''
    mu = np.nan
    if species == 'Xe':
        Tc = 289.7 # See Table 1 in Stiel and Thodos
        Tr = TnK/Tc
        mu = 2.3e-4*Tr**(0.71+0.29/Tr)
    elif species == 'Ar':
        Tc = 151.2 # See Table 1 in Stiel and Thodos
        Tr = TnK/Tc
        xi = 0.0276
        mu = 17.78e-5*(4.58*Tr - 1.67)**(5./8.)
        mu /= xi
        mu /= 100.0 # Convert from centipoise to Poise
    else:
        raise("Species should be either Xe or Ar")

    return mu

def poiseuille_flow(length,radius,flow_rate_SI,TnK,P_out,species):
    '''
    Computes the typical Poiseuille flow for a compressible gas, assuming isothermal gas at TnK.
    Inputs:
        - length: Channel length (m)
        - radius: Channel radius (m)
        - flow_rate_SI: Flow rate in SI units (kg/s)
        - TnK: Gas temperature (K)
        - P_out: Outlet pressure (Pa)
        - species: the species of interest, either "Xe" or "Ar"
    '''
    M = np.nan 
    if species == 'Xe':
        M = 131.293
    elif species == 'Ar':
        M = 39.948

    M *= atomic_mass 
    Rg = kB/M # Gas constant

    ret = dynamic_viscosity(TnK,species) / 10.0 # Convert from Poise to Pa-s 
    ret *= flow_rate_SI
    ret *= TnK
    ret *= length/radius**4
    ret *= 16.0/np.pi
    ret *= Rg 

    ret += P_out**2

    ret = np.sqrt(ret)

    return ret 

### Grab data, but not Domonkos cases (since feed-system loss were estimated in the disseration already)
data = pd.read_hdf("../../data/cathode_database.h5",key="data")
local_data = data.query('upstreamPressurePoint < 300')

### Define variables 
Lup = local_data['upstreamPressurePoint']*1e-3
dc = local_data['insertDiameter']*1e-3
mdot = local_data['massFlowRate_SI']
Pxp = local_data['totalPressure_SI'] # in Torr; poiseuille_flow also outputs the pressure in Torr
species = local_data['gas']
cat = local_data['cathode']

### Compute feed-system loss
Tgas = 1000 # K
# Empty dataframe to fill
poiseuille_df = pd.DataFrame(columns=['cathode','upstreamPressurePoint','massFlowRate','insertDiameter','species','Pup','Pdown','Pratio'])
# Compute for Xe and Ar
for L,d,md,sp,Pup,lcat in zip(Lup,dc,mdot,species,Pxp,cat):
    # poiseuille_flow computes the *upstream* pressure from the downstream one. 
    # We're performing the opposite: we try to find the downstream pressure from the upstream one,
    # which means we have to solve a non-linear equation
    # Mathemathically, we're solving f(Pdown) = 0 = Pupstream - poiseuille_flow(Pdownstream), i.e., 
    # which downstream pressure gives us the experimental upstream one
    if sp == 'Xe':
        sol = root(lambda Pout: Pup-poiseuille_flow(L,d/2.0,md,Tgas,Pout,'Xe'),Pup)
    elif sp == 'Ar':
        sol = root(lambda Pout: Pup-poiseuille_flow(L,d/2.0,md,Tgas,Pout,'Ar'),Pup)
    
    Pdown = sol.x[0]
    # Ratio of upstream to downstream: how much bigger is the upstream pressure as compared
    # to the downstream one?
    r = Pup/Pdown
    # We store r-1 in the dataframe. This corresponds to the ratio (Pup - Pdown) / Pdown
    # Multiply by 100 to get the pressure increase from downstream to upstream as a percent
    # of the downstream value.
    poiseuille_df.loc[-1] = [lcat,L,md,d,sp,Pup,Pdown,r-1]
    poiseuille_df.index = poiseuille_df.index + 1
    poiseuille_df = poiseuille_df.sort_index()

### Output info
tdf = (poiseuille_df.Pratio*100).describe(percentiles=[0.05,0.95])
print("Pressure increase from downstream to upstream (%)")
print("Min value::5% percentile::Average::95% percentile::Max value")
print(f'{tdf.loc["min"]:.3}',"::",
         f'{tdf.loc["5%"]:.3}',"::",
         f'{tdf.loc["mean"]:.3}',"::",
         f'{tdf.loc["95%"]:.3}',"::",
         f'{tdf.loc["max"]:.3}')
