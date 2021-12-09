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
File: range_of_parameters.py
Author: Pierre-Yves Taunay
Date: December, 2021
Description: reproduces Figure 2 as a boxplot. The boxplots include more information than the 
original Figure 2 since it displays quartiles and outliers. The insert and orifice Reynolds numbers
have been separated here into two plots to avoid overlap.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_hdf("../../data/cathode_database.h5",key="data")

# Merge some of the data
data.loc[data.cathode=='JPL-1.5cm-3mm','cathode'] = 'JPL-1.5cm'
data.loc[data.cathode=='JPL-1.5cm-5mm','cathode'] = 'JPL-1.5cm'

data.loc[data.cathode=='Salhi-Ar-0.76','cathode'] = 'Salhi (Ar and Xe)'
data.loc[data.cathode=='Salhi-Ar-1.21','cathode'] = 'Salhi (Ar and Xe)'
data.loc[data.cathode=='Salhi-Xe','cathode'] = 'Salhi (Ar and Xe)'

data.loc[data.cathode=='Siegfried-NG','cathode'] = 'Siegfried (Ar and Xe)'
data.loc[data.cathode=='Siegfried','cathode'] = 'Siegfried (Hg)'


# Add a year for sorting
data.loc[data.cathode=='Siegfried (Hg)','year'] = 1979
data.loc[data.cathode=='Siegfried (Ar and Xe)','year'] = 1980
data.loc[data.cathode=='Friedly','year'] = 1990
data.loc[data.cathode=='Salhi (Ar and Xe)','year'] = 1994
data.loc[data.cathode=='T6','year'] = 1998
data.loc[data.cathode=='AR3','year'] = 1999
data.loc[data.cathode=='SC012','year'] = 1999
data.loc[data.cathode=='EK6','year'] = 1999
data.loc[data.cathode=='NEXIS','year'] = 2005
data.loc[data.cathode=='NSTAR','year'] = 2005
data.loc[data.cathode=='JPL-1.5cm','year'] = 2011
data.loc[data.cathode=='PLHC','year'] = 2020

data.sort_values(by='year',axis=0,inplace=True)

fig,ax = plt.subplots(2,2)

# First column: Reynolds numbers
sns.boxplot(x=data.reynoldsNumber,y=data.cathode,ax=ax[1][0])
sns.boxplot(x=data.reynoldsNumberInsert,y=data.cathode,ax=ax[0][0])

ax[0][0].set_xscale("log")
ax[1][0].set_xscale("log")
ax[0][0].set_xlim([1e-2,1e2])
ax[1][0].set_xlim([1e-2,1e2])

ax[0][0].set_xlabel("Insert Reynolds number")
ax[1][0].set_xlabel("Orifice Reynolds number")

# Knudsen number
sns.boxplot(x=data.orificeKnudsenNumber,y=data.cathode,ax=ax[0][1])

ax[0][1].set_xscale("log")
ax[0][1].set_xlabel("Orifice Knudsen number")
ax[0][1].set_xlim([1e-2,1e1])

# Entrance length
sns.boxplot(x=data.entranceLength,y=data.cathode,ax=ax[1][1])

ax[1][1].set_xscale("log")
ax[1][1].set_xlabel("Orifice entrance length")
ax[1][1].set_xlim([1e-3,1e1])

# Remove extra labels
ax[0][1].get_yaxis().set_visible(False)
ax[1][1].get_yaxis().set_visible(False)

plt.show()