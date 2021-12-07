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

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error

from lmfit import Model, Parameters

from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr


class TheoreticalModelEstimator(BaseEstimator):
    '''
    Models to try: 1/4 - log(PI2) + c0 * PI5 + ...
    ... c1 * (1/PI2**2-1) * PI2**c2 * PI3**c3 * PI4**c4 * PI5**c5 * PI6**c6
    
    The pi-products may/may not be included by setting a mask value to true or false. 
    icdf is a dataframe of initial conditions. The index icidx is used to grab a given initial 
    condition in icdf. cmin and cmax limit the range of constant c0 that appears in front of Pi5.
    '''    
    def __init__(self,icdf, m2=True, m3=True, m4=True, m5=True, m6=True, icidx=0,cmin=1,cmax=10):
        '''
        Estimator constructor. Parameters:
            - icdf: Dataframe of initial conditions
            - icidx: Initial condition to pick
            - cmin, cmax: Bounds of the constant that appears in front of PI5
            - m2,m3,m4,m5,m6: Should those PI products be included in the fit
        '''
        super().__init__()
        self.params_ = None
        self.mse_ = -1
        self.n_parameters = 0
        self.mask_ = []
        self.p0 = np.zeros(7)
        self.icdf = icdf
        
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        self.m6 = m6
        self.icidx = icidx

        self.cmin = cmin
        self.cmax = cmax

    def create_function_string(self):
        fstr = ''
        lambcoeff = []
        lambvar = []
        
        # Start with 1/4 - log(PI2) + c0 * PI5
        fstr = 'log10(1/4 - log(x0) + a0*x3'
        
        lambcoeff = ['a0']
        lambvar = ['x0','x3']
        
        # Build parameters with the initial condition stored in p0
        self.params_ = Parameters()
        self.params_.add('a0', value=self.p0[0], min=self.cmin, max=self.cmax)
        
        self.n_parameters = 1
        if(any(self.mask_)):
            # If any other pi-product is included (i.e. a1 =/= 0), then add on the corresponding term
            fstr += '+ a1 * (x0**(-2)-1)'
            
            # One more parameter: a1 
            self.n_parameters += 1

            lambcoeff.append('a1')
            self.params_.add('a1', value=self.p0[1], min=0)
            
            # Add the pi-products to their respective power to be found
            for kk,v in enumerate(self.mask_):
                if v:
                    aname = 'a' + str(kk+2)
                    mname = 'm' + str(kk+2)
                    xname = 'x' + str(kk)
                    
                    fstr += '*(' + xname + '/' + mname + ')**' + aname
                    self.n_parameters += 1
                    
                    
                    lambcoeff.append(aname)
                    self.params_.add(aname, value=self.p0[kk+2], min=-5,max=5)
                    
                    if kk != 0 and kk != 3:
                        lambvar.append('x' + str(kk))

        # Finish the string: closing parenthesis of log10
        fstr += ')'
        
        # Sort the lambda variables
        lambvar.sort()
        lamb = ['x0','x1','x2','x3','x4',
                'a0','a1','a2','a3','a4','a5','a6',
                'm2','m3','m4','m5','m6']
        
        return lamb, lambvar, lambcoeff, fstr

    def fit(self, X, y):
        '''
        Estimator fit function. Parameters:
            - X: Input parameters
            - y: Input base truth
        Steps:
            - Grab the correct initial condition from the dataframe and stores the parameter
            values in p0
            - Generate the functional string to be lambdified
            - Define the fit
            - Perform the fit
        '''
        ### Check X and Y
        X, y = check_X_y(X, y, accept_sparse=True,copy=True)
          
        ### Create the mask: we pick only those PI-Products
        self.mask_ = [self.m2,self.m3,self.m4,self.m5,self.m6]
        mask = self.mask_ # Necessary for the dataframe query
        
        ### Get the initial conditions by querying the dataframe
        qry = 'm2==@mask[0] and m3==@mask[1] and m4==@mask[2] and m5==@mask[3] and m6==@mask[4]'
        xf = self.icdf.query(qry).reset_index(drop=True)
        xf = xf[['p0','p1','p2','p3','p4','p5','p6']]
        self.p0 = np.array(xf.loc[self.icidx])
        
        ##############################
        ### Build the function to fit
        ##############################
        lamb, lambvar, lambcoeff, fstr = self.create_function_string()

        # Parse the string to create a callable function
        self.func_ = lambdify(lamb,parse_expr(fstr),modules='numpy')

        def fun(x0=1,x1=1,x2=1,x3=1,x4=1,
                a0=4,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,
                m2=1,m3=1,m4=1,m5=1,m6=1):    
            return self.func_(x0,x1,x2,x3,x4,a0,a1,a2,a3,a4,a5,a6,m2,m3,m4,m5,m6)

        ##############################
        ### Define LMFIT model
        ##############################
        # Define the lmfit model; omit nan's to avoid any exceptions
        self.fit_model_ = Model(fun, 
                                independent_vars=lambvar,
                                param_names=lambcoeff,
                                nan_policy='propagate')
        
        ##############################
        ### Perform the fit
        ##############################
        self.fit_ = self.fit_model_.fit(np.log10(y),
                                        params=self.params_,
                                        x0=X[:,0],
                                        x1=X[:,1],
                                        x2=X[:,2],
                                        x3=X[:,3],
                                        x4=X[:,4],
                                        m2=1.0,
                                        m3=1.0,
                                        m4=1.0,
                                        m5=1.0,
                                        m6=1.0,
                                        method='least_squares')
        
        # The model is now fitted.
        self.is_fitted_ = True
        return self        

    def predict(self, X):
        '''
        Estimator predict function. Use the function self.func_ defined in the "fit" method. 
        We check that the estimator has been fitted (and, therefore, func_ has been defined).
        '''
        X = check_array(X, accept_sparse=True,copy=True)
        check_is_fitted(self, 'is_fitted_')

        # Extract params
        alist = []
        
        for kk in range(7):
            try:
                alist.append(self.fit_.params['a'+str(kk)].value)
            except:
                alist.append(1e10)
         
        mlist = []
        for kk in range(X.shape[1]):
            mlist.append(1.0)
        
        # Extract variables
        x0 = X[:,0]
        x1 = X[:,1]
        x2 = X[:,2]
        x3 = X[:,3]
        x4 = X[:,4]
                
        # Predicted Y using PI2-PI6 and the coefficients
        Yp = self.func_(x0,x1,x2,x3,x4,*alist,*mlist)

        return Yp

    def score(self, X,y):
        '''
        Scoring function. We use the AIC as computed by the estimator itself.
        '''
        X, y = check_X_y(X, y, accept_sparse=True,copy=True)
        y = np.log10(y)
        
        check_is_fitted(self, 'is_fitted_')
                
        nsamples = X.shape[0]

        aic = 1e10
        # Predicted y
        ypred = self.predict(X)

        self.mse_ = mean_squared_error(y,ypred)
        
        aic = self.fit_.aic

        if np.isnan(aic):
            print(self.n_parameters,nsamples,self.mse_)
        
        return -aic