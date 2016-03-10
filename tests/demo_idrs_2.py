#!/usr/bin/env python

# Copyright (c) 2015 Reinaldo Astudillo and Martin B. van Gijzen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

if __name__ == '__main__':
    import numpy as np
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import math
    import time
    import sys
    sys.path.append('../')
    from idrs_solver.idrs import *
    from scipy.sparse.linalg import gmres
    from scipy.sparse.linalg import bicg 
    from scipy.sparse.linalg import bicgstab
#
# This Python script defines a 3D discretised convection-diffusion-reaction problem on the unit cube.
# The problem is solved with IDR(1), IDR(2), IDR(4), IDR(8). 
# This script is based of the following MATLAB script:
# http://ta.twi.tudelft.nl/nw/users/gijzen/example_idrs.m
    def tridiag(n,a,b,c):
        aux = np.ones([n])
        data = np.array([ a*aux, b*aux, c*aux])
        return sp.spdiags(data, [-1,0,1], n, n)
#
#   Start
    print 'FDM discretisation of a 3D convection-diffusion-reaction problem on a unit cube'
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
#
#   Define system
#   Defaults:
    h = 0.025
    eps = 1
    beta = np.array([0/math.sqrt(5), 250/math.sqrt(5), 500/math.sqrt(5)])
    r = 400
#    
#   Generate matrix
    m = int(round(1/h)-1)

    if  m < 1 : 
        print 'h too small, should be large than 0.5'
        exit(1)
    n = int(m*m*m)
    Sx = tridiag(m,-eps/h**2-beta[0]/(2*h),2*eps/h**2,-eps/h**2+beta[0]/(2*h))
    Sy = tridiag(m,-eps/h**2-beta[1]/(2*h),2*eps/h**2,-eps/h**2+beta[1]/(2*h))
    Sz = tridiag(m,-eps/h**2-beta[2]/(2*h),2*eps/h**2,-eps/h**2+beta[2]/(2*h))
    
    I = sp.eye(n,n)
    Is = sp.eye(m,m)
    A = sp.kron(sp.kron(Is,Is),Sx) + sp.kron(sp.kron(Is,Sy),Is)+ sp.kron(sp.kron(Sz,Is),Is) -r*I
    
    x = np.linspace(h,1-h,m)
    sol = np.kron(np.kron(x*(1-x),x*(1-x)),x*(1-x)).T
    b = (A.dot(sol))
    
    print '\nThe parameters of the problem are :'
    print 'Gridsize h = ',h,';'
    print 'Number of equations = ',n,';';
    print 'Diffusion parameter = ', eps,';';
    print 'Convection parameters = ('+str(beta[0])+','+str(beta[1])+','+str(beta[2])+');'
    print 'Reaction parameter = ',r,' (Note: positive reaction parameter gives negative shift to matrix);\n'
    
    msg = "Method {:8} Time = {:5.3f} Matvec = {:d} Residual = {:g}"
    tol = 1e-8
    maxit = 1000
    x0 = np.zeros([n,1])
    bnrm2 = np.linalg.norm(b)
    
    class ConvHistory:
        def __init__(self):
            self.data = []
        def func1(self,x):
            self.data  = np.append(self.data, np.linalg.norm(x))
        def func2(self,x):
            self.data  = np.append(self.data, np.linalg.norm(b-A.dot(x)))
        def normalize(self, alpha):
            self.data = self.data/alpha
        def addr0(self):
            self.data  = np.append([1.0], self.data)

    resIDR1 = ConvHistory()        
    t = time.time()
    x, info = idrs(A,b, tol=1e-8, s=1, callback = resIDR1.func2) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(x))/bnrm2
    print msg.format('IDR(1)', elapsed_time, len(resIDR1.data), residual)

    resIDR2 = ConvHistory()        
    t = time.time()
    x, info = idrs(A, b, tol=1e-8, s=2,callback = resIDR2.func2) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(x))/bnrm2
    print msg.format('IDR(2)', elapsed_time, len(resIDR2.data),residual)

    resIDR4 = ConvHistory()        
    t = time.time()
    x, info = idrs(A,b, tol=1e-8, s=4, callback = resIDR4.func2) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(x))/bnrm2
    print msg.format('IDR(4)', elapsed_time, len(resIDR4.data),residual)

    resIDR8 = ConvHistory()        
    t = time.time()
    x, info = idrs(A,b, tol=1e-8, s=8, callback = resIDR8.func2) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(x))/bnrm2
    print msg.format('IDR(8)', elapsed_time, len(resIDR8.data),residual)
    
    resBiCG = ConvHistory()        
    t = time.time()
    xb1, info = bicg(A,b, tol=1e-8, callback = resBiCG.func2) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(xb1))/bnrm2
    print msg.format('bicg', elapsed_time, 2*len(resBiCG.data),residual)

    resBiCGSTAB = ConvHistory()        
    t = time.time()
    xb2, info = bicgstab(A,b, tol=1e-8, callback = resBiCGSTAB.func2) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(xb2))/bnrm2
    print msg.format('bicgstab', elapsed_time, 2*len(resBiCGSTAB.data), residual)
    
    resGMRES = ConvHistory()
    t = time.time()
    xg, info = gmres(A,b, tol=1e-8, restart=200, callback = resGMRES.func1) 
    elapsed_time = time.time() - t
    residual =  np.linalg.norm(b - A.dot(xg))/bnrm2
    print msg.format('gmres', elapsed_time, len(resGMRES.data),residual)
    
    resIDR1.normalize(bnrm2)
    resIDR2.normalize(bnrm2)
    resIDR4.normalize(bnrm2)
    resIDR8.normalize(bnrm2)
    resBiCG.normalize(bnrm2)
    resBiCGSTAB.normalize(bnrm2)
    
    resBiCGSTAB.addr0()
    resBiCG.addr0()
    resGMRES.addr0()
    resIDR1.addr0()
    resIDR2.addr0()
    resIDR4.addr0()
    resIDR8.addr0()
    plt.semilogy(resGMRES.data, label='GMRES')
    plt.semilogy(resIDR1.data, label='IDR(1)')
    plt.semilogy(resIDR2.data, label='IDR(2)')
    plt.semilogy(resIDR4.data, label='IDR(4)')
    plt.semilogy(resIDR8.data, label='IDR(8)')
    plt.semilogy(range(0,2*len(resBiCG.data),2),resBiCG.data, label='Bi-CG')
    plt.semilogy(range(0,2*len(resBiCGSTAB.data),2),resBiCGSTAB.data, label='Bi-CGSTAB')
    plt.legend()
    plt.grid()
    plt.savefig('figure1.png')
