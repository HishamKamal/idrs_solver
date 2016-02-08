# IDR(s) Solver in Python
The Induced Dimension Reduction method (IDR(s)) is a short-recurrences Krylov method that
solves the system of linear equation,
                                      Ax = b.
This Python implementation is based on [2]. The interface of the idrs function is compatible
with the Krylov method implemented in Scipy.

      idrs(A, b, x0=None, tol=1e-5, s=4, maxit=None, xtype=None, M=None, callback=None)
# Parameter
    Basic
    -------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    Returns
    -------
    x : {array, matrix}
        The converged solution.
    info : integer
        Provides convergence information:
          * 0  : successful exit
          * >0 : convergence to tolerance not achieved, number of iterations
          * <0 : illegal input or breakdown (not implemented yet)
    Other parameters
    ----------------
    x0 : {array, matrix}
        Starting guess for the solution (a vector of zeros by default).
    tol : float
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`.
    s : integer, optional
        specifies the dimension of the shadow space. Norrmally, a higher 
        s gives faster convergence, but also makes the method more expensive.
        Default is 4.
    maxiter : integer, optional
	Maximum number of iterations.  Iteration will stop after maxiter
	steps even if the specified tolerance has not been achieved.
    xtype : {'f','d','F','D'}
        This parameter is DEPRECATED --- avoid using it.
        The type of the result.  If None, then it will be determined from
        A.dtype.char and b.  If A does not have a typecode method then it
        will compute A.matvec(x0) to get a typecode.   To save the extra
        computation when A does not have a typecode attribute use xtype=0
        for the same type as b or use xtype='f','d','F',or 'D'.
        This parameter has been superseded by LinearOperator.
    M : {sparse matrix, dense matrix, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve.  Effective preconditioning 
        dramatically improves the rate of convergence, which implies that 
        fewer iterations are needed to reach a given error tolerance.  
        By default, no preconditioner is used.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

# License

This software is distributed under the [MIT License](http://opensource.org/licenses/MIT).

# References
1. IDR(s): a family of simple and fast algorithms for solving large 
        nonsymmetric linear systems. 
        P. Sonneveld and M. B. van Gijzen
        SIAM J. Sci. Comput. Vol. 31, No. 2, pp. 1035--1062, 2008 
2. Algorithm 913: An Elegant IDR(s) Variant that Efficiently Exploits 
        Bi-orthogonality Properties. 
        M. B. van Gijzen and P. Sonneveld
        ACM Trans. Math. Software,, Vol. 38, No. 1, pp. 5:1-5:19, 2011
3. This file is a translation of the following MATLAB implementation:
http://ta.twi.tudelft.nl/nw/users/gijzen/idrs.m
4. IDR(s)' webpage http://ta.twi.tudelft.nl/nw/users/gijzen/IDR.html
