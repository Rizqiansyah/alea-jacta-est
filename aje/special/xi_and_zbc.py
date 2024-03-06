import numpy as np
from scipy import special, optimize
import pytensor.tensor as at
from pytensor import function as at_function

__all__ = ["xi_from_zbc", "zbc_from_xi"]

at_x = at.dvector('at_x')
at_z = at.switch(at.le(at_x,1e1), -1e-1, 
                 at.switch(at.le(at_x,1e2), -1e-2,
                          at.switch(at.le(at_x,1e3), -1e-3, 
                                    at.switch(at.le(at_x,1e4), -1e-4, 
                                             at.switch(at.le(at_x,1e5), -1e-5,
                                                       at.switch(at.le(at_x,1e6), -1e-6,
                                                                 at.switch(at.le(at_x,1e7), -1e-7, -1e-8)
                                                                )
                                                      )
                                             )
                                   )
                          )
                )
init_guess = at_function([at_x], at_z)
"""
Function to initialize the solver for xi_from_zbc
Arguments:
----------
zbc : float or array-like
    The centered upper bound parameter of the GEV Weibull distribution. Must be positive.

Returns
-------
float or array-like
    The initial guess for the xi_from_zbc solver.
"""

def func_xi(xi, z_bc):
    """
    xi solver
    For converting between centered upper bound (z_bc) and GEV shape parameter (xi) and vice versa
    Only need to import `xi_from_zbc()` and/or `zbc_from_xi()`
    tensor (aesara or pytensor) function for use with pymc is also available in `stats.stats_pymc` module
    For the `init_guess` argument, automatic initial guess is available by:
    `from stats.stats_pymc import init_guess`
    """
    if np.any(xi>=0):
        if isinstance(xi, float):
            return 999999
        else:
            return np.ones(xi.shape) * 999999
    g1 = special.gamma(1-xi)
    g2 = special.gamma(1-2*xi)
    value = np.log(g1)-0.5*np.log(g2-g1**2)-np.log(z_bc)
    return value

def fprime_xi(xi, z_bc):
    g1 = special.gamma(1-xi)
    g2 = special.gamma(1-2*xi)
    g1_prime = g1*special.digamma(1-xi)
    g2_prime = g2*special.digamma(1-2*xi)
    try:
        fprime = -special.digamma(1-xi) - (-g2_prime + g1*g1_prime) /(g2-g1*g1)
    except:
        print(xi)
        print(g1)
        print(g2)
    return fprime

def xi_from_zbc(z_bc, init_guess, **kwargs):
    return optimize.newton(func_xi, init_guess, fprime = fprime_xi, args=(z_bc,), **kwargs)

    
def zbc_from_xi(xi):
    return special.gamma(1-xi) / np.sqrt(special.gamma(1-2*xi) - special.gamma(1-xi)**2)