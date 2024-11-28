# Mostly working for now
# Some notes:
# ppf and rvs can sometimes fail when E or zb has large value
# This is because the gradient (pdf) is very small.
# I suspect this will be a problem when doing Bayesian inference as well.
# So will need to figure out how to standardize the input into unit scale.
# e.g., in GEV, the standardization is (x - mu)/sigma results in GEV(0, 1, xi)
# How about CDS?

from pytensor.tensor import sum, exp, log, max
from numpy import inf
from pytensor.tensor import gamma
from pytensor.tensor import gt, ge, pow, switch
from aje.stats.distributions.utils.pytensor import vector_2d

def _multidimensional_t(x, E, zb, xi):
    """
    GEV t(x; E, zb, xi) function, given mean, upper bound, and shape.
    Uses Coles parameterization, i.e., xi < 0 for Weibull, always.

    Note this treats the parameter as a vector, and the x as another vector.
    It applies each parameter set (i.e., tuples of (E, zb, xi)) to each x value.
    Parameter vector and x vector does not have to be the same length.

    Parameters
    ----------
    x : numpy.ndarray
        Values to evaluate the function at
    E : numpy.ndarray
        Mean of the distribution
    zb : numpy.ndarray
        Upper bound of the distribution
    xi : numpy.ndarray
        Shape parameter of the distribution. Follows Coles (2001) parameterization.
    """
    E, zb, xi = vector_2d(E), vector_2d(zb), vector_2d(xi)
    x = x.flatten()
    out = pow(( gamma(1-xi) * ((gt(zb, x)) * (zb-x))  / (zb-E)), (-1/xi))
    return out
    #return out.reshape(())

def argcheck(**kwargs):
    if any(ge(kwargs["xi"], 0)):
        raise ValueError("xi must be less than 0")
    if any(ge(kwargs["E"], kwargs["zb"])):
        raise ValueError("zb must be greater than E")
    return kwargs

def support(**kwargs):
    return (-inf, max(kwargs["zb"]))

def logcdf(x, E, zb, xi):
    return sum(-_multidimensional_t(x, E, zb, xi), axis=0)

def cdf(x, E, zb, xi):
    return exp(logcdf(x, E, zb, xi))

def pdf(x, E, zb, xi):
    """
    PDF of CDS distribution.

    Notes
    -----
    The function will sometimes throw a warning when any of the x is equal to any of the zb.
    The warning is
    ```
    RuntimeWarning: invalid value encountered in divide
        -_multidimensional_t(x, E, zb, xi)/(xi* (zb-x))
    ```
    This is okay. The reason for the error is when zb = x, then the function divides by zero.
    The limit of the function as x -> zb is zero (However, this needs to be checked more thoroughly!). 
    So this function already handles this case by assigning the correct zero value when the warning is given.
    
    """
    t = _multidimensional_t(x, E, zb, xi)
    zb, xi = vector_2d(zb), vector_2d(xi)
    return sum(
        switch(
            gt(zb, x),
            -t/(xi* (zb-x)),
            0,
        ), axis = 0
    ) * exp(sum(-t, axis=0))

def logpdf(x, E, zb, xi):
    t = _multidimensional_t(x, E, zb, xi)
    zb, xi = vector_2d(zb), vector_2d(xi)
    return log(sum(
        switch(
            gt(zb, x),
            -t/(xi* (zb-x)),
            0,
        ), axis = 0
    )) + sum(-t, axis=0)

def ppf(q, E, zb, xi, init_guess=None):
    raise NotImplementedError("ppf for pytensor is not implemented yet")