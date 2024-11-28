# Mostly working for now
# Some notes:
# ppf and rvs can sometimes fail when E or zb has large value
# This is because the gradient (pdf) is very small.
# I suspect this will be a problem when doing Bayesian inference as well.
# So will need to figure out how to standardize the input into unit scale.
# e.g., in GEV, the standardization is (x - mu)/sigma results in GEV(0, 1, xi)
# How about CDS?

from numpy import sum, exp, log, max, ones_like
from numpy import inf
from scipy.special import gamma
#Import name to match pytensor namespace
from numpy import greater as gt, greater_equal as ge, equal as eq, power as pow, where as switch
from aje.stats.distributions.utils.numpy import vector_2d

#Not pytensor compatible functions
from scipy.optimize import newton

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
    return pow(( gamma(1-xi) * ((gt(zb, x)) * (zb-x))  / (zb-E)), (-1/xi))

def argcheck(**kwargs):
    if any(ge(kwargs["xi"], 0)):
        raise ValueError("xi must be less than 0")
    if any(ge(kwargs["E"], kwargs["zb"])):
        raise ValueError("zb must be greater than E")
    return kwargs

def support(**kwargs):
    return (-inf, max(kwargs["zb"]))

def logcdf(x, E, zb, xi, *args, **kwargs):
    return sum(-_multidimensional_t(x, E, zb, xi), axis=0)

def cdf(x, E, zb, xi, *args, **kwargs):
    return exp(logcdf(x, E, zb, xi))

def pdf(x, E, zb, xi, *args, **kwargs):
    """
    PDF of CDS distribution.

    Notes
    -----
    The function will sometimes throw a warning when any of the x is equal to any of the zb.
    The first warning is
    ```
    RuntimeWarning: invalid value encountered in divide
        -_multidimensional_t(x, E, zb, xi)/(xi* (zb-x))
    ```
    This is okay. The reason for the error is when zb = x, then the function divides by zero.
    The limit of the function as x -> zb is zero (However, this needs to be checked more thoroughly!). 

    The second warning is
    ```
    RuntimeWarning: invalid value encountered in multiply
    term1 * term2
    ```
    This is also okay. 
    The reason is as x -> -inf, term1 -> inf and term2 -> 0.
    However, term2 grows faster, so lim(x->-inf) f(x) = 0.
    Logically, as x -> we're going away from the center of the distribution, so the density should approach zero too.

    So this function already handles this case by assigning the correct zero value when the warning is given.
    
    """
    t = _multidimensional_t(x, E, zb, xi)
    zb, xi = vector_2d(zb), vector_2d(xi)

    #pdf = term1 * term2
    #as x -> -inf, term1 -> inf and term2 -> 0
    #However, term2 grows faster, so lim(x->-inf) f(x) = 0
    #logically, as x -> we're going away from the center of the distribution, so the density should approach zero too.
    term1 = sum( 
        switch(
                gt(zb, x),
                -t/(xi* (zb-x)),
                0,
            ), axis = 0
        )
    term2 = exp(sum(-t, axis=0))

    return switch(
        eq(term2, 0),
        0,
        term1 * term2
    )

    # return sum(
    #     switch(
    #         gt(zb, x),
    #         -t/(xi* (zb-x)),
    #         0,
    #     ), axis = 0
    # ) * exp(sum(-t, axis=0))

def logpdf(x, E, zb, xi, *args, **kwargs):
    t = _multidimensional_t(x, E, zb, xi)
    zb, xi = vector_2d(zb), vector_2d(xi)

    #Use switch case to cover x -> -inf and x-> zb.
    #See the pdf method above for explanation.

    term1 = log(sum( 
        switch(
                gt(zb, x),
                -t/(xi* (zb-x)),
                0,
            ), axis = 0
        ))
    term2 = sum(-t, axis=0)

    return switch(
        eq(term2, -inf),
        -inf,
        term1 + term2
    )

    # return log(sum(
    #     switch(
    #         gt(zb, x),
    #         -t/(xi* (zb-x)),
    #         0,
    #     ), axis = 0
    # )) + sum(-t, axis=0)

from aje.special import kp_from_xi
from numpy import argmax, vectorize
from scipy.optimize import brentq, fmin_tnc
def ppf(q, E, zb, xi, init_guess=None, method="fmin_tnc", *args, **kwargs):
    """
    ppf of CDS distribution.

    Parameters
    ----------
    q : float or array-like
        quantile to evaluate the function at
    E : array-like
        Mean of the individual GEV distribution. Vector of E.
    zb : array-like
        Upper bound of the individual GEV distribution. Vector of zb.
    xi : array-like
        Shape parameter of the individual GEV distribution. Vector of xi.
    init_guess : float or array-like, optional
        Initial guess for the solver. If None, will use default value.
        Ignored when using brentq method
    method : str, optional
        Method to use for the solver. Default is "fmin_tnc".
        Other options are "newton", "newton_without_fprime", and "brentq".

    Returns
    -------
    x : float or array-like
        Quantile of the function evaluated at q.
    
    Notes
    -----
        The default "fmin_tnc" method is the most stable.
        It uses the gradient and the upper bound of the distribution to find the solution.
        However, it is also the slowest.

        Both newton methods are somewhat unstable.
        Sometimes it can go to x > zb, giving 0 pdf and getting stuck.
        The newton_without_fprime method is slower than the newton method with fprime, but may help in case it got stuck because of the pdf.

        The brentq method can be stable and faster than fmin_tnc for low q.
        However, it requires that the returned value is the opposite sign of the upper bound, which is not always true, especially for high q.

    """
    if method == "fmin_tnc":
        ubound = max(zb) #Set the maximum of the upper bound as the upper bound
        idx = argmax(E)
        init_guess = max(E) if init_guess is None else init_guess #Usually good enough starting guess
        def _f(_q):
            fsq = lambda x: (cdf(x, E, zb, xi) - _q)**2
            fsqprime = lambda x: 2 * (cdf(x, E, zb, xi) - _q) * pdf(x, E, zb, xi)
        
            # lbound = E[idx] + (zb[idx] - E[idx]) * kp_from_xi(xi=xi[idx], p=(1-_q))
            # lbound = lbound - (ubound - lbound)
            lbound = -inf
            return fmin_tnc(fsq, init_guess, fprime=fsqprime, bounds=[(lbound, ubound)], messages=0)[0]
        return vectorize(_f) (q)

    if method == "newton" or method =="newton_with_fprime":
        #Newton method with fprime
        #Fastest, but sometimes goes to x > zb, giving 0 pdf and getting stuck
        init_guess = max(E) if init_guess is None else init_guess #Usually good enough starting guess
        return newton(lambda x: cdf(x, E, zb, xi) - q, ones_like(q) * init_guess, fprime=lambda x: pdf(x, E, zb, xi))

    if method == "newton_without_fprime":
        #Newton method without fprime
        #Not as fast as with fprime, uses zp as the init point
        idx = argmax(E)
        if init_guess is None:
            init_guess = E[idx] + (zb[idx] - E[idx]) * kp_from_xi(xi=xi[idx], p=(1-q)) #Usually good enough starting guess
            print(init_guess)
        else:
            #Make sure init_guess has the same length as q
            init_guess = ones_like(q) * init_guess
        return newton(lambda x: cdf(x, E, zb, xi) - q, init_guess)

    #brentq method with bounds
    #Most stable so far
    if method == "brentq":
        idx = argmax(E)
        ubound = max(zb)
        ubound = 1. if ubound <= 0. else ubound
        def _f(_q):
            lbound = E[idx] + (zb[idx] - E[idx]) * kp_from_xi(xi=xi[idx], p=(1-_q))
            lbound = lbound - (ubound - lbound) #Make sure solution is in bounds
            lbound = -1. if lbound >= 0. else lbound
            print(lbound, ubound)
            return brentq(lambda x: cdf(x, E, zb, xi) - _q, lbound, ubound)
        return vectorize(_f) (q)
    
    raise ValueError("method must be one of 'newton', 'newton_without_fprime', or 'brentq'")