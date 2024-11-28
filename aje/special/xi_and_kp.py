import numpy as np
from scipy import special, optimize

__all__ = ["xi_from_kp", "kp_from_xi"]

def log_1mkp(xi, p=0.25):
    """
    Log of 1-kp.
    See kp_from_xi for the definition of the kp function.

    Parameters
    ----------
    xi : float or array-like
        xi parameter of the GEV Weibull distribution. Must be negative.
    p : float, optional
        the probability of non-exceedance. The default is 0.25. Must be between 0 and 1.
    """
    #Negative absolute the xi so it never returns positive value
    xi = -np.abs(xi)
    cp = -np.log(1-p)
    return -np.log(special.gamma(1-xi)) - xi*np.log(cp)

def func_log_1mkp(xi, kp, p=0.25):
    r"""
    Function to compute the difference between log(1-kp) and the log(1-kp_target) given xi and kp_target.
    Solving this function is the same as the same as solving the inverse of kp(xi)

    Parameters
    ----------
    xi : float or array-like
        xi parameter of the GEV Weibull distribution. Must be negative.
    kp : float or array-like
        kp to be used as target.
    p : float, optional
        the probability of non-exceedance. The default is 0.25. Must be between 0 and 1.
    """
    xi = np.array(xi)
    return log_1mkp(xi, p=p) - np.log(1-kp)

def fprime_log_1mkp(xi, kp, p=0.25):
    """
    The gradient to the log(1-kp)
    The kp argument is there to comply with the xi_from_kp solver syntax. It is not actually being used in the computation.
    """
    cp = -np.log(1-p)
    return special.digamma(1-xi) - np.log(cp)

#this method is about 50x faster than using the root and directly solving for kp
#using the log and newton allows the gradient to be computed easily and used by the solver
#this also allows array as input
def xi_from_kp(kp, init_guess=-0.5, p=0.25, **kwargs):
    r"""
    Compute the inverse of the following function
    k_p (\xi) = 1 - \frac{c_p^{-\xi}}{\Gamma(1-\xi)}
    where c_p = -\ln(1-p)
    Useful for computing GEV likelihood using the quantile parameterization.

    Verified to and produce correct results.

    Parameters
    ----------
    kp : float or array-like
        kp to be used as target.
    init_guess : float or array_like, optional
        Initial guess for the solver. The default is -0.5.
        The default is fairly good for the typical range of xi expected in most problem (-1 to 0.0)
        Any values below this range will require a better initial guess. 
        Usually putting the initial guess to a negative large number will work, at the cost of more computation.
    """
    return optimize.newton(func_log_1mkp, np.ones_like(kp) * init_guess, args=(kp, p), fprime=fprime_log_1mkp, **kwargs)

def kp_from_xi(xi, p=0.25):
    r"""
    Compute kp from the xi parameter of the GEV Weibull distribution
    k_p = 1 - \frac{c_p^{-\xi}}{\Gamma(1-\xi)}
    where c_p = -\ln(1-p)

    Parameters
    ----------
    xi : float or array-like
        xi parameter of the GEV Weibull distribution. Must be negative.
    p : float, optional
        the probability of non-exceedance. The default is 0.25. Must be between 0 and 1.
    """
    cp = -np.log(1-p)
    return 1 - cp**(-xi)/special.gamma(1-xi)