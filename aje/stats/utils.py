import numpy as np
from scipy import stats, integrate
from arviz.plots.plot_utils import calculate_point_estimate
import numpy as np
from scipy import optimize, special
from aje.utils import numpy_like_func_on_axis as _numpy_like_func_on_axis

__all__ = [
    "mode", 
    "numerical_moment", 
    "numerical_standardised_moment", 
    "bounded_gaussian_kde", 
    "gaussian_kde", 
    "ecdf", 
    "retake_block_max", 
    "retake_block_max_multi", 
    "retake_block_argmax", 
    "retake_block_max_pair", 
    "sum_N", 
    "sum_N_multi"
]

# ==================================
# Moments and mode

def mode(samples, axis=None, bw="default", circular=False):
    """
    Calculate the mode of a set of samples.
    Method similar to what arviz uses.
    Ref: https://discourse.pymc.io/t/how-is-the-mode-point-estimate-calculated-by-the-plot-posterior-method/5326/2
    Most recent version of arviz: https://github.com/arviz-devs/arviz/blob/main/arviz/plots/plot_utils.py
    Parameters
    ----------
    samples : array-like
        Samples to calculate the mode of.
    axis : Optional[int]
        Axis to calculate the mode along. If None, calculate the mode across all values.
    bw: Optional[float or str]
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `circular` is False
        and "taylor" (for now) when `circular` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    circular: Optional[bool]
        If True, it interprets the values passed are from a circular variable measured in radians
        and a circular KDE is used. Only valid for 1D KDE. Defaults to False.
    Returns
    -------
    float
        Mode of the samples.
    """
    def func(arr):
        return calculate_point_estimate(
            values = arr,
            point_estimate="mode",
            bw=bw,
            circular=circular,
        )

    return _numpy_like_func_on_axis(func, samples, axis)

def numerical_moment(fx, order, integral_lb, integral_ub, integral_method="quad", **integral_kwargs):
    """
    Numerical Moment Functions

    Calculate non-central moments numericaly. Useful when there is no (readily available) closed form expresion of the moments.

    Arguments:
    ----------
    fx = The PDF of the distribution
    order = The k-th moment
    integral_lb = The lower limit of the integral
    integral_ub = The lower limit of the integral
    integral_method = The numerical method of integration
    **integral_kwargs = kwargs to be passed to the integration function
    """
    if integral_method == "romberg":
        print("'romberg' integration method contains error right now. Please use with caution!")
            
        if (integral_lb == -np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
            print("ERR: 'romberg' integration method selected, integral_lb must be > -infinity")
            return "nan"

        if (integral_ub == np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
            print("ERR: 'romberg' integration method selected, integral_ub must be < infinity")
            return "nan"
        
    #function x^k * f(x), for uncentered moment calculations
    def xk_fx(x):
        return (x**order) * fx(x)

    #Integrate via numerical method
    if integral_method == "quad":
        return integrate.quad(func = xk_fx, a = integral_lb, b = integral_ub, **integral_kwargs)[0]
    elif integral_method == "romberg":
        return integrate.romberg(function = xk_fx, a = integral_lb, b = integral_ub, **integral_kwargs)
    else:
        print("ERR: Invalid integration method. Please set 'integral_method' to either 'quad' or 'romberg'")
        return "nan"
        
def numerical_standardised_moment(fx, order, integral_lb, integral_ub, integral_method="quad", **integral_kwargs):
    """
    Numerical Moment Functions

    Calculate central moments numericaly. Useful when there is no (readily available) closed form expresion of the moments.

    Arguments:
    ----------
    fx = The PDF of the distribution
    order = The k-th moment
    integral_lb = The lower limit of the integral
    integral_ub = The lower limit of the integral
    integral_method = The numerical method of integration
    **integral_kwargs = kwargs to be passed to the integration function
    """
    if integral_method == "romberg":
        print("'romberg' integration method contains error right now. Please use with caution!")
            
        if (integral_lb == -np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
            print("ERR: 'romberg' integration method selected, integral_lb must be > -infinity")
            return "nan"
        
        if (integral_ub == np.inf) and (integral_method == "romberg"): #Catch error if method is romberg
            print("ERR: 'romberg' integration method selected, integral_ub must be < infinity")
            return "nan"

    first_moment = numerical_moment(fx = fx, 
                                    order = 1, 
                                    integral_lb = integral_lb, 
                                    integral_ub = integral_ub, 
                                    integral_method = integral_method, 
                                    **integral_kwargs)
    if order==1:
        return first_moment
    else:
        #shifted and translated, for standardised moment calculations
        def std_xk_fx(x, k, translation, scale):
            return (((x-translation)/scale) ** k) * fx(x)

        if integral_method == "quad":
            second_centered_moment = integrate.quad(func = std_xk_fx, a = integral_lb, b = integral_ub, args = (2, first_moment, 1), **integral_kwargs)[0]
        elif integral_method == "romberg":
            second_centered_moment = integrate.romberg(function = std_xk_fx, a = integral_lb, b = integral_ub, args = (2, first_moment, 1), **integral_kwargs)
        else:
            print("ERR: Invalid integration method. Please set 'integral_method' to either 'quad' or 'romberg'")
            return "nan"

        if order==2:
            return second_centered_moment
        else:
            if integral_method == "quad":
                return integrate.quad(func = std_xk_fx, a = integral_lb, b = integral_ub, args = (order, first_moment, np.sqrt(second_centered_moment)), **integral_kwargs)[0]
            elif integral_method == "romberg":
                return integrate.romberg(function = std_xk_fx, a = integral_lb, b = integral_ub, args = (order, first_moment, np.sqrt(second_centered_moment)), **integral_kwargs)
            else:
                print("ERR: Invalid integration method. Please set 'integral_method' to either 'quad' or 'romberg'")
                return "nan"

# ==================================
# Gaussian KDE

def bounded_gaussian_kde(samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return x, y

def gaussian_kde(samples, lower = None, upper = None, npts = 100):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, npts)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    lower_x = x[0] - 3 * width if lower is None else lower
    upper_x = x[-1] + 3 * width if upper is None else upper

    x = np.concatenate([[lower_x], x, [upper_x]])
    y = np.concatenate([[0], y, [0]])
    return x, y

# ==================================
# Block Maximas

def ecdf(data):
    #Produce ranked results
    #Treatment separated columnwise
    x_ecdf = np.sort(data, axis = 0)
    n = x_ecdf.shape[0]
    F = np.arange(1, n+1) / float(n+1) #F = i/(n+1)
    return x_ecdf, F

def retake_block_max(data, period):
    #period must be int
    if not isinstance(period, int):
        raise ValueError("`period` must be type `int`")

    #Cutoff data at the end
    data = data[0: int(data.size - (data.size%period))]
    
    tmp = np.reshape(data, (int(data.size/period), period))
    return np.max(tmp, axis = 1)

def retake_block_max_multi(data, period):
    result = np.zeros((int(data.shape[0]/period), data.shape[1]))
    for i in range (0, data.shape[1]):
        result[:, i] = retake_block_max(data[:,i], period)
    return result

def retake_block_argmax(data, period):
    """
    Get the index of the block maximas in the data, per block
    Similar behaviour to retake_block_max

    Parameters
    ----------
    data : np.array
        Array of data to find the block maximas of.
    period : int
        Period of the data. Must be int.

    Returns
    -------
    np.array
        Array of the index of the block maximas.
    """

    #period must be int
    if not isinstance(period, int):
        raise ValueError("`period` must be type `int`")

    #Cutoff data at the end
    data = data[0: int(data.size - (data.size%period))]
    
    tmp = np.reshape(data, (int(data.size/period), period))
    return np.argmax(tmp, axis = 1)

def retake_block_max_pair(data, period, *pair_data):
    """
    Function to take the maximum of a block of data with pair data.
    For example, if you have {x1, x2, ..., xn} and y, and you want the maximum of y and the corresponding x.
    
    Parameters
    ----------
    data : array_like
        Array to take the maximum of.
    period : int
        Period of the data.
    *pair_data : array_like
        Array to take the maximum of with `data`.
        Can be multiple arrays, just keep adding the args at the end.

    Returns
    -------
    tuple
        Tuple of arrays with the maximum of `data` and `pair_data`.
    """

    #period must be int
    if not isinstance(period, int):
        raise ValueError("`period` must be type `int`")

    #Cutoff data at the end, then reshape for easy maximum calculation
    n_keep = int(data.size - (data.size%period))
    n_shape = int(data.size/period)
    data = data[0: n_keep]
    data = np.reshape(data, (n_shape, period))
    pair_data_tmp = [None] * len(pair_data)
    for i, pdata in enumerate(pair_data):
        pair_data_tmp[i] = np.reshape(pdata[0: n_keep], (n_shape, period))
    
    idx_max = retake_block_argmax(data, period)
    
    out = [None] * (len(pair_data) + 1)
    out[0] = np.max(data, axis = 1)
    for i, pdata in enumerate(pair_data_tmp):
        out[i+1] = pdata[np.arange(0, len(idx_max)), idx_max]

    return tuple(out)
    
def sum_N(data, period):
    """
    Function to sum up N within a period
    Useful e.g., for calculating number of monthly traffic (N) from daily data
    """
    #Cutoff data at the end
    data = data[0: data.size - (data.size%period)]
    
    tmp = np.reshape(data, (int(data.size/period), period))
    return np.sum(tmp, axis = 1)

def sum_N_multi(data, period):
    """
    Function to sum up N within a period
    Useful e.g., for calculating number of monthly traffic (N) from daily data
    """
    result = np.zeros((int(data.shape[0]/period), data.shape[1]))
    for i in range (0, data.shape[1]):
        result[:, i] = sum_N(data[:,i], period)
    return result