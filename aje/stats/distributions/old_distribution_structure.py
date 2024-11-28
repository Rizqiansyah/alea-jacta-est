"""
Deprecation Warning:
This module will be deprecated in the future.
This module will be re-written to be more consistent with scipy.stats, and refactored into the .stats.dist module.
This will allow pytensor and numpy methods to be written for the distributions.
"""

import numpy as np
from aje.special import xi_from_zbc, zbc_from_xi
from aje.special.xi_and_zbc import init_guess
from aje.stats.utils import numerical_moment, numerical_standardised_moment, retake_block_max
from scipy.stats import genextreme, uniform
from scipy.stats import beta as scipy_beta_dist
from scipy import special

__all__ = ["gentruncated", "genmaxima", "genextreme_weibull", "beta", "mixturedistribution", "multimodalbeta"]

#=======================================
#         gentruncated() class
#=======================================
#Generic Truncated Distribution
#Provide a class for upper, lower, and upper & lower trucated distribution, for any parent distribution
#REF: https://timvieira.github.io/blog/post/2020/06/30/generating-truncated-random-variates/

class gentruncated():
    """ 
    Class for generic truncated random variable
    
    Arguments:
    parent:
        The parent distribution.
    lower:
        The lower truncation point. If "lower" argument is less than equal the parent support, will use methods from the parent itself + a warning (surpressable).
        Default is "None" for no lower truncation.
    
    upper:
        The upper truncation point. If "upper" argument is more than equal the parent support, will use methods from the parent itself + a warning (surpressable).
        Default is "None" for no upper truncation.
    """
    #Helper methods since not using rv_continuous class from scipy
    def __init__(self, parent, lower = None, upper = None, warning = True):
        self.parent = parent
        self._set_lower(lower, warning)
        self._set_upper(upper, warning)
        if not self.argcheck():
            raise ValueError("lower truncation point is set equal to or higher than upper truncation point. Check the truncation points!")
            
    def _set_lower(self, lower, warning = True):
        if lower == None:
            self.lower = self.parent.support()[0]
            self.lower_flag = True
        else:
            self.lower = lower
            self.check_lower_support(warning)
            
    def _set_upper(self, upper, warning = True):
        if upper == None:
            self.upper = self.parent.support()[1]
            self.upper_flag = True
            self._upper_flag_is_none = True
        else:
            self.upper = upper
            self.check_upper_support(warning)
        
    def check_lower_support(self, warning = True):
        if self.parent.support()[0] >= self.lower:
            if warning:
                print("Warning: lower truncation point is set lower than or equal to parent lower support.")
            self.lower_flag = True
            return True
        elif self.lower >= self.parent.support()[1]: #Lower truncation higher than parent upper support
            raise ValueError("Lower truncation point CANNOT be higher or equal than parent upper support. Check both parent support and truncation point!")
        else:
            self.lower_flag = False
            return False
    
    def check_upper_support(self, warning = True):
        if self.parent.support()[1] <= self.upper:
            if warning:
                print("Warning: upper truncation point is set higher than or equal to parent upper support.")
            self.upper_flag = True
            return True
        elif self.upper <= self.parent.support()[0]: #Upper truncation lower than parent lower support
            raise ValueError("Upper truncation point CANNOT be lower or equal than parent lower support. Check both parent support and truncation point!")
        else:
            self.upper_flag = False
            return False
        
    def check_support(self, warning = True):
        self.check_lower_support(warning)
        self.check_upper_support(warning)
        
    def set_parent(self, parent, warning = True):
        if warning:
            print("WARNING: This method may contain bugs with the truncation point support. It is recommended you create a new gentruncated object instead! ")
        self.parent = parent
        self._set_lower(self.lower, warning)
        self._set_upper(self.upper, warning)
        if not self.argcheck():
            raise ValueError("lower truncation point is set equal to or higher than upper truncation point. Check the truncation points!")
    
    def get_parent(self):
        return self.parent
    
    def set_lower(self, lower, warning = True):
        self._set_lower(lower, warning)
        if not self.argcheck():
            raise ValueError("lower truncation point is set equal to or higher than upper truncation point. Check the truncation points!")
    
    def get_lower(self):
        return self.lower
    
    def set_upper(self, upper, warning = True):
        self._set_upper(upper, warning)
        if not self.argcheck():
            raise ValueError("lower truncation point is set equal to or higher than upper truncation point. Check the truncation points!")
    
    def get_upper(self):
        return self.upper
    
    # Methods similar to scipy, where possible
    def argcheck(self):
        if self.lower >= self.upper:
            return False
        else:
            return True
    
    def support(self):
        #Alias for get_support() method
        return self.get_support()
    
    def get_support(self):
        if self.lower_flag:
            lower = self.parent.support()[0]
        else:
            lower = self.lower
        
        if self.upper_flag:
            upper = self.parent.support()[1]
        else:
            upper = self.upper
        return (lower, upper)
    
    def _get_F_support(self):
        #Method to get CDF at the truncation points/ support
        if self.lower_flag:
            F_a = 0
        else:
            F_a = self.parent.cdf(self.lower)
        
        if self.upper_flag:
            F_b = 1.0
        else:
            F_b = self.parent.cdf(self.upper)
        return (F_a, F_b)
    
    def rvs(self, size=1, random_state=None):
        u = uniform.rvs(size = size, random_state = random_state)
        return self.ppf(u)
        
    def pdf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        F_a, F_b = self._get_F_support()
        out = np.zeros(x.shape)
        x_filter = np.logical_and(x >= self.lower, x <= self.upper)
        out[x_filter] = self.parent.pdf(x[x_filter]) / (F_b - F_a)
        return out
    
    def logpdf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        F_a, F_b = self._get_F_support()
        out = np.ones(x.shape) * -np.inf
        x_filter = np.logical_and(x >= self.lower, x <= self.upper)
        out[x_filter] = self.parent.logpdf(x[x_filter]) - np.log(F_b - F_a)
        return out
        
    def cdf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        F_a, F_b = self._get_F_support()
        out = np.zeros(x.shape)
        out[x > self.upper] = 1
        x_filter = np.logical_and(x >= self.lower, x <= self.upper)
        out[x_filter] = (self.parent.cdf(x[x_filter]) - F_a) / (F_b - F_a)
        return out
    
    def logcdf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        F_a, F_b = self._get_F_support()
        out = np.ones(x.shape) * -np.inf
        out[x > self.upper] = 0
        x_filter = np.logical_and(x >= self.lower, x <= self.upper)
        if self.lower_flag:
            out[x_filter] = self.parent.logcdf(x[x_filter]) - np.log(F_b)
        else:
            out[x_filter] = np.log(self.parent.cdf(x[x_filter]) - F_a) - np.log(F_b - F_a)
        return out
    
    def sf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return 1-self.cdf(x)
    
    def logsf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return np.log(self.sf(x))
    
    def ppf(self, q):
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        F_a, F_b = self._get_F_support()
        return self.parent.ppf(F_a + q*(F_b - F_a))
    
    def isf(self, q):
        print("NOT CODED YET")
        return "nan"
    
    def entropy(self):
        print("NOT CODED YET")
        return "nan"
    
    def fit(self, data):
        print("NOT CODED YET")
        return "nan"
    
    def interval(self, confidence):
        print("NOT CODED YET")
        return "nan"
    
    def expect(self, args=(), lb=None, ub=None, conditional=False, **kwds):
        print("NOT CODED YET")
        return "nan"
    
    #Methods to calculate statistical moments
    #All moments are calculated numerically, so there will be some numerical error
    
    #Method to calculate non-central moment
    def moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        if integral_lb == None:
            integral_lb = self.get_support()[0]
        if integral_ub == None:
            integral_ub = self.get_support()[1]
        
        return numerical_moment(fx = self.pdf,
                                order = order, 
                                integral_lb = integral_lb, 
                                integral_ub = integral_ub, 
                                integral_method = integral_method, 
                                **integral_kwargs)
            
    #Method to calculate standardised moment
    #Unique to this, not availabel in scipy. Maybe better to code the expect() method for this
    def std_moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        if integral_lb == None:
            integral_lb = self.get_support()[0]
        if integral_ub == None:
            integral_ub = self.get_support()[1]
        
        return numerical_standardised_moment(fx = self.pdf,
                                             order = order, 
                                             integral_lb = integral_lb, 
                                             integral_ub = integral_ub, 
                                             integral_method = integral_method, 
                                             **integral_kwargs)
    
    #Method to get statistics, similar to stats() method in scipy distribution
    #Does not behave exactly the same as scipy. Need to code properly.
    def stats(self, moments, **kwargs):
        moments = list(moments)
        for a in moments:
            if a == 'm':
                return self.moment(order = 1, **kwargs)
            elif a == 'v':
                return self.std_moment(order = 2, **kwargs)
            elif a == 's':
                return self.std_moment(order = 3, **kwargs)
            elif a == 'k':
                return self.std_moment(order = 4, **kwargs) - 3.0 #Excess kurtosis
            else:
                print("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")
    
    def mean(self, **kwargs):
        return self.moment(order = 1, **kwargs)
    
    def var(self, **kwargs):
        return self.std_moment(order = 2, **kwargs)
    
    def std(self, **kwargs):
        return np.sqrt(self.var(**kwargs))
    
    def median(self):
        print("NOT CODED YET")
        return "nan"
    
        

#=======================================
#         genmaxima() class
#=======================================
#Generic Maxima Distribution
#Distribution of a maxima of some generic parent distribution, given a block size
#Theoretically this converge to a GEV given large enough block size (N)
#However this should give precise values for any block size, with some exceptions:
# 1. All statistical moments are numerical computed, so there will be some computational error
# 2. GEV equivalent parameterisation and CDF comaparison assumes convergence to GEV, and support ONLY Weibull domain of attraction
#
#TO DO:
# IMPORTANT: COME UP WITH A BETTER NUMERICAL INTEGRATION. Its currently producing inaccurate results at large N
# ROMBERG INTEGRATION ISNT WORKING. NEED TO REDO
# 1. Some scipy equivalent methods are not coded yet (e.g. entropy, median)
# 2. Some sort of hypothesis testing to check if the genmaxima distribution has converged to a GEV distribution
#
# NEED A LOT MORE TESTING

class genmaxima():
    r"""Class for generic maxima random variable
    
    **Critical Note**: Numerical instability at large and 0 < N << 1 has been observed. The root cause is due to errors from numerical integration that are used
    to calculate the statistical moments. Please use with caution!. Any suggestions to improve this is more than welcomed!
    
    To circumvent this, it is recommended to convert this distribution to a genxtreme_WR object via get_genxtreme_WR() method. From there, you can
    proceed all statistical analysis again assuming that your distribution has converged. **Note this is only available for parent distributions**
    **with bounded upper bound and Weibull domain of attraction**. It is your responsibility to check the domain of attraction and convergence to GEV distribution
    
    Arguments:
    parent: 
        The parent distribution. You must specify all arguments for the parent at initialisation, as it cannot be changed later on.
    N:
        The block size. inf > N > 0
        Default to 1. Can be changed and set via set_N() method.
    
    """
    
    #Helper methods since not using rv_continuous class from scipy
    def __init__(self, parent, N = 1.0):
        self.N = N
        self.parent = parent
        
    def set_N(self, N):
        self.N = N
        
    def get_N(self):
        return self.N
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_parent(self):
        return self.parent
    
    # Methods similar to scipy, where possible
    def argcheck(self):
        return (self.N > 0) and (self.N < np.inf) and (np.isreal(self.N))
    
    def support(self):
        #Alias for get_support() method
        return self.get_support()
    
    def get_support(self):
        #Should be the same as the parent, although need more study to verify
        return self.parent.support()
    
    def rvs(self, size=1, random_state=None, method="direct", progress_bar=None, max_mem = int(1e8)):
        
        """
        Generate random number. Follows scipy arguments.
        Two different sampling methods, set via 'method' argument:
        1. 'direct': Directly sample from the genmaxima distribution
        2. 'parent': Sample from parent, then take block maxima. Only available for N is integer and N >= 1 
        """
        
        if method == 'direct':
            u = uniform.rvs(size = size, random_state = random_state)
            return self.ppf(u)
        
        elif method == 'parent':
            #Check that N>=1
            if (self.N < 1):
                print("Invalid N for sampling method 'parent'. N must be equal or larger than 1 to use 'parent' sampling method")
                return "nan"
            if not isinstance(self.N, int):
                print("Invalid N for sampling method 'parent'. N must be an integer ('int' instance)")
                return "nan"
            
            max_mem = int(max_mem)
            
            #Determine if we can sample all at once, or need to be looped
            if (self.N * size) > max_mem:
                #Loop sample
                out_arr = np.zeros(size)
                #Determine max number of parallel sampling
                max_parallel_dim = max(int(max_mem/self.N),1)
                #Calculate the amount of loops needed
                num_loops = int(np.ceil(size/max_parallel_dim))
                
                if (progress_bar == False) or (num_loops < 2):
                    for i in range(0,num_loops):
                        #Calculate index to store
                        start_idx = i*max_parallel_dim
                        end_idx = np.min(((i+1)*max_parallel_dim, size))
                        num_parallel_dim = end_idx-start_idx
                        #Sample
                        out_arr[start_idx:end_idx] = np.max(self.parent.rvs(size=(self.N, num_parallel_dim), 
                                                                            random_state=random_state), 
                                                            axis = 0)
                else:
                    from tqdm import tqdm
                    for i in tqdm(range(0,num_loops)):
                        #Calculate index to store
                        start_idx = i*max_parallel_dim
                        end_idx = np.min(((i+1)*max_parallel_dim, size))
                        num_parallel_dim = end_idx-start_idx
                        #Sample
                        out_arr[start_idx:end_idx] = np.max(self.parent.rvs(size=(self.N, num_parallel_dim), 
                                                                            random_state=random_state), 
                                                            axis = 0)
            else:
                out_arr = retake_block_max(self.parent.rvs(size=(self.N*size), random_state=random_state), self.N)
                
            return out_arr
        else:
            print("Invalid 'method' argument specified. Must be either 'direct' or 'parent'")
            return "nan"
            
    
    def pdf(self, x):
        #Assume that if return nan, the pdf is zero
        parent_F = self.parent.cdf(x)
        t2 = np.power(parent_F, (self.N - 1), out = np.zeros_like(x), where = (parent_F != 0)) 
        return self.N * t2 * self.parent.pdf(x)
    
    def logpdf(self, x):
        return np.log(self.N) + (self.N - 1) * self.parent.logcdf(x) + self.parent.logpdf(x)
    
    def cdf(self, x):
        return self.parent.cdf(x) ** self.N
    
    def logcdf(self, x):
        return self.N * self.parent.logcdf(x)
    
    def sf(self, x):
        return 1-self.cdf(x)
    
    def logsf(self, x):
        return np.log(self.sf(x))
    
    def ppf(self, q):
        return self.parent.ppf(q ** (1/self.N))
    
    def isf(self, q):
        #return self.parent.ppf(1 - (q ** (1/self.N)))  <===== NEED TO CHECK IF CORRECT
        print("NOT CODED YET")
        return "nan"
    
    def entropy(self):
        print("NOT CODED YET")
        return "nan"
    
    def fit(self, data):
        print("NOT CODED YET")
        return "nan"
    
    def interval(self, confidence):
        print("NOT CODED YET")
        return "nan"
    
    def expect(self, args=(), lb=None, ub=None, conditional=False, **kwds):
        print("NOT CODED YET")
        return "nan"
    
    #Methods to calculate statistical moments
    #All moments are calculated numerically, so there will be some numerical error
    
    #Method to calculate non-central moment
    def moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        if integral_lb == None:
            integral_lb = self.get_support()[0]
        if integral_ub == None:
            integral_ub = self.get_support()[1]
        
        return numerical_moment(fx = self.pdf,
                                order = order, 
                                integral_lb = integral_lb, 
                                integral_ub = integral_ub, 
                                integral_method = integral_method, 
                                **integral_kwargs)
            
    #Method to calculate standardised moment
    #Unique to this, not availabel in scipy. Maybe better to code the expect() method for this
    def std_moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        if integral_lb == None:
            integral_lb = self.get_support()[0]
        if integral_ub == None:
            integral_ub = self.get_support()[1]
        
        return numerical_standardised_moment(fx = self.pdf,
                                             order = order, 
                                             integral_lb = integral_lb, 
                                             integral_ub = integral_ub, 
                                             integral_method = integral_method, 
                                             **integral_kwargs)
        
    #Method to get statistics, similar to stats() method in scipy distribution
    #Does not behave exactly the same as scipy. Need to code properly.
    def stats(self, moments, **kwargs):
        moments = list(moments)
        for a in moments:
            if a == 'm':
                return self.moment(order = 1, **kwargs)
            elif a == 'v':
                return self.std_moment(order = 2, **kwargs)
            elif a == 's':
                return self.std_moment(order = 3, **kwargs)
            elif a == 'k':
                return self.std_moment(order = 4, **kwargs) - 3.0 #Excess kurtosis
            else:
                print("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")
    
    def mean(self, **kwargs):
        return self.moment(order = 1, **kwargs)
    
    def var(self, **kwargs):
        return self.std_moment(order = 2, **kwargs)
    
    def std(self, **kwargs):
        return np.sqrt(self.var(**kwargs))
    
    def median(self):
        print("NOT CODED YET")
        return "nan"
    
    # Methods to calculate equivalent GEV
    def get_genextreme_parameter(self, parameterisation = "coles"):
        """
        Calculate the "equivalent" GEV parameters
        Based on the mean, variance and upper bound support
        
        **Default return using Coles (2001) parameterisation**
        **Change parameterisation argument to "scipy" to use scipy parameterisation**
        
        For Coles (2001) parameterisation, either see the original textbook, or see Wikipedia on Generalised Extreme Value distribution
        scipy parameterisation has the xi parameter flipped in its sign.
        
        **This method is currently only supported for upper bounded parent distributions**
        **And assumes parent has Weibull domain of attraction**
        
        Will return "nan" for non upper bounded distribution.
        It is your responsibility to check the domain of attraction.
        The method will happily return some numbers even if the domain of attraction is incorrect.
        """
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            dist_mean = self.mean()
            dist_std = self.std()
            dist_zbc = (upper_bound - dist_mean)/dist_std #centered upper bound
            
            xi = xi_from_zbc(dist_zbc, init_guess([dist_zbc]))[0];
            g1 = special.gamma(1-xi)
            kx = (dist_mean - upper_bound)/g1
            sigma = kx*xi
            mu = dist_mean - kx*(g1-1)
            
            if parameterisation == "coles":
                return (mu, sigma, xi);
            elif parameterisation == "scipy":
                return (mu, sigma, -xi);
            else:
                print ("Unsupported GEV parameterisation. Either 'coles' or 'scipy'")
                return ("nan", "nan", "nan");
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            return ("nan", "nan", "nan");
    
    def compare_genextreme_cdf(self, x, scale='linear', custom_scale_fn=None):
        """
        Method to compare CDF to GenExtreme (gev) CDF
        Compare pointwise, given the point(s) in argument 'x'
        
        Return the difference between GEV CDF and genmaxima CDF, i.e.
        return F_{GEV}(x) - F_{genmaxima}(x)
        
        **This method is currently only supported for upper bounded parent distributions**
        **And assumes parent has Weibull domain of attraction**
        
        Will return "nan" for non upper bounded distribution.
        It is your responsibility to check the domain of attraction.
        The method will happily return some numbers even if the domain of attraction is incorrect.
        
        
        Arguments:
        x:
            Points along which the CDF will be compared
        scale:
            The scale in which the distribution will be compared. Supported are:
                + 'linear': comparison of CDF in linear scale (no transformation)
                + 'gumbel': comparison in gumbel probability plot scale, i.e. -ln(-ln(CDF))
                + 'custom': provide your own via the custom_scale_fn argument
        custom_scale_fn:
            The custom scaling function. You MUST SET the 'scale' argument for this function to be used.
        """
        
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            #Get the GEV parameters
            gev_param = self.get_genextreme_parameter(parameterisation='scipy')
            if scale == 'linear':
                return genextreme(loc = gev_param[0], scale = gev_param[1], c = gev_param[2]).cdf(x) - self.cdf(x)
            elif scale == 'gumbel':
                return -np.log(-np.log(genextreme(loc = gev_param[0], scale = gev_param[1], c = gev_param[2]).cdf(x))) - -np.log(-np.log(self.cdf(x)))
            elif scale == 'custom':
                return custom_scale_fn(genextreme(loc = gev_param[0], scale = gev_param[1], c = gev_param[2]).cdf(x)) - custom_scale_fn(self.cdf(x))
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            return "nan";
    
    #Method to return reparameterised GEV distribution with Weivbull domain of attraction (genextreme_WR object)
    def get_genextreme_WR_dist(self, warning = True):
        #
        # TO DO:
        # Should check for convergence here
        #
        if warning:
            print("Warning: returned GEV distribution assumes 1) Convergence, 2) Weibull domain of attraction")
            print("It is your responsibility to check both of these assumptions")
        
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            return genextreme_WR(E = self.mean(), V = self.var(), z_b = self.support()[1])
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            print("Please also check the domain of attraction!")
            return "nan";
    
    #Method to return scipy genextreme distribution (scipy.stats.genextreme object)
    def get_genextreme_dist(self, warning = True):
        #
        # TO DO:
        # Should check for convergence here
        #
        
        if warning:
            print("Warning: returned GEV distribution assumes 1) Convergence, 2) Weibull domain of attraction")
            print("It is your responsibility to check both of these assumptions")
        
        upper_bound = self.get_support()[1]
        if (upper_bound < np.inf) and np.isreal(upper_bound):
            param = self.get_genextreme_parameter(parameterisation='scipy');
            return genextreme(loc = param[0], scale = param[1], c = param[2])
        else:
            print("Unsupported parent distribution. Only support parent with bounded upper bound.")
            print("Please also check the domain of attraction!")
            return "nan";

class GenextremeWeibull():
    """
    Reparameterised GEV Distribution, Strictly for Weibull Domain of Attraction
    Reparameterised genextreme distribution (GEV distribution), based on mean (E), variance(V), and upper bound (z_b).
    Allows the calculation of all statistical properties (mean, var, CDF, etc.) when raised to the power of N.
    That is equivalent to taking the maxima from a GEV distribution with block size = N.
    Still allows the regular parameterisation too.
    
    **Only support Weibull Domain of Attraction**
    i.e. z_b < infinity and xi < 0
    
    Note on the definition of N:
    The larger the N, the larger the size of the block is. For instance if you have 10 million samples, and then
    you get the maxima every 1,000 samples, i.e. now you have 10,000 samples of the maxima, then N = 1,000
    Note that N need not to be an integer. The mathematics works for non integer N. e.g. if you have maxima data only,
    each with block size 1,000, you could theoretically find equivalent GEV distributions of sample with block maxima of 
    10 by setting N = 10/1,000 = 0.01. Note that there are caveats to this, and the interpretation is up to you as the user.
    """
    def __init__(self, E = None, V = None, z_b = None, mu  = None, sigma = None, xi = None, parameterisation = 'coles'):
        #For the purpose of all calculation shall be done in coles parameterisation. Output can be specified to be in scipy if desired by the user.        
        self.parameterisation = parameterisation
        if (not (E==None)) and (not (V==None)) and (not (z_b==None)):
            self.E = E
            self.V = V
            self.z_b = z_b
            og_param = self.calculate_og_base_parameter()
            self.mu = og_param[0]
            self.sigma = og_param[1]
            self.xi = og_param[2]
        elif (not (mu==None)) and (not (sigma==None)) and (not (xi==None)):
            self.mu = mu
            self.sigma = sigma
            if parameterisation == 'coles':
                self.xi = xi
            elif parameterisation == 'scipy':
                self.xi = -xi
            else:
                print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
            self.E = self.mean()
            self.V = self.var()
            self.z_b = self.get_support()[1]
        else:
            print("Error: Either 'E', 'V' and 'z_b' must be supplied, or 'mu', 'sigma' and 'xi'")
        
        if not self.argcheck():
            print("Error parameter out of bounds")
            
    
    def argcheck(self):
        return ((self.sigma > 0) \
                and (self.xi < 0) \
                and (self.V > 0) \
                and (self.z_b < np.inf) \
                and (self.z_b > -np.inf) \
                and np.isreal(self.z_b) \
                and (self.z_b > self.E))
    
    def support(self):
        #Alias for get_support
        return self.get_support()
    
    def get_support(self):
        return genextreme(loc = self.mu, scale = self.sigma, c = -self.xi).support()
    
    def calculate_og_base_parameter(self, parameterisation = 'coles'):
        z_bc = (self.z_b - self.E) / np.sqrt(self.V)
        xi = xi_from_zbc(z_bc, init_guess([z_bc]))[0]
        g1 = special.gamma(1-xi)
        kx = (self.E - self.z_b)/g1
        sigma = kx * xi
        mu = self.E - kx * (g1-1)
        if parameterisation == 'coles':
            return (mu, sigma, xi)
        elif parameterisation == 'scipy':
            return (mu, sigma, -xi)
        else:
            print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
            return ("nan", "nan", "nan")
    
    #All the methods from this point allows raising of the distribution to the power of N, for some N>0
    #This is equivalent to taking the maxima of sample with block size = N
    #Based on C. Caprani (2004)
    
    def get_parameter(self, N=1, method='og'):
        """
        Method to get the parameter mean, variance and upper bound (E, V, and z_b)
        Raised to the power of N. Default N to 1.
        
        'method' argument can be either 'og' or 'direct'.
        'og' calculates the parameter via mu, sigma and xi. Ref: Caprani, 2004. Coles, 2001.
        'direct' calculates the parameters directly.
        Either method in theory should produce the same results, barring some floating point error.
        Use method check_get_parameter() to check if both methods agree, and the difference in them.
        """
        N = 1/N #Invert input N to mathematical N. Same with all the rest of the methods.
        if N==1:
            return (self.E, self.V, self.z_b)
        elif N>0:
            if method == 'og':
                raised_og_parameter = self.get_og_parameter(N=1/N) #Note on inversion of N due the difference in definition
                mu_N = raised_og_parameter[0]
                sigma_N = raised_og_parameter[1]

                E_N = mu_N + (sigma_N/self.xi) * (special.gamma(1-self.xi) - 1)
                V_N = np.abs((sigma_N**2/self.xi**2) * (special.gamma(1-2*self.xi) - special.gamma(1-self.xi)**2))
            elif method == 'direct':
                E_N = self.E / (N ** self.xi) - self.z_b * ( 1/(N ** self.xi) - 1 )
                V_N = self.V / (N ** (2*self.xi))
            else:
                print("Invalid argument 'method'. Should be either 'og' or 'direct'")
                return ("nan", "nan", "nan")
            
            return (E_N, V_N, self.z_b)
        else:
            print("Invalid argument 'N'. 'N' must be larger than 0")
            return ("nan", "nan", "nan")
        
    def check_get_parameter(self, N = 1):
        """
        Method to check the difference between 'og' and 'direct' in get_parameter() method
        Return two tuples. First tuple is the difference, second tuple is if they are within machine tolerance using numpy isclose() function.
        """
        og = self.get_parameter(N=N, method='og')
        direct = self.get_parameter(N=N, method='direct')
        diff = ( og[0] - direct[0], og[1] - direct[1], og[2] - direct[2] )
        close = ( np.isclose(og[0], direct[0]), np.isclose(og[1], direct[1]), np.isclose(og[2], direct[2]) )
        return (diff, close)
        
                         
    
    def get_og_parameter(self, N=1, parameterisation = 'coles'):
        #Note that the mathematical definition and the definition used in this function is inverted. So
        # the input N = 1 / mathematical N
        N = 1/N #Invert input N to mathematical N.
        if N==1:
            if parameterisation == 'coles':
                return (self.mu, self.sigma, self.xi)
            elif parameterisation == 'scipy':
                return (self.mu, self.sigma, -self.xi)
            else:
                print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
                return ("nan", "nan", "nan")
        elif N>0:
            mu_N = self.mu - (self.sigma/self.xi)*(1 - N**(-self.xi))
            sigma_N = self.sigma * N**(-self.xi)
            if parameterisation == 'coles':
                return (mu_N, sigma_N, self.xi)
            elif parameterisation == 'scipy':
                return (mu_N, sigma_N, -self.xi)
            else:
                print("Error 'parameterisation' argument must be either 'coles' or 'scipy'")
                return ("nan", "nan", "nan")
        else:
            print("Invalid argument 'N'. 'N' must be larger than 0")
            return ("nan", "nan", "nan")
    
    def rvs(self, size=1, random_state=None, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).rvs(size=size, random_state=random_state)
        
    def pdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).pdf(x)
    
    def logpdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).logpdf(x)
    
    def cdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).cdf(x)
    
    def logcdf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).logcdf(x)
    
    def sf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).sf(x)
    
    def logsf(self, x, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).logsf(x)
    
    def ppf(self, q, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).ppf(q)
    
    def isf(self, q, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).isf(q)
    
    def moment(self, order, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).moment(order=order)
    
    def stats(self, moments='mv', N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).stats(moments=moments)
    
    def entropy(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).entropy()
    
    def fit(self, data, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).fit(data)
    
    def expect(self, func, args=(), lb=None, ub=None, conditional=False, N=1, **kwds):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).expect(
            func = func, 
            args = args, 
            lb = lb, 
            ub = ub, 
            conditional = conditional, 
            kwds = kwds,
        )
    
    def median(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).median()
    
    def mean(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).mean()
    
    def var(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).var()
    
    def std(self, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).std()
    
    def interval(self, confidence, N=1):
        param = self.get_og_parameter(N=N, parameterisation = 'scipy')
        return genextreme(loc = param[0], scale = param[1], c = param[2]).interval(confidence=confidence)
    

def genextreme_weibull(mu = None, sigma = None, xi = None, E = None, V = None, zb = None, parameterization = 'coles'):
    """
    Genextreme distribution with Weibull domain of attraction.
    This can take any three parameters from mu, sigma, xi, E, V, zb.

    Arguments:
    ----------
    mu: float
        Location parameter
    sigma: float
        Scale parameter
    xi: float
        Shape parameter
    E: float
        Mean of the distribution
    V: float
        Variance of the distribution
    zb: float
        Upper bound of the distribution
    parameterization: str
        The parameterization of the distribution. Either 'coles' or 'scipy'
        'coles' is the parameterization used in Coles (2001)
        'scipy' is the parameterization used in scipy.stats.genextreme
        Note that the xi parameter is flipped in sign between the two parameterization

    Returns:
    --------
    A GenextremeWeibull distribution object.
    See GenextremeWeibull class for more details.
    """
    args = (mu, sigma, xi, E, V, zb)
    args_name = ('mu', 'sigma', 'xi', 'E', 'V', 'zb')
    #At least three args must be supplied
    if sum([x is not None for x in args]) != 3:
        raise ValueError('Exactly three arguments must be supplied')

    #Get the not None args as dict
    args = {args_name[i]: x for i, x in enumerate(args) if x is not None}

    if parameterization not in ["scipy", "coles"]:
        raise ValueError("parameterization must be either 'scipy' or 'coles'")
    if parameterization == "scipy" and "xi" in args:
        args["xi"] = -args["xi"]

    #Check args
    if "mu" in args:
        if not np.isfinite(args['mu']):
            raise ValueError('mu must be finite')
    if "sigma" in args:
        if args["sigma"] <=0 or not np.isfinite(args['sigma']):
            raise ValueError('sigma must be strictly positive and finite')
    if "xi" in args:
        if args["xi"] >=0 or not np.isfinite(args['xi']):
            raise ValueError('xi must be strictly negative and finite')
    if "E" in args:
        if not np.isfinite(args['E']):
            raise ValueError('E must be finite')
    if "V" in args:
        if (args['V'] <= 0 or not np.isfinite(args['V'])):
            raise ValueError('V must be strictly positive and finite')
    if "zb" in args:
        if not np.isfinite(args['zb']):
            raise ValueError('zb must be finite')
        if "E" in args:
            if args['zb'] <= args['E']:
                raise ValueError('zb must be larger than E')
        
    
    if "mu" in args and "sigma" in args and "xi" in args:
        return GenextremeWeibull(mu=mu, sigma=sigma, xi=xi)
    
    elif "E" in args and "V" in args and "zb" in args:
        return GenextremeWeibull(E=E, V=V, z_b=zb)
    
    #Anything else must be converted to the other parameterization
    elif "E" in args and "zb" in args and "xi" in args:
        #Compute zbc
        zbc = zbc_from_xi(xi)
        V = ((zb-E)/zbc)**2
        return GenextremeWeibull(E=E, V=V, z_b=zb)
    
    elif "E" in args and "V" in args and "xi" in args:
        zbc = zbc_from_xi(xi)
        return GenextremeWeibull(E=E, V=V, z_b=E + zbc*np.sqrt(V))
    
    else:
        raise NotImplementedError("This combination of arguments is not implemented yet")


genextreme_WR = genextreme_weibull
"""
legacy class name of Genextreme with Weibull domain of attraction. 
This is the same as calling genextreme_weibull.
"""


from scipy.stats import beta as beta_dist

def beta(a=None, b=None, mode=None, cov=None, loc=0.0, scale=1.0):
    """
    Beta distribution.
    Allows 2 paramterization:
        - (alpha, beta) -  this is the common shape parameterization
        - (mode, cov) - Parameterize based on the mode and coefficient of variation

    Arguments:
    ----------
    a: float
        alpha parameter
    b: float
        beta parameter
    mode: float
        mode of the distribution
    cov: float
        coefficient of variation
    loc: float
        location parameter. Default 0.0.
    scale: float
        scale parameter. Default 1.0.
        
    Notes:
    ------
    Either alpha and beta must be supplied together, or mode and cov must be supplied together, but not both.

    For the (mode, cov) parameterization, loc < mode < scale+loc.
    The distribution only admits the unimodal case, i.e. alpha, beta > 1.

    """
    # Check that alpha and beta are in pair
    if a is not None or b is not None:
        if b is None and a is None:
            raise ValueError("alpha and beta must be supplied together")
        if mode is not None or cov is not None:
            raise ValueError("Only one of (alpha, beta) or (mode, cov) must be supplied")
        else:
            return beta_dist(a, b, loc=loc, scale=scale)
    elif mode is not None or cov is not None:
        if cov is None and mode is None:
            raise ValueError("mode and cov must be supplied together")
        if a is not None or b is not None:
            raise ValueError("Only one of (alpha, beta) or (mode, cov) must be supplied")
        else:
            if cov <= 0:
                raise ValueError("cov must be positive")
            if mode <= loc or mode >= scale + loc:
                raise ValueError("mode must be within the support")
            
            C = cov
            M = mode
            s = scale
            t = loc

            a3 = - s*(C*M)**2/(t-M)**3

            a2_num = C**2*M*(M * (-3*M - t + s) + 2*t * (s + 2*t)) + (t-M)**2*(t-M + s)
            a2_denom = (t-M)**3
            a2 = a2_num/a2_denom

            a1_nom = -(2*(t-M) + s)* (C**2*t * (2*M* (2*t-3*M + s) + t*(s + 2*t)) + s*(t-M)**2)
            a1_denom = s*(t-M)**3
            a1 = a1_nom / a1_denom

            a0_num = (C*t)**2 * ( 3*(t-M) + s) * (2*(t-M) + s)**2
            a0_denom = (s**2*(t-M)**3)
            a0 = a0_num/a0_denom

            a = np.roots([a3, a2, a1, a0])
            a = np.max(a[np.isreal(a)])
            b = (s*(a-1) - (M-t)*(a-2)) / (M-t)

            return beta_dist(a, b, loc=loc, scale=scale)



class mixturedistribution():
    """
    Mixture distribution class.
    This is a general class that can be used to construct (almsot) any mixture distributions
    """
    def __init__(self, distributions, weight):
        """
        Arguments:
        ----------
        distributions: list of scipy distribution objects (or objects that support scipy distribution methods)
            The list of distributions to be mixed
        weight: list of float
            The weight of each distribution. Must sum to 1
        """
        self._distributions = np.array(distributions).flatten()
        self._weight = np.array(weight).flatten()
        self._nmode = len(self._distributions)

        if len(self._distributions) != len(self._weight):
            raise ValueError("The number of distributions and weight must be the same")
        
        if np.sum(self._weight) != 1:
            raise ValueError("The sum of weight must be 1")

    def support(self):
        return self.get_support()

    def get_support(self):
        return (np.min([d.support()[0] for d in self._distributions]), np.max([d.support()[1] for d in self._distributions]))

    def rvs(self, size=1, random_state=None, mixed = True):
        """
        Return random sample
        Arguments:
        ----------
        size: int
            The number of samples to return
        random_state: int
            The random seed to use
        mixed: bool
            If True, return mixed sample.
            If False, return the samples separated by the mode.
            The returned sample will be a list with size equal to the number of distribution. 
            Each list is a sample from each mode.
            Note that the sample is flattened and the size is not retained when mixed is False.
        """
        if mixed:
            return np.vectorize(lambda p: self._distributions[p].rvs(random_state=random_state))(
                np.random.default_rng(seed=random_state).choice(self._nmode, p = self._weight, size = size)
            )
        else:
            mode_sample = np.random.default_rng(seed=random_state).choice(self._nmode, p = self._weight, size = size)
            sample = [None] * self._nmode
            for n in range(self._nmode):
                sample[n] = self._distributions[n].rvs(np.sum(mode_sample == n), random_state = random_state)
            return sample

    def pdf(self, x):
        return np.sum(np.array([_d.pdf(x) * self._weight[i] for i, _d in enumerate(self._distributions)]), axis = 0)

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        return np.sum(np.array([_d.cdf(x) * self._weight[i] for i, _d in enumerate(self._distributions)]), axis = 0)

    def logcdf(self, x):
        return np.log(self.cdf(x))

    def sf(self, x):
        return 1 - self.cdf(x)

    def logsf(self, x):
        return np.log(self.sf(x))

    def ppf(self, q):
        raise NotImplementedError("ppf is not implemented yet")

    def isf(self, q):
        raise NotImplementedError("isf is not implemented yet")

    def moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        if integral_lb == None:
            integral_lb = self.get_support()[0]
        if integral_ub == None:
            integral_ub = self.get_support()[1]
        
        return numerical_moment(fx = self.pdf,
                                order = order, 
                                integral_lb = integral_lb, 
                                integral_ub = integral_ub, 
                                integral_method = integral_method, 
                                **integral_kwargs)

    #Method to calculate standardised moment
    #Unique to this, not availabel in scipy. Maybe better to code the expect() method for this
    def std_moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
        #Auto set bound if not specified to distribution bounds
        if integral_lb == None:
            integral_lb = self.get_support()[0]
        if integral_ub == None:
            integral_ub = self.get_support()[1]
        
        return numerical_standardised_moment(fx = self.pdf,
                                             order = order, 
                                             integral_lb = integral_lb, 
                                             integral_ub = integral_ub, 
                                             integral_method = integral_method, 
                                             **integral_kwargs)

    #Method to get statistics, similar to stats() method in scipy distribution
    #Does not behave exactly the same as scipy. Need to code properly.
    def stats(self, moments, **kwargs):
        moments = list(moments)
        for a in moments:
            if a == 'm':
                return self.moment(order = 1, **kwargs)
            elif a == 'v':
                return self.std_moment(order = 2, **kwargs)
            elif a == 's':
                return self.std_moment(order = 3, **kwargs)
            elif a == 'k':
                return self.std_moment(order = 4, **kwargs) - 3.0 #Excess kurtosis
            else:
                print("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")

    def entropy(self, **kwargs):
        raise NotImplementedError("Entropy is not implemented yet")

    def mean(self, **kwargs):
        return self.moment(order = 1, **kwargs)

    def var(self, **kwargs):
        return self.std_moment(order = 2, **kwargs)
    
    def std(self, **kwargs):
        return np.sqrt(self.var(**kwargs))
    
    def median(self):
        raise NotImplementedError("Median is not implemented yet")
    

def multimodalbeta(alpha, beta, weight):
    """
    Multimodal Beta distribution

    Arguments:
    ----------
    alpha: list of float
        The alpha parameter of each beta distribution
    beta: list of float
        The beta parameter of each beta distribution
    weight: list of float
        The weight of each beta distribution. Must sum to 1

    Note that alpha, beta and weight must be the same length.
    """
    #Check length of alpha, beta and weight are the same
    if len(alpha) != len(beta) or len(alpha) != len(weight):
        raise ValueError("Length of alpha, beta and weight must be the same")
    return mixturedistribution(
        distributions = [scipy_beta_dist(alpha[i], beta[i]) for i in range(len(alpha))],
        weight = weight
    )


# def _t(x, E, zb, xi):
#     """
#     GEV t(x) function, given mean, upper bound, and shape.
#     Uses Coles parameterization, i.e., xi < 0 for Weibull, always.

#     This function can compute for any size of x.
#     Note this can only process one parameter set at a time. Use list comprehension or numpy vectorization to evaluate multiple parameter sets.
    

#     Parameters
#     ----------
#     x : array_like
#         Value to evaluate the function at
#     E : float
#         Mean of the distribution
#     zb : float
#         Upper bound of the distribution
#     xi : float
#         Shape parameter of the distribution. Follows Coles (2001) parameterization.
#     """
#     if np.any(xi >= 0):
#         raise ValueError("xi must be strictly negative (Weibull).")
#     ndim = np.array(x).ndim
#     if ndim > 0:
#         out = np.zeros_like(x)
#         #Where x < zb, return the calculated
#         out[x < zb] = (scipy_gamma_special(1-xi) * (zb-x[x < zb]) / (zb-E))**(-1/xi)
#     else:
#         out = 0 if x >= zb else (scipy_gamma_special(1-xi) * (zb-x) / (zb-E))**(-1/xi)
#     return out

# class CumulativeDistributionStatistics():
#     """
#     Cumulative Distribution Statistics distribution.
#     After Caprani, C.C. (2005)

#     Parameters
#     ----------
#     E : array_like
#         Mean of individual GEV distribution
#     zb : array_like
#         Upper bound of individual GEV distribution
#     xi : array_like
#         Shape parameter of individual GEV distribution. Follows Coles (2001) parameterization.
#         It is the negative of the shape parameter in scipy.stats.genextreme.
#         Must be strictly negative (Weibull).

#     Notes
#     -----
#     - The PDF and CDF have been verified with numerical gradient.
#     - The PPF and random number generator uses numerical solver. Performance is okay, at less than 3s for 1 million samples.
#     - The moments are all numerical moments.

#     """
#     def __init__(self, E,  zb, xi):
#         self._E = np.array(E)
#         self._zb = np.array(zb)
#         self._xi = np.array(xi)

#         if not self.argcheck():
#             raise ValueError("Error parameter out of bounds. Make sure all E < zb and xi < 0.")

#         self._params = list(zip(E, zb, xi))
#         self._support = (-np.inf, max(zb)) 
#         self._init_guess = self.mean() #For ppf solver

#     def argcheck(self):
#         return not (np.any(self._xi >= 0) or np.any(self._zb - self._E <= 0))

#     @property
#     def support(self):
#         return self._support
    
#     @property
#     def params(self):
#         return self._params
    
#     @property
#     def E(self):
#         return self._E
    
#     @property
#     def zb(self):
#         return self._zb
    
#     @property
#     def xi(self):
#         return self._xi
    
#     def get_support(self):
#         return self.support
    
#     def rvs(self, init_guess = None, *args, **kwargs):
#         u = scipy_uniform_dist.rvs(*args, **kwargs)
#         return self.ppf(q=u, init_guess=init_guess)

#     def logcdf(self, x):
#         return np.sum([-_t(x, *param) for param in self.params], axis = 0)

#     def cdf(self, x):
#         return np.exp(self.logcdf(x))
    
#     def _pdfterm(self, x, E, zb, xi):
#         #The term for logcdf function, that goes into the summation
#         #Just separating it so computation can be improved, if possible
#         #for any zb, lim x -> zb should reach 0, though this need to be checked more rigurously
#         ndim = np.array(x).ndim
#         if ndim > 0:
#             out = np.zeros_like(x)
#             filt = x < zb
#             out[filt] = -_t(x[filt], E, zb, xi)/(xi * (zb - x[filt]))
#         else:
#             out = 0 if x >= zb else -_t(x, E, zb, xi)/(xi * (zb - x))
#         return out

#     def logpdf(self, x):
#         #return np.log(np.sum([self._lcdfterm(x, *param) for param in self.params], axis = 0)) + self.logcdf(x) #Less stable for pdf calculation, as np.log(0) is possible
#         return np.log(self.pdf(x))

#     def pdf(self, x):
#         #return np.exp(self.logpdf(x))
#         return np.sum([self._pdfterm(x, *param) for param in self.params], axis = 0) * self.cdf(x)
    
#     def sf(self, x):
#         return 1-self.cdf(x)
    
#     def logsf(self, x):
#         return np.log(self.sf(x))
    
#     def ppf(self, q, init_guess=None):
#         # Use newton solver to solve for the CDF
#         # Performance is okay, it took less than 3s to solve 1 million points
#         init_guess = self._init_guess if init_guess is None else init_guess
#         return newton(lambda x: self.cdf(x) - q, np.ones_like(q) * init_guess, fprime=self.pdf)
    
#     def isf(self, q):
#         return self.ppf(1-q)
    
#     def entropy(self, N=1):
#         raise NotImplementedError("Not implemented yet")
    
#     def fit(self, data, N=1):
#         raise NotImplementedError("Not implemented yet")
    
#     def expect(self, *args, **kwargs):
#         raise NotImplementedError("Not implemented yet")
    
#     def median(self):
#         raise NotImplementedError("Not implemented yet")
    
#     #Method to calculate non-central moment
#     def moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
#         #Auto set bound if not specified to distribution bounds
#         if integral_lb == None:
#             integral_lb = self.get_support()[0]
#         if integral_ub == None:
#             integral_ub = self.get_support()[1]
        
#         return numerical_moment(fx = self.pdf,
#                                 order = order, 
#                                 integral_lb = integral_lb, 
#                                 integral_ub = integral_ub, 
#                                 integral_method = integral_method, 
#                                 **integral_kwargs)
            
#     #Method to calculate standardised moment
#     #Unique to this, not availabel in scipy. Maybe better to code the expect() method for this
#     def std_moment(self, order, integral_method="quad", integral_lb = None, integral_ub = None, **integral_kwargs):
#         #Auto set bound if not specified to distribution bounds
#         if integral_lb == None:
#             integral_lb = self.get_support()[0]
#         if integral_ub == None:
#             integral_ub = self.get_support()[1]
        
#         return numerical_standardised_moment(fx = self.pdf,
#                                              order = order, 
#                                              integral_lb = integral_lb, 
#                                              integral_ub = integral_ub, 
#                                              integral_method = integral_method, 
#                                              **integral_kwargs)
        
#     #Method to get statistics, similar to stats() method in scipy distribution
#     #Does not behave exactly the same as scipy. Need to code properly.
#     def stats(self, moments, **kwargs):
#         moments = list(moments)
#         for a in moments:
#             if a == 'm':
#                 return self.moment(order = 1, **kwargs)
#             elif a == 'v':
#                 return self.std_moment(order = 2, **kwargs)
#             elif a == 's':
#                 return self.std_moment(order = 3, **kwargs)
#             elif a == 'k':
#                 return self.std_moment(order = 4, **kwargs) - 3.0 #Excess kurtosis
#             else:
#                 print("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")

    
#     def mean(self, **kwargs):
#         return self.moment(order = 1, **kwargs)
    
#     def var(self, **kwargs):
#         return self.std_moment(order = 2, **kwargs)
    
#     def std(self, **kwargs):
#         return np.sqrt(self.var(**kwargs))
    
#     def median(self):
#         print("NOT CODED YET")
#         return "nan"
