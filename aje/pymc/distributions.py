from typing import List, Tuple, Union
import numpy as np
from scipy import stats
import pytensor.tensor as at
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.var import TensorVariable
from pymc.pytensorf import floatX
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.continuous import BoundedContinuous, Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc import Interpolated
from aje.stats.distributions.cls.cds import cds
from aje.stats.distributions.pytensor_methods.cds import logpdf, logcdf

__all__ = ["GenExtremeWeibull", "GenExtreme", "CDStats", "from_posterior"]

def from_posterior(param, samples, lower = None, upper = None, npts = 100):
    """
    from_posterior

    Estimate a distribution using Kernel Density Estimate (KDE).
    Useful for updating prior or other cases when no analytical solution is known.
    Note that the function linearly extrapolates beyond the sample, up to the specified `upper` and `lower` argument
    Arguments:
    ----------
        name:
            Name of the variable, must be of type <str>
        samples:
            Sample of prior/posterior from the trace
        lower:
            Lower bound of the KDE. If 'None' defaults to sample min - 3 x sample width.
        upper:
            Upper bound of the KDE. If 'None' defaults to sample max + 3 x sample width.
        npts:
            Number of KDE points to be returned as Interpolated object. Defaults to 100.
    Return:
    -------
        pymc.Interpolated object
    """
    smin, smax = np.min(samples), np.max(samples)
    #Check lower and upper is not within the sample
    if not lower is None:
        if lower > smin:
            raise ValueError("ERR: 'lower' argument is larger than the sample minima")
    
    if not upper is None:
        if upper < smax:
            raise ValueError("ERR: 'upper' argument is smaller than the sample maxima")

    width = smax - smin
    x = np.linspace(smin, smax, npts)
    y = stats.gaussian_kde(samples)(x)
    

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    lower_x = x[0] - 3 * width if lower is None else lower
    upper_x = x[-1] + 3 * width if upper is None else upper

    x = np.concatenate([[lower_x], x, [upper_x]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)


#=========================================================
#             GenExtremeWeibull distribution
#=========================================================

class GenExtremeWeibullRV(RandomVariable):
    """
    Generalized Extreme Value Distribution, for Weibull domain of attraction ONLY.
    """
    name: str = "Generalized Extreme Value"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("Generalized Extreme Value", "\\operatorname{GEV}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, z_b = 1.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, z_b, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: Union[np.random.RandomState, np.random.Generator],
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        z_b: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        # Notice negative here, since remainder of GenExtreme is based on Coles parametrization
        return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng, size=size)

gevw = GenExtremeWeibullRV()

class GenExtremeWeibull(BoundedContinuous):
    r"""
        Univariate Generalized Extreme Value log-likelihood

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \exp\left[ -\left(1 + \xi z\right)^{-\frac{1}{\xi}} \right]

    where

    .. math::

        z = \frac{x - \mu}{\sigma}

    and is defined on the set:

    .. math::

        \left\{x: 1 + \xi\left(\frac{x-\mu}{\sigma}\right) > 0 \right\}.

    Note that this parametrization is per Coles (2001), and differs from that of
    Scipy in the sign of the shape parameter, :math:`\xi`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        sigmas = [2., 2., 4.]
        xis = [-0.3, 0.0, 0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \in [\mu - \sigma/\xi, +\infty]`, when :math:`\xi > 0`
              * :math:`x \in \mathbb{R}` when :math:`\xi = 0`
              * :math:`x \in [-\infty, \mu - \sigma/\xi]`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma(g_1 - 1)/\xi`, when :math:`\xi \neq 0, \xi < 1`
              * :math:`\mu + \sigma \gamma`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 1`
                where :math:`\gamma` is the Euler-Mascheroni constant, and
                :math:`g_k = \Gamma (1-k\xi)`
    Variance  * :math:`\sigma^2 (g_2 - g_1^2)/\xi^2`, when :math:`\xi \neq 0, \xi < 0.5`
              * :math:`\frac{\pi^2}{6} \sigma^2`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu: float
        Location parameter.
    sigma: float
        Scale parameter (sigma > 0).
    xi: float
        Shape parameter
    scipy: bool
        Whether or not to use the Scipy interpretation of the shape parameter
        (defaults to `False`).

    References
    ----------
    .. [Coles2001] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    """

    rv_op = gevw
    bound_args_indices = (None, 6) 

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, z_b = 1, scipy=False, **kwargs):
        # If SciPy, use its parametrization, otherwise convert to standard
        if scipy:
            xi = -xi
        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))
        xi = at.as_tensor_variable(floatX(xi))
        z_b = at.as_tensor_variable(floatX(z_b))

        return super().dist([mu, sigma, xi, z_b], **kwargs)

    def logp(value, mu, sigma, xi, z_b):
        """
        Calculate log-probability of Generalized Extreme Value distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = at.switch(
            at.isclose(xi, 0),
            -at.log(sigma) - scaled - at.exp(-scaled),
            -at.log(sigma)
            - ((xi + 1) / xi) * at.log1p(xi * scaled)
            - at.pow(1 + xi * scaled, -1 / xi),
        )

        #logp_expression = at.switch(at.gt(logp_expression, -np.inf), logp_expression, np.log(1e-323))

        logp = at.switch(at.gt(1 + xi * scaled, 0.0), logp_expression, -np.inf)
        logp = at.switch(at.gt(logp, -np.inf), logp, np.log(1e-323))
        #logp = at.switch(at.gt(value, mu-(sigma/xi)), -np.inf, logp)
        
        return check_parameters(
            logp, sigma > 0, at.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def logcdf(value, mu, sigma, xi, z_b):
        """
        Compute the log of the cumulative distribution function for Generalized Extreme Value
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        logc_expression = at.switch(
            at.isclose(xi, 0), -at.exp(-scaled), -at.pow(1 + xi * scaled, -1 / xi)
        )

        logc = at.switch(1 + xi * (value - mu) / sigma > 0, logc_expression, -np.inf)

        return check_parameters(
            logc, sigma > 0, at.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def moment(rv, size, mu, sigma, xi, z_b):
        r"""
        Using the mode, as the mean can be infinite when :math:`\xi > 1`
        """
        mode = at.switch(at.isclose(xi, 0), mu, mu + sigma * (at.pow(1 + xi, -xi) - 1) / xi)
        if not rv_size_is_none(size):
            mode = at.full(size, mode)
        return mode


from pymc.distributions.transforms import _default_transform
from pymc.distributions.continuous import bounded_cont_transform

@_default_transform.register(GenExtremeWeibull)
def gen_extreme_weibull_default_transform(op, rv):
    return bounded_cont_transform(op, rv, GenExtremeWeibull.bound_args_indices)

#=========================================================
#             GenExtreme distribution
#=========================================================

class GenExtremeRV(RandomVariable):
    """
    Generalized Extreme Value Distribution, for Weibull, Gumbel and Frechet domain of attraction.
    """
    name: str = "Generalized Extreme Value"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("Generalized Extreme Value", "\\operatorname{GEV}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: Union[np.random.RandomState, np.random.Generator],
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        # Notice negative here, since remainder of GenExtreme is based on Coles parametrization
        return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng, size=size)


gev = GenExtremeRV()


class GenExtreme(Continuous):
    r"""
    Univariate Generalized Extreme Value log-likelihood

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \exp\left[ -\left(1 + \xi z\right)^{-\frac{1}{\xi}} \right]

    where

    .. math::

        z = \frac{x - \mu}{\sigma}

    and is defined on the set:

    .. math::

        \left\{x: 1 + \xi\left(\frac{x-\mu}{\sigma}\right) > 0 \right\}.

    Note that this parametrization is per Coles (2001), and differs from that of
    Scipy in the sign of the shape parameter, :math:`\xi`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        sigmas = [2., 2., 4.]
        xis = [-0.3, 0.0, 0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \in [\mu - \sigma/\xi, +\infty]`, when :math:`\xi > 0`
              * :math:`x \in \mathbb{R}` when :math:`\xi = 0`
              * :math:`x \in [-\infty, \mu - \sigma/\xi]`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma(g_1 - 1)/\xi`, when :math:`\xi \neq 0, \xi < 1`
              * :math:`\mu + \sigma \gamma`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 1`
                where :math:`\gamma` is the Euler-Mascheroni constant, and
                :math:`g_k = \Gamma (1-k\xi)`
    Variance  * :math:`\sigma^2 (g_2 - g_1^2)/\xi^2`, when :math:`\xi \neq 0, \xi < 0.5`
              * :math:`\frac{\pi^2}{6} \sigma^2`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    xi : float
        Shape parameter
    scipy : bool
        Whether or not to use the Scipy interpretation of the shape parameter
        (defaults to `False`).

    References
    ----------
    .. [Coles2001] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    """

    rv_op = gev

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, scipy=False, **kwargs):
        # If SciPy, use its parametrization, otherwise convert to standard
        if scipy:
            xi = -xi
        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))
        xi = at.as_tensor_variable(floatX(xi))

        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        """
        Calculate log-probability of Generalized Extreme Value distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Pytensor tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = at.switch(
            at.isclose(xi, 0),
            -at.log(sigma) - scaled - at.exp(-scaled),
            -at.log(sigma)
            - ((xi + 1) / xi) * at.log1p(xi * scaled)
            - at.pow(1 + xi * scaled, -1 / xi),
        )

        logp = at.switch(at.gt(1 + xi * scaled, 0.0), logp_expression, -np.inf)

        return check_parameters(
            logp, sigma > 0, at.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def logcdf(value, mu, sigma, xi):
        """
        Compute the log of the cumulative distribution function for Generalized Extreme Value
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        logc_expression = at.switch(
            at.isclose(xi, 0), -at.exp(-scaled), -at.pow(1 + xi * scaled, -1 / xi)
        )

        logc = at.switch(1 + xi * (value - mu) / sigma > 0, logc_expression, -np.inf)

        return check_parameters(
            logc, sigma > 0, at.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def moment(rv, size, mu, sigma, xi):
        r"""
        Using the mode, as the mean can be infinite when :math:`\xi > 1`
        """
        mode = at.switch(at.isclose(xi, 0), mu, mu + sigma * (at.pow(1 + xi, -xi) - 1) / xi)
        if not rv_size_is_none(size):
            mode = at.full(size, mode)
        return mode

class CDStatsRV(RandomVariable):
    """
    Cumulative Distribution Statistics Distribution. After Caprani, C.C. (2005).
    With E, zb, and xi parameterization.
    """
    name: str = "Cumulative Distribution Statistics"
    ndim_supp: int = 0 #Output dimension. Should be just 0.
    ndims_params: List[int] = [1, 1, 1] #Parameter dimension. Each parameter should be a vector.
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("Cumulative Distribution Statistics", "\\operatorname{CDS}")

    def __call__(self, E = np.array([0.0]), zb = np.array([1.0]), xi = np.array([-0.1]), size=None, **kwargs) -> TensorVariable:
        return super().__call__(E, zb, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: Union[np.random.RandomState, np.random.Generator],
        E: np.ndarray,
        zb: np.ndarray,
        xi: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        return cds.rvs(E = E, zb = zb, xi = xi, size=size, random_state=rng)
    
cdstats = CDStatsRV()        

class CDStats(Continuous):
    """
    Cumulative Distribution Statistics Distribution. After Caprani, C.C. (2005).
    With E, zb, and xi parameterization.
    """
    rv_op = cdstats

    @classmethod
    def dist(cls, E = np.array([0.0]), zb = np.array([1.0]), xi = np.array([-0.1]), **kwargs):
        E = at.as_tensor_variable(floatX(E))
        zb = at.as_tensor_variable(floatX(zb))
        xi = at.as_tensor_variable(floatX(xi))

        return super().dist([E, zb, xi], **kwargs)
    

    def logp(value, E, zb, xi):
        return logpdf(value, E, zb, xi)

    def logcdf(value, E, zb, xi):
        return logcdf(value, E, zb, xi)