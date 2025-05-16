import pacal as pc
from scipy.stats._distn_infrastructure import rv_continuous
from typing import Optional

__all__ = ["pacal_to_scipy"]

def pacal_to_scipy(pacal_dist: pc.distr, a:Optional[float] =None, b:Optional[float]=None):
    """
    Convert a pacal distribution to a scipy distribution.

    Parameters
    ----------
    pacal_dist : pacal distribution
        The pacal distribution to convert.
    a : float, optional
        The lower bound of the distribution. If None, the lower bound of the pacal distribution is used.
    b : float, optional
        The upper bound of the distribution. If None, the upper bound of the pacal distribution is used.

    Returns
    -------
    scipy distribution
        The scipy distribution corresponding to the pacal distribution.

    Notes
    -----
    The lower and upper bounds of the pacal distribution can sometimes be incorrect.
    It is useful to set them manually in some cases.
    """

    a = a if a is not None else pacal_dist.range()[0]
    b = b if b is not None else pacal_dist.range()[1]
    class Pacal2Scipy_gen(rv_continuous):
        def _get_support(self):
            return a, b

        def _pdf(self, x):
            return pacal_dist.pdf(x)

        def _cdf(self, x):
            return pacal_dist.cdf(x)

        def _ppf(self, q):
            return pacal_dist.quantile(q)
        
        def _rvs(self, size=None, random_state=None):
            # print(random_state)
            # if random_state is not None:
            #     raise NotImplementedError("Random state not implemented for pacal distributions.")
            return pacal_dist.rand(size)
        
    pacal2scipy = Pacal2Scipy_gen(name=pacal_dist.getName())
    return pacal2scipy()