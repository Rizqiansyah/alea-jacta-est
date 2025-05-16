from scipy.special import gamma, polygamma, lambertw
import numpy as np
from typing import List
from scipy.optimize import newton

__all__ = ['gammainv']

alpha = 1.461632144968362341262659542325721328468196204006446351295988409
"""
The input to the $\Gamma(\cdot)$ function such that $\Gamma(\alpha)# is the minimum for $\alpha > 0$.
Ref: https://oeis.org/A030171
"""
beta = gamma(alpha)
"""
The minimum value of the $\Gamma(\cdot)$ function in the positive $x$ domain.
"""

gamma_alpha = gamma(alpha)
trigamma_alpha = polygamma(1, alpha)
e_neg1 = np.exp(-1)
root_2pi = np.sqrt(2 * np.pi)
def gammainv_approx(y: float | List[float]) -> tuple[float, float] | tuple[List[float], List[float]]:
    """
    Approximate of the inverse of the gamma function in the positive domain, 
    i.e., find the solution $x$ given $y$ to the equation:
    \[
        \Gamma(x) = y, 
    \]
    and $x > 0$. $y$ must be larger than $\alpha \approx 1.461...$, the minimum value of the gamma function in the positive $x$ domain.
    Two solutions are given, the first is for $x<\alpha$ and the second is for $x>\alpha$.
    For $x<\alpha$,
    \[
        x = \begin{cases}
            \frac{1}{y}, & y \geq 1                                                                [2]\\
            \alpha - \sqrt{\frac{2(y-\Gamma(\alpha))}{\psi(\alpha)\Gamma(\alpha)}},  & y < 1       [1]
        \end{cases}
    \]
    For $x>\alpha$, 
    \[
        x = \begin{cases}
            \frac{1}{2} + \frac{\log(y/\sqrt{2\pi})}{W(-e^{-1}\log(y/\sqrt{2\pi}))}, & y \geq 1    [1]*\\
            \alpha + \sqrt{\frac{2(y-\Gamma(\alpha))}{\psi(\alpha)\Gamma(\alpha)}}, & y < 1        [2]
        \end{cases}
    \]
    * Note this case is also given in [2], but with a typo. Check [1], Eqn. 67 for the correct equation.

    Parameters
    ----------
    y : float
        The value to find the inverse of the gamma function for. Must be larger than $\beta$,
        the minimum value of the gamma function in the positive $x$ domain, i.e.
        \beta = $\Gamma(\alpha) \approx 0.8856...$.

    Returns
    -------
    first_soln: float
        The solution for $x<\alpha$.
    second_soln: float
        The solution for $x>\alpha$.

    References
    ----------
    [1] J.M. Borwein and R. M. Corless, The Gamma function in the Monthly American Math Monthly.
    [2] Pedersen, Henrik (9 September 2013). ""Inverses of gamma functions"". Constructive Approximation. 7 (2): 251-267. arXiv:1309.2167. doi:10.1007/s00365-014-9239-1. S2CID 253898042.
    
    NOTE:
    This function has quite a large error for some $y$.
    It is advisable that a more accurate numerical method be used, with the initial guess
    being the output of this function.
    """
    principal_root = np.sqrt((2*(y-gamma_alpha)) / (trigamma_alpha*gamma_alpha))
    first_soln = np.where(
        y < 1.0,
        alpha - principal_root,
        1/y
    )
    second_soln = np.where(
        y < 1.0,
        alpha + principal_root,
        1/2 + np.log(y/root_2pi) / (lambertw(e_neg1 * np.log(y/root_2pi)).real)
    )
    
    return first_soln, second_soln

def gammainv(y: float|List[float], **kwargs) -> tuple[float, float] | tuple[List[float], List[float]]:
    """
    The inverse of the gamma function in the positive domain, 
    i.e., find the solution $x$ given $y$ to the equation:
    \[
        \Gamma(x) = y, 
    \]
    and $x > 0$. $y$ must be larger than $\alpha \approx 1.461...$, the minimum value of the gamma function in the positive $x$ domain.
    Two solutions are given, the first is for $x<\alpha$ and the second is for $x>\alpha$.

    Numerical scheme is used (Newton's method), with the initial guess using the `gammainv_approx` function.

    Parameters
    ----------
    y : float
        The value to find the inverse of the gamma function for. Must be larger than $\beta$,
        the minimum value of the gamma function in the positive $x$ domain, i.e.
        \beta = $\Gamma(\alpha) \approx 0.8856...$.
    kwargs : dict
        Additional arguments to pass to the `newton` function.
        See `scipy.optimize.newton` for more details.

    Returns
    -------
    first_soln: float
        The solution for $x<\alpha$.
    second_soln: float
        The solution for $x>\alpha$.
    """
    kwargs.setdefault("x0", gammainv_approx(y))
    soln = newton(lambda x: gamma(x) - y, **kwargs)
    return soln[0, ...], soln[1, ...]