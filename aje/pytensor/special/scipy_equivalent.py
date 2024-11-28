# PyTensor implementation of scipy special functions
# Although the objective is similar to pytensor-special package
# This module instead assumes that the parameter of the functions are floats, not tensors.
# This module also only accepts scalar or vector, and the arguments must be of the same type (scalars or vectors, but not both)
# An advantage of this module is it is a lot faster than the pytensor-special package, as the pytensor-special package does not have c code implementation yet.
# It is expected that in the future, once c implementation is done in the pytensor-package, this module will be deprecated.

import numpy as np
from pytensor.graph.op import Op
import pytensor.tensor as at
from aje.pytensor.utils import elemwise_constructor
from scipy.special import beta as scipy_beta
from scipy.special import betaincinv as scipy_betaincinv
# from scipy.special import gamma as scipy_gamma
# from scipy.special import gammainc as scipy_gammainc
from scipy.special import gammaincinv as scipy_gammaincinv
from  scipy.special import polygamma as scipy_polygamma

__all__ = ["betaincinv", "gammaincinv", "polygamma", "ndtr", "ndtri"]

# Inverse of incomplete regularized beta function
# Refer to https://pytensor.readthedocs.io/en/latest/extending/creating_an_op.html
# For gradient: https://mathworld.wolfram.com/IncompleteBetaFunction.html 
class BetaIncGrad(Op):
    """
    Pytensor implementation of the gradient of the regularized incomplete beta function with respect to x
    """
    __props__ = ("a", "b")

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad_denom = scipy_beta(a, b)
        super().__init__()

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = output_storage[0]
        y[0] = x**(self.a-1)*(1-x)**(self.b-1)/scipy_beta(self.a, self.b)

    def grad(self, inputs, output_grads):
        #not really required for NUTS to run BetaIncInv
        x = inputs[0]
        nom = at.pow(x, self.a-2)*(-at.pow(1-x, self.b-2)) * (self.a*(x-1) + (self.b-2)*x + 1)
        return [output_grads[0] * nom/self.grad_denom]

class BetaIncInv(Op):
    """
        Pytensor class of the inverse regularized incomplete beta function.
        This is also the inverse CDF (or PPF, the probability point function) of the beta distribution.
    """
    __props__ = ("a", "b")

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.at_betaincgrad = BetaIncGrad(self.a, self.b)
        super().__init__()

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = output_storage[0]
        y[0] = scipy_betaincinv(self.a, self.b, x)

    def grad(self, inputs, output_grads):
        z, = inputs
        gz, = output_grads
        x = BetaIncInv(self.a, self.b)(z)
        return [
            output_grads[0]/self.at_betaincgrad(x)
        ]
    
betaincinv = elemwise_constructor(BetaIncInv)
"""
Pytensor implementation of the inverse regularized incomplete beta function.
This is also the inverse CDF (or PPF, the probability point function) of the beta distribution.

Parameters
----------
a: float or list of float
    alpha parameter of the function
b: float or list of float
    beta parameter of the function
x: pytensor.tensor.vector
    values to be evaluated
"""


class GammaIncInv(Op):
    """
    Pytensor implementation of the inverse of the regularized lower incomplete gamma function with respect to x
    """
    __props__ = ("a",)
    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, a):
        self.a = a
        super().__init__()

    def perform(self, node, inputs, output_storage):
        z = inputs[0]
        y = output_storage[0]
        y[0] = scipy_gammaincinv(self.a, z)

    def grad(self, inputs, output_grads):
        z, = inputs
        x = GammaIncInv(self.a)(z)
        gz, = output_grads
        return [
            gz * at.gamma(self.a) / (at.pow(x, self.a-1) * at.exp(-x))
        ]
    
gammaincinv = elemwise_constructor(GammaIncInv)
"""
Pytensor implementation of the inverse regularized lower incomplete gamma function.

Parameters
----------
a: float or list of float
    shape parameter of the function
x: pytensor.tensor.vector
    values to be evaluated
"""

class Polygamma(Op):
    """
    Pytensor class of the polygamma function.
    See scipy.special.polygamma for the definition of the polygamma function.
    """

    _props_ = ('n',)

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, n):
        self.n = n
        super().__init__()

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = output_storage[0]
        y[0] = scipy_polygamma(self.n, x)

    def grad(self, inputs, output_grads):
        z, = inputs
        gz, = output_grads
        return [
            gz * Polygamma(self.n + 1)(z)
        ]
    
polygamma = elemwise_constructor(Polygamma)
"""
Pytensor implementation of the polygamma.

Parameters
----------
p: float or list of float
    the order of the polygamma function. zero corresponds to the digamma function.
x: pytensor.tensor.vector
    values to be evaluated
"""

def ndtr(mu, sigma, x):
    """
        Pytensor implementation of the normal distribution CDF

        Parameters
        ----------
        mu: float
            Mean of the normal distribution
        sigma: float
            Standard deviation of the normal distribution

        Returns
        -------
        float
            The CDF of the normal distribution at x
    """
    return 0.5*(1 + at.erf((x-mu)/(sigma*np.sqrt(2))))

def ndtri(mu, sigma, x):
    """
        Pytensor implementation of the inverse normal distribution CDF

        Parameters
        ----------
        mu: float
            Mean of the normal distribution
        sigma: float
            Standard deviation of the normal distribution

        Returns
        -------
        float
            The inverse CDF of the normal distribution at x
    """
    return mu + sigma*at.sqrt(2)*at.erfinv(2*x-1)