import numpy as np
import pytensor.tensor as at
from pytensor.graph.op import Op
from aje.special import xi_from_zbc
from aje.special.xi_and_zbc import init_guess
from aje.special import xi_from_zbc as numpy_xi_from_zbc

__all__ = ["xi_from_zbc", "zbc_from_xi"]

class XiFromZbc(Op):
    """
    AESARA xi SOLVER
    See for reference: https://docs.pymc.io/en/v3/Advanced_usage_of_Theano_in_PyMC3.html
    Reference still applies for pymc4, just use aesara instead of theano
    """
    itypes = [at.dvector]
    otypes = [at.dvector]
    
    def perform(self, node, inputs, outputs):
        z_bc, = inputs
        xi = numpy_xi_from_zbc(z_bc, init_guess(z_bc))
        outputs[0][0] = np.array(xi)
        
    def grad(self, inputs, g):
        z_bc, = inputs
        xi = self(z_bc)
        g1 = at.gamma(1-xi)
        g2 = at.gamma(1-2*xi)
        dg1 = at.digamma(1-xi)
        dg2 = at.digamma(1-2*xi)
        
        nom = at.pow(g2-at.pow(g1, 2), 1.5)
        denom = g1*g2*(dg2 - dg1)
        
        return [g[0] * nom/denom]
    
xi_from_zbc = XiFromZbc()

def zbc_from_xi(xi):
    """
    Inverse of xi_from_zbc_function. This one is more straightforward.
    Here for convenience.
    """
    return at.gamma(1-xi) / at.sqrt(at.gamma(1-2*xi) - at.pow(at.gamma(1-xi), 2))
