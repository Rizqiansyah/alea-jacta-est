import numpy as np
import pytensor.tensor as at
from pytensor.graph.op import Op
from aje.pytensor.utils import elemwise_constructor
from aje.special import xi_from_kp as numpy_xi_from_kp

class XiFromKp(Op):
    """
    Pytensor class of the xi_from_kp function.
    """

    __props__ = ('p',)
    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, p):
        self.p = p
        self.cp = -np.log(1-p)
        super().__init__()

    def perform(self, node, inputs, outputs):
        kp = inputs[0]
        xi = numpy_xi_from_kp(kp, p=self.p)
        outputs[0][0] = np.array(xi)

    def grad(self, inputs, g):
        # 1/f'(f^-1 (kp))
        kp, = inputs
        xi = self(kp)
        cp = self.cp
        return [
            g[0] * at.gamma(1-xi) / ( cp**(-xi) * (at.log(cp)-at.digamma(1-xi)) )
        ]
    
xi_from_kp = elemwise_constructor(XiFromKp)
"""
Pytensor function of the xi_from_kp function.
Please note the sequence of the parameter. 
This function does not accept keyword arguments, it must be sequential.
Notably, the sequence is (p, kp); as opposed to (kp, p) in the original function.
For future notes, may be flip this around as it can be confusing.

Parameters
----------
p: float
    Probability of non-exceedance.
kp: float
    The value of (zp - E) / (zb - E), where E, zp, zb are the mean, the p-th return level, and the upper bound respectively.

"""

def kp_from_xi(xi, p):
    return 1 - (-at.log(1-p))**(-xi)/at.gamma(1-xi)