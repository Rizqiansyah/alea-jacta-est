import numpy as np
import pytensor.tensor as at

def elemwise_constructor(op_class):
    """
    Constructor function, so that a regular Op behave closer to Elemwise Ops.
    Note it does not completely replicate Elemwise Ops.
    """
    def out_fn(*args):
        x = args[-1]
        #Put into numpy array for all args
        input_args = [np.array(arg) for arg in args[:-1]]
        
        #Check if all args are scalars
        ndim = [arg.ndim for arg in input_args]
        if all([n == 0 for n in ndim]):
            input_args = [arg[()] for arg in input_args]
            if at.as_tensor(x).ndim == 0:
                #Assume all scalar
                return op_class(*input_args)(at.flatten(at.as_tensor([x])))[0]
            else:
                return op_class(*input_args)(at.flatten(at.as_tensor(x)))
        
        else:
            #Check all ndim are the same
            if not all([n == ndim[0] for n in ndim]):
                raise ValueError('All input arguments must have the same number of dimensions')
            out = at.zeros_like(x)
            for i in range(len(input_args[0])):
                out = at.set_subtensor(out[i], op_class(*[arg[i] for arg in input_args])(at.flatten(at.as_tensor([x[i]])))[0])
            return out
    return out_fn