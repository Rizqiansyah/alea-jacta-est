import numpy as np

__all__ = ["numpy_like_func_on_axis"]

def numpy_like_func_on_axis(func, arr, axis=None, **kwargs):
    """
    Function to apply a function to an array along a given axis.
    Like numpy function, e.g. np.mean(arr, axis = (...)), but can be used on any function.
    Parameters
    ----------
    func : function
        Function to apply to the array.
    arr : array_like
        Array to apply the function to.
    axis : int or tuple of ints
        Axis or axes along which to apply the function. If None, apply the function across all values.
    **kwargs
        Keyword arguments to pass to the function. This is passed to the function in `func` argument.
    Returns
    -------
    array_like
        Array with the function applied.
    """
    #If axis is None, assume calculate mode across all values.
    if axis is None:
        arr = arr.flatten()
        axis = 0
    #Ensure we can get the length of axis
    axis = np.atleast_1d(axis)
    #Reshape the array so the first n dimension are the one being operated over
    #and the last dimension is the one being operated on
    arr = np.moveaxis(arr, axis, range(len(axis)))
    #Flatten the first n axis
    arr = arr.reshape(-1, *arr.shape[len(axis):])
    #Apply the **kwargs to the function
    def ufunc(arr):
        return func(arr, **kwargs)
    #Apply the function and return
    return np.apply_along_axis(ufunc, 0, arr)