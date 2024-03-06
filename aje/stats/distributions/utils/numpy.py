from numpy import ones, transpose, equal as eq, where as switch

__all__ = ["vector_2d"]

def vector_2d(
    x,
):
    """
    This is a convenience function to create a 2d vector from a 1d vector or float.
    Useful when trying to do operation of two vectors u and v of different lengths, 
    and we want to perform operation for each element of u on each element of v (different to elementwise!).
    

    Parameters
    ----------
    x : array or float
        The input vector or float.
        

    Returns
    -------
    array
        The 2d vector of shape (n, 1), where n is the length of the input vector.

    Notes
    -----
    The function is only intended to be used for a float, 1d vector, or 2d vector input.
    
    If x has ndim > 1, then it will return itself.
    If x is an n x m matrix, where m > 1, then it will be transposed.


    Example
    -------
    Multiplying for each elements of two vectors u and v of different lengths.
    ```
    output = np.zeros((len(u), len(v)))
    for i, elem_u in enumerate(u):
        for j, elem_v in enumerate(v):
            output[i, j] = elem_u * elem_v
    ```
    is equivalent to
    ```
    output = vector_2d(u) * v
    ```

    """
    if x.ndim <= 1:
        return transpose(ones((1, 1)) * x)
    elif x.ndim == 2:
        return switch(
            eq(x.shape[0], 1),
            transpose(x),
            x,
        )
    else:
        return x