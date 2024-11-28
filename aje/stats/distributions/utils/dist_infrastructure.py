from numpy import sqrt, prod, exp, log, ones_like, array
from numpy import inf

#Not pytensor compatible functions
from scipy.optimize import newton
from scipy.stats import uniform as scipy_uniform_dist
from aje.stats.utils import numerical_moment, numerical_standardised_moment

__all__ = ["vector_rv_generator"]

def vector_rv_generator(
    name,  
    check_args=None, 
    params=None,
    _support=None,
    _cdf=None,
    _pdf=None,
    _logcdf=None,
    _logpdf=None,
    _sf=None,
    _ppf=None,
    _isf=None,
    _rvs=None,
    _moment=None,
    _std_moment=None,
    _entropy=None,
    _fit=None,
    _expect=None,
    _median=None,
    _stats=None,
    _mean=None,
    _var=None,
    _std=None,
):
    if check_args is None:
        def _f(**kwargs):
            return kwargs
        check_args = _f

    if params is not None:
        def _check_args(**kwargs):
            if not all([key in kwargs for key in params]):
                raise ValueError(f"Missing one or more keywords argument: all of {params} must be specified")
            return check_args(**kwargs)

    if _cdf is None and _logcdf is not None:
        def _cdf(*args, **kwargs):
            return exp(_logcdf(*args, **kwargs))

    if _pdf is None and _logpdf is not None:
        def _pdf(*args, **kwargs):
            return exp(_logpdf(*args, **kwargs))
        
    if _logcdf is None and _cdf is not None:
        def _logcdf(*args, **kwargs):
            return log(_cdf(*args, **kwargs))
        
    if _logpdf is None and _pdf is not None:
        def _logpdf(*args, **kwargs):
            return log(_pdf(*args, **kwargs))
        
    if _sf is None and _cdf is not None:
        def _sf(*args, **kwargs):
            return 1 - _cdf(*args, **kwargs)
        
    if _ppf is None:
        if _isf is not None:
            def _ppf(q, *args, **kwargs):
                return _isf(1-q, *args, **kwargs)
        
        if _cdf is not None and _pdf is not None:
            def _ppf(q, *args, **kwargs):
                #Delete the params from kwargs
                solver_kwargs = {key: kwargs[key] for key in kwargs if key not in params}
                return newton(lambda x: _cdf(x, *args, **kwargs) - q, ones_like(q), fprime=lambda x: _pdf(x, *args, **kwargs), *args, **solver_kwargs)
            
    if _isf is None:
        if _ppf is not None:
            def _isf(self, q, *args, **kwargs):
                return _ppf(self, 1-q, *args, **kwargs)
            
    if _rvs is None:
        if _ppf is not None:
            def _rvs(size=None, random_state=None, *args, **kwargs):
                if size == () or size == []:
                    size = None
                #Delete the params from kwargs
                rng_kwargs = {key: kwargs[key] for key in kwargs if key not in params}
                u = scipy_uniform_dist.rvs(size=prod(size), random_state=random_state, *args, **rng_kwargs)
                if size is None:
                    return _ppf(q=u, *args, **kwargs)[0]
                return _ppf(q=u, *args, **kwargs).reshape(size)
            
    if _moment is None and _pdf is not None:
        exc_list = ["integral_lb", "integral_ub", "integral_method"] + params
        def _moment(order, *args, **kwargs):
            try:
                supp = _support(**kwargs)
            except NotImplementedError:
                supp = [-inf, inf]
            
            if "integral_lb" not in kwargs or kwargs["integral_lb"] is None:
                integral_lb = supp[0]

            if "integral_ub" not in kwargs or kwargs["integral_ub"] is None:
                integral_ub = supp[1]

            if "integral_method" not in kwargs or kwargs["integral_method"] is None:
                integral_method = "quad"

            #Delete the params from kwargs
            integral_kwargs = {key: kwargs[key] for key in kwargs if key not in exc_list}

            return numerical_moment(fx = lambda x: _pdf(x, *args, **kwargs),
                                    order = order, 
                                    integral_lb = integral_lb, 
                                    integral_ub = integral_ub,
                                    integral_method = integral_method, 
                                    **integral_kwargs)
        
    if _std_moment is None and _pdf is not None:
        exc_list = ["integral_lb", "integral_ub", "integral_method"] + params
        def _std_moment(order, *args, **kwargs):
            try:
                supp = _support(**kwargs)
            except NotImplementedError:
                supp = [-inf, inf]
            
            if "integral_lb" not in kwargs or kwargs["integral_lb"] is None:
                integral_lb = supp[0]

            if "integral_ub" not in kwargs or kwargs["integral_ub"] is None:
                integral_ub = supp[1]

            if "integral_method" not in kwargs or kwargs["integral_method"] is None:
                integral_method = "quad"

            #Delete the params from kwargs
            integral_kwargs = {key: kwargs[key] for key in kwargs if key not in exc_list}

            return numerical_standardised_moment(fx = lambda x: _pdf(x, *args, **kwargs),
                                                order = order, 
                                                integral_lb = integral_lb, 
                                                integral_ub = integral_ub, 
                                                integral_method = integral_method, 
                                                **integral_kwargs)

    if _stats is None and _std_moment is not None and _moment is not None:
        def _stats(*args, **kwargs):
            moments = kwargs["moments"]
            moments = list(moments)
            kwargs = {key: kwargs[key] for key in kwargs if key not in ["moments"]}
            for a in moments:
                if a == 'm':
                    return _moment(order = 1, *args, **kwargs)
                elif a == 'v':
                    return _std_moment(order = 2, *args, **kwargs)
                elif a == 's':
                    return _std_moment(order = 3, *args, **kwargs)
                elif a == 'k':
                    return _std_moment(order = 4, *args, **kwargs) - 3.0
                else:
                    raise ValueError("Unrecognised 'moments' argument. Should be 'm', 'v', 's', 'k', or combinations of those. See scipy doc for detail")

    if _mean is None and _moment is not None:
        def _mean(*args, **kwargs):
            return _moment(order = 1, *args, **kwargs)
        
    if _var is None and _std_moment is not None:
        def _var(*args, **kwargs):
            return _std_moment(order = 2, *args, **kwargs)
        
    if _std is None and _var is not None:
        def _std(*args, **kwargs):
            return sqrt(_var(*args, **kwargs))

    if _support is None:
        def _support(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")

    if _cdf is None:
        def _cdf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _pdf is None:
        def _pdf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _logcdf is None:
        def _logcdf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _logpdf is None:
        def _logpdf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _sf is None:
        def _sf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _ppf is None:
        def _ppf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _isf is None:
        def _isf(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
    
    if _rvs is None:
        def _rvs(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _moment is None:
        def _moment(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _std_moment is None:
        def _std_moment(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
    
    if _entropy is None:
        def _entropy(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _fit is None:
        def _fit(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _expect is None:
        def _expect(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _median is None:
        def _median(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _stats is None:
        def _stats(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _mean is None:
        def _mean(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    if _var is None:
        def _var(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")

    if _std is None:
        def _std(*args, **kwargs):
            raise NotImplementedError("Not implemented yet")
        
    class vector_rv_gen():
        support = (-inf, inf)
        def __init__(self, seed=None, *args, **params):
            pass

        def __call__(self, seed=None, *args, **kwargs):
            return vector_rv_frozen(seed=seed, *args, **kwargs)
        
        def get_support(cls, **params):
            _check_args(**params)
            return _support(**params)
        
        def cdf(cls, x, **params):
            _check_args(**params)
            return _cdf(x, **params)
        
        def pdf(cls, x, **params):
            _check_args(**params)
            return _pdf(x, **params)
        
        def logcdf(cls, x, **params):
            _check_args(**params)
            return _logcdf(x, **params)
        
        def logpdf(cls, x, **params):
            _check_args(**params)
            return _logpdf(x, **params)
        
        def sf(cls, x, **params):
            _check_args(**params)
            return _sf(x, **params)
        
        def ppf(cls, q, **params):
            _check_args(**params)
            return _ppf(q, **params)
        
        def isf(cls, q, **params):
            _check_args(**params)
            return _isf(q, **params)
        
        def rvs(cls, size=None, random_state=None, *args, **kwargs):
            _check_args(**kwargs)
            return _rvs(size=size, random_state=random_state, *args, **kwargs)
        
        def moment(cls, order, *args, **kwargs):
            _check_args(**kwargs)
            return _moment(order,  *args, **kwargs)
        
        def std_moment(cls, order, *args, **kwargs):
            _check_args(**kwargs)
            return _std_moment(order,  *args, **kwargs)
        
        def entropy(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _entropy(*args, **kwargs)
        
        def fit(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _fit(*args, **kwargs)
        
        def expect(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _expect(*args, **kwargs)
        
        def median(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _median(*args, **kwargs)
        
        def stats(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _stats(*args, **kwargs)
        
        def mean(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _mean(*args, **kwargs)
        
        def var(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _var(*args, **kwargs)
        
        def std(cls, *args, **kwargs):
            _check_args(**kwargs)
            return _std(*args, **kwargs)

    class vector_rv_frozen():
        def __init__(self, seed=None, *args, **kwargs):
            self._params = _check_args(**kwargs)
            self._dist = vector_rv_gen(seed)
            self.name = name

        @property
        def params(self):
            return self._params
        
        @property
        def support(self):
            return _support(**self.params)
        
        def get_support(self):
            return self.support

        def cdf(self, x):
            return _cdf(x, **self.params)

        def pdf(self, x):
            return _pdf(x, **self.params)

        def logcdf(self, x):
            return _logcdf(x, **self.params)

        def logpdf(self, x):
            return _logpdf(x, **self.params)
        
        def sf(self, x):
            return _sf(x, **self.params)
        
        def ppf(self, q):
            return _ppf(q, **self.params)
        
        def isf(self, q):
            return _isf(q, **self.params)
        
        def rvs(self, size=None, random_state=None, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _rvs(size=size, random_state=random_state, *args, **kwargs)
        
        def moment(self, order, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _moment(order,  *args, **kwargs)
        
        def std_moment(self, order, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _std_moment(order,  *args, **kwargs)

        def entropy(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _entropy(*args, **kwargs)
        
        def fit(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _fit(*args, **kwargs)
        
        def expect(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _expect(*args, **kwargs)
        
        def median(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _median(*args, **kwargs)
        
        def stats(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _stats(*args, **kwargs)
        
        def mean(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _mean(*args, **kwargs)
        
        def var(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _var(*args, **kwargs)
        
        def std(self, *args, **kwargs):
            kwargs = {**self.params, **kwargs}
            return _std(*args, **kwargs)

    return vector_rv_gen()

