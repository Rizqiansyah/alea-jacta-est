from aje.stats.distributions.utils.dist_infrastructure import vector_rv_generator
from aje.stats.distributions.numpy_methods.cds import argcheck, support, cdf, logcdf, pdf, logpdf, ppf

cds = vector_rv_generator(
    name="CDS",
    params=["E", "zb", "xi"],
    check_args=argcheck,
    _support = support,
    _cdf = cdf,
    _logcdf = logcdf,
    _pdf = pdf,
    _logpdf = logpdf,
    _ppf = ppf
)