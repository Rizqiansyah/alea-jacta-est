# alea-jacta-est
Extra statistic distributions and functions in the style of scipy. Also supports PyMC and pytensor.

## Current features
### Statistical distributions
- Generic truncated `gentruncated` and general block maxima `genmaxima` distribution class, with methods similar to that of `scipy.stats`. This distribution class can take any `scipy.stats` distribution as its base distribution, and use numerical intergration to calculate the PDF, CDF and other statistics.
- Generic multimodal/mixture distribution `mixturedistribution`, which can take any number of `scipy.stats` distributions as its base distributions. e.g., you can create a multimodal beta distribution, multimodal normal, or even mixture of distributions, e.g., Lognormal-Normal-Weibull distribution. A sample multimodal beta distribution is distributed via the `multimodalbeta` class.
- Cumulative distribution statistics (CDS) after Caprani, 2004. Available both in `scipy` like class, and for use in `pymc`.

### Pytensor special functions
- Implement some `scipy.special` functions for `pytensor`. e.g., `betaincinv`, `gammaincinv`, `polygamma`, etc.
- Also implement some functions specific to the GEV distribution.

### PyMC distribution
- CDS distribution
- GEV Weibull distribution, i.e., GEV distribution restricted to Weibull domain of attraction.
- `from_posterior` class, which uses interpolation to approximate the probability density of a posterior sample. Can be useful for Bayesian updatin, although the interaction between parameters are lost.


## Wishlist
- More `scipy.special` functions implementation for pytensor.
- Implementation of generic multimodal, genmaxima and gentruncated distributions for PyMC.
- More exotic distributions for both pymc and scipy like classes.

## Disclaimer
Initially, I wrote this libary as a private library that I found useful for my PhD. However, some of the codes were heavily inspired from multiple sources, and somewhere along the way, I have lost these sources. If you see any codes that you believe belongs to you, or heavily inspired by your work, please let me know (email: akbar.rizqiansyah@monash.edu) and I will credit you or remove the code. I apologise for this oversight.