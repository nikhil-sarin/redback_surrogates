# `Redback_surrogates`
This is an optional package built to be used via [Redback](https://redback.readthedocs.io/en/latest/) or independently. 
This package contains the data and methods to build/call surrogate models for EM transients using machine learning built on grids of numerical simulations.
If your aim is to just evaluate the models you could use this package independently but if you wish to simulate realistic transients or perform inference with these models, use [Redback](https://redback.readthedocs.io/en/latest/)

[![Documentation Status](https://readthedocs.org/projects/redback-surrogates/badge/?version=latest)](https://redback-surrogates.readthedocs.io/en/latest/?badge=latest)
![Python package](https://github.com/nikhil-sarin/redback_surrogates/workflows/Python%20application/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/nikhil-sarin/redback_surrogates/badge.svg?branch=main)](https://coveralls.io/github/nikhil-sarin/redback_surrogates?branch=main)
![PyPI](https://img.shields.io/pypi/v/redback_surrogates)

# Redback
Introducing REDBACK, a bayesian inference software package for fitting electromagnetic transients

### Online documentation for `Redback_surrogates`

- [Installation](https://redback-surrogates.readthedocs.io/en/latest/installation.html)
- [Examples](https://github.com/nikhil-sarin/redback_surrogates/tree/master/examples)
- [Documentation](https://redback-surrogates.readthedocs.io/en/latest/)

### Online documentation for `Redback`
- [Installation](https://redback.readthedocs.io/en/latest/)
- [Examples](https://github.com/nikhil-sarin/redback/tree/master/examples)
- [Documentation](https://redback.readthedocs.io/en/latest/)

### Motivation and why redback might be useful to you.
The launch of new telescopes/surveys is leading to an explosion of transient observations. 
Redback is a software package for end-to-end interpretation and parameter estimation of these transients.

- Download data for supernovae, tidal disruption events, gamma-ray burst afterglows, kilonovae, prompt emission from 
  different catalogs/telescopes; Swift, BATSE, Open access catalogs. Users can also provide their own data or use simulated data
- Redback processes the data into a homogeneous transient object, plotting lightcurves and doing other processing.
- The user can then fit one of the models implemented in redback. Or fit their own model. Models for several different types of electromagnetic transients are implemented and range from simple analytical models to numerical surrogates.
- All models are implemented as functions and can be used to simulate populations, without needing to provide data. This way redback can be used simply as a tool to simulate realistic populations, no need to actually fit anything.
- [Bilby](https://lscsoft.docs.ligo.org/bilby/index.html) under the hood. Can easily switch samplers/likelihoods etc. By default the choice is made depending on the data.
- Fitting returns a homogenous result object, with functionality to plot lightcurves and the posterior/evidence etc.

### Contributing
If you are interested in contributing please join the redback 
[slack](https://join.slack.com/t/redback-group/shared_invite/zt-2n735gnnl-EUuZKv5Db2kHVuSKbwUo2Q)
and email [Nikhil Sarin](mailto:nsarin.astro@gmail.com?subject=Contributing%20to%20redback).

To make changes to redback, we require users to use pull requests. 
