## GCM Filters

[![pre-commit](https://github.com/ocean-eddy-cpt/gcm-filters/workflows/pre-commit/badge.svg)](https://github.com/ocean-eddy-cpt/gcm-filters/actions?query=workflow%3Apre-commit)
[![Tests](https://github.com/ocean-eddy-cpt/gcm-filters/workflows/Tests/badge.svg)](https://github.com/ocean-eddy-cpt/gcm-filters/actions?query=workflow%3ATests)
[![Documentation Status](https://readthedocs.org/projects/gcm-filters/badge/?version=latest)](https://gcm-filters.readthedocs.io/en/latest/?badge=latest)

GCM-Filters: Diffusion-based Spatial Filtering of Gridded Data from General Circulation Models

### Description

**GCM-Filters** is a python package that performs spatial filtering analysis in a flexible and efficient way.
The GCM-Filters algorithm applies a discrete Laplacian to smooth a field through an iterative process that resembles diffusion (`Grooms et al., 2021 <https://doi.org/10.1002/essoar.10506591.1>`_).
The package is specifically designed to work with gridded data that is produced by General Circulation Models (GCMs) of ocean, weather, and climate.
Such GCM data come on complex curvilinear grids, whose geometry is respected by the GCM-Filters Laplacians.
Through integration with `dask <https://dask.org/>`_, GCM-Filters enables parallel, out-of-core filter analysis on both CPUs and GPUs.

### Installation

GCM-Filters can be installed with `pip`:

```shell
pip install gcm_filters
```

### Getting Started

To learn how to use GCM-Filters for your data, visit the `GCM-Filters documentation <https://dask.org/>`_.


## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/ocean-eddy-cpt/gcm-filters).


## License and copyright

ioos_pkg_skeleton is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/ocean-eddy-cpt/gcm-filters>.
