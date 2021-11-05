## GCM Filters

[![codecov](https://codecov.io/gh/ocean-eddy-cpt/gcm-filters/branch/master/graph/badge.svg?token=ZKRiulYe68)](https://codecov.io/gh/ocean-eddy-cpt/gcm-filters)
[![pre-commit](https://github.com/ocean-eddy-cpt/gcm-filters/workflows/pre-commit/badge.svg)](https://github.com/ocean-eddy-cpt/gcm-filters/actions?query=workflow%3Apre-commit)
[![Tests](https://github.com/ocean-eddy-cpt/gcm-filters/workflows/Tests/badge.svg)](https://github.com/ocean-eddy-cpt/gcm-filters/actions?query=workflow%3ATests)
[![Documentation Status](https://readthedocs.org/projects/gcm-filters/badge/?version=latest)](https://gcm-filters.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gcm_filters.svg)](https://anaconda.org/conda-forge/gcm_filters)
[![PyPI version](https://badge.fury.io/py/gcm-filters.svg)](https://badge.fury.io/py/gcm-filters)

GCM-Filters: Diffusion-based Spatial Filtering of Gridded Data

### Description

**GCM-Filters** is a python package that performs spatial filtering analysis in a flexible and efficient way.
The GCM-Filters algorithm applies a discrete Laplacian to smooth a field through an iterative process that resembles diffusion ([Grooms et al., 2021](https://doi.org/10.1029/2021MS002552)).
The package can be used for either gridded observational data or gridded data that is produced by General Circulation Models (GCMs) of ocean, weather, and climate.
Such GCM data come on complex curvilinear grids, whose geometry is respected by the GCM-Filters Laplacians.
Through integration with [dask](https://dask.org/), GCM-Filters enables parallel, out-of-core filter analysis on both CPUs and GPUs.

### Installation

GCM-Filters can be installed with pip:

```shell
pip install gcm_filters
```

or conda
```shell
conda install -c conda-forge gcm_filters
```

### Getting Started

To learn how to use GCM-Filters for your data, visit the [GCM-Filters documentation](https://gcm-filters.readthedocs.io/).


## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/ocean-eddy-cpt/gcm-filters).


## License and copyright

GCM-Filters is licensed under version 3 of the Gnu Lesser General Public License.

Development occurs on GitHub at <https://github.com/ocean-eddy-cpt/gcm-filters>.
