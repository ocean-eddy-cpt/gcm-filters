.. gcm-filters documentation master file, created by
   sphinx-quickstart on Tue Jan 12 09:24:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GCM Filters: Diffusion-based Spatial Filtering of Gridded Data
===============================================================

**GCM-Filters** is a python package that performs spatial filtering analysis in a flexible and efficient way.
The GCM-Filters algorithm applies a discrete Laplacian to smooth a field through an iterative process that resembles diffusion (see :doc:`theory` or `Grooms et al., 2021 <https://doi.org/10.1029/2021MS002552>`_).
The package can be used for either gridded observational data or gridded data that is produced by General Circulation Models (GCMs) of ocean, weather, and climate.
Such GCM data come on complex curvilinear grids, whose geometry is respected by the GCM-Filters Laplacians.
Through integration with `dask <https://dask.org/>`_, GCM-Filters enables parallel, out-of-core filter analysis on both CPUs and GPUs.

Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   theory
   basic_filtering
   gpu
   examples/example_filter_types
   examples/example_tripole_grid
   examples/example_vector_laplacian
   examples/example_numerical_instability
   examples/example_satellite_observations
   api
   how_to_contribute



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
