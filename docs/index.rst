.. gcm-filters documentation master file, created by
   sphinx-quickstart on Tue Jan 12 09:24:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GCM Filters: Diffusion-based Spatial Filtering of Gridded Data from General Circulation Models
===========

GCM-Filters is a python package that allows users to perform spatial filtering analysis in an easy, flexible and efficient way. The package implements the filtering method that was introduced by `Grooms et al. (2021) <https://doi.org/10.1002/essoar.10506591.1>`_. The filtering algorithm is analogous to smoothing via diffusion; hence the name diffusion-based filters. GCM-Filters is designed to work with gridded data that is produced by General Circulation Models (GCMs) of ocean, weather, and climate.


.. toctree::
   :maxdepth: 2

   theory
   tutorial
   tutorial_GPU
   tutorial_filter_types
   tutorial_tripole_grid
   tutorial_vector_laplacian
   tutorial_numerical_instability
   API Reference <api>
   how_to_contribute



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
