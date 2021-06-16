---
title: 'GCM-Filters: A Python package for Diffusion-based Spatial Filtering of Gridded Data from General Circulation Models'
tags:
  - Python
  - ocean modeling
  - climate modeling
  - fluid dynamics
authors:
  - name: TBD
    affiliation: 1
affiliations:
  - name: TBD
    index: 1
date: 16 June 2021
bibliography: paper.bib

---

# Summary

`GCM-Filters` is a python package that allows scientists to perform spatial filtering analysis in an easy, flexible and efficient way. The package implements the filtering method that was introduced by @grooms2021diffusion. The filtering algorithm is analogous to smoothing via diffusion; hence the name "diffusion-based filters". `GCM-Filters` is designed to work with gridded data that is produced by General Circulation Models (GCMs) of ocean, weather, and climate. Spatial filtering of GCM data is a common analysis method in the Earth Sciences, for example to study oceanic and atmospheric motions at different spatial scales or to develop subgrid-scale parameterizations for ocean models.

`GCM-Filters` provides filters that are highly configurable, with the goal to be useful for a wide range of scientific applications. The user has different options for selecting the filter scale and filter shape.
The filter scale can be defined in several ways: a fixed length scale (e.g., 100 km), a scale tied to a model grid scale (e.g., 1$^\circ$), or a scale tied to a varying dynamical scale (e.g., the Rossby radius of deformation). As an example, \autoref{fig1} shows unfiltered and filtered relative vorticity, where the filter scale is set to a model grid scale of 4$^\circ$. `GCM-Filters` also allows for anisotropic, i.e., direction-dependent, filtering.
Finally, the filter shape -- currently: either Gaussian or Taper -- determines how sharply the filter separates scales above and below the target filter scale.

![(Left) Snapshot of unfiltered surface relative vorticity  $\zeta = \partial_x v - \partial_y u$ from a global 0.1$^\circ$ simulation with MOM6 [@adcroft2019MOM6]. (Right) Relative vorticity filtered to 4$^\circ$, obtained by applying `GCM-Filters` to the field $\zeta$ on the left. The plots are made with `matplotlib` [@Hunter2007] and `cartopy` [@Cartopy].\label{fig1}](filtered_vorticity.png){ width=100% }

# Statement of Need

Spatial filtering is commonly used as a scientific tool for analyzing gridded data. An example of an existing spatial filtering tool in python is `SciPy`'s [@2020SciPy-NMeth] `ndimage.gaussian_filter` function, implemented as a sequence of convolution filters. While being a valuable tool for image processing (or blurring), `SciPy`'s Gaussian filter is of limited use for GCM data; it assumes a regular and rectangular Cartesian grid, employs a simple boundary condition, and the definition of filter scale and shape have little flexibility. The python package `GCM-Filters` is specificially designed to filter GCM data, and seeks to solve a number of challenges for the user:

1. GCM data comes on irregular curvilinear grids with spatially varying grid-cell geometry.
2. Continental boundaries require careful / special treatment when filtering ocean GCM output.
3. Earth Science applications benefit from configurable filters, where the definition of filter scale and shape is flexible.
4. GCM output often comes in very large out-of-memory datasets.

The `GCM-Filters` algorithm [@grooms2021diffusion] applies a discrete Laplacian to smooth a field through an iterative process that resembles diffusion. The discrete Laplacian takes into account the varying grid-cell geometry and uses a no-flux boundary condition, mimicking how diffusion is internally implemented in GCMs. The no-flux boundary conditions ensures that the filter preserves the integral: $\int_{\Omega} \bar{f}(x,y) dx dy = \int_{\Omega} f (x,y) dx dy$, where $f$ is the original field, $\bar{f}$ the filtered field, and $\Omega$ the ocean domain. Conservation of the integral is a desirable property for many physical quantities, for example energy or ocean salinity. More details on the filter properties can be found in @grooms2021diffusion.

The main `GCM-Filters` class that the user will interface with is the `gcm_filters.Filter` object. When creating a filter object, the user specifies how they want to smooth their data, including the desired filter shape and filter scale. At this stage, the user also picks from a predefined list the grid type that is approporiate to their GCM data. Each grid type has an associated discrete Laplacian, and requires different “grid variables” that the user must provide (the latter are usually available to the user as part of the GCM output). Currently, `GCM-Filters` provides a number of different grid types and associated discrete Laplacians:

* Grid types with scalar Laplacians that can be used for filtering scalar fields, for example temperature or vorticity (see \autoref{fig1}). The currently implemented grid types are compatible with different ocean GCM grids including MOM5, MOM6 and the POP tripole grid.
* Grid types with vector Laplacians that can be used for filtering vector fields, such as horizontal velocity $(u,v)$. The currently implemented grid type is compatible with ocean GCM grids that use an Arakawa C-grid convention, e.g., MOM6.

Users are encouraged to contribute more grid types and Laplacians via pull requests if needed.

Another important goal of `GCM-Filters` is to enable computationally efficient filtering. The user can employ `GCM-Filters` on either CPUs or GPUs, with `NumPy` [@harris2020array] or `CuPy` [@cupy2017learningsys] input data. `GCM-Filters` leverages `Dask`[@dask] and `Xarray`[@hoyer2017xarray] to support filtering of larger-than-memory datasets and computational flexibility.

# Computational Efficiency


# Acknowledgements


# References
