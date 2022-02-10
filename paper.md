---
title: 'GCM-Filters: A Python Package for Diffusion-based Spatial Filtering of Gridded Data'
tags:
  - Python
  - ocean modeling
  - climate modeling
  - fluid dynamics
authors:
  - name: Nora Loose
    orcid: 0000-0002-3684-9634
    affiliation: 1
  - name: Ryan Abernathey
    orcid: 0000-0001-5999-4917
    affiliation: 2
  - name: Ian Grooms
    orcid: 0000-0002-4678-7203
    affiliation: 1
  - name: Julius Busecke
    orcid: 0000-0001-8571-865X
    affiliation: 2
  - name: Arthur Guillaumin
    orcid: 0000-0003-1571-4228
    affiliation: 3
  - name: Elizabeth Yankovsky
    orcid: 0000-0003-3612-549X
    affiliation: 3
  - name: Gustavo Marques
    orcid: 0000-0001-7238-0290
    affiliation: 4
  - name: Jacob Steinberg
    orcid: 0000-0002-2609-6405
    affiliation: 5
  - name: Andrew Slavin Ross
    orcid: 0000-0002-2368-6979
    affiliation: 3
  - name: Hemant Khatri
    orcid: 0000-0001-6559-9059
    affiliation: 6
  - name: Scott Bachman
    orcid: 0000-0002-6479-4300
    affiliation: 4
  - name: Laure Zanna
    orcid: 0000-0002-8472-4828
    affiliation: 3
  - name: Paige Martin
    orcid: 0000-0003-3538-633X
    affiliation: "2, 7"
affiliations:
  - name: Department of Applied Mathematics, University of Colorado Boulder, Boulder, CO, USA
    index: 1
  - name: Lamont-Doherty Earth Observatory, Columbia University, New York, NY, USA
    index: 2
  - name: Courant Institute of Mathematical Sciences, New York University, New York, NY, USA
    index: 3
  - name: Climate and Global Dynamics Division, National Center for Atmospheric Research, Boulder, CO, USA
    index: 4
  - name: Woods Hole Oceanographic Institution, Woods Hole, MA, USA
    index: 5
  - name: Earth, Ocean and Ecological Sciences, University of Liverpool, UK
    index: 6
  - name: Research School of Earth Sciences, Australian National University, Canberra, Australia
    index: 7

date: 1 November 2021
bibliography: paper.bib

---

# Summary

`GCM-Filters` is a python package that allows scientists to perform spatial filtering analysis in an easy, flexible and efficient way. The package implements the filtering method based on the discrete Laplacian operator that was introduced by @grooms2021diffusion. The filtering algorithm is analogous to smoothing via diffusion; hence the name *diffusion-based filters*. `GCM-Filters` can be used with either gridded observational data or gridded data that is produced by General Circulation Models (GCMs) of ocean, weather, and climate. Spatial filtering of observational or GCM data is a common analysis method in the Earth Sciences, for example to study oceanic and atmospheric motions at different spatial scales or to develop subgrid-scale parameterizations for ocean models.

`GCM-Filters` provides filters that are highly configurable, with the goal to be useful for a wide range of scientific applications. The user has different options for selecting the filter scale and filter shape.
The filter scale can be defined in several ways: a fixed length scale (e.g., 100 km), a scale tied to a model grid scale (e.g., 1$^\circ$), or a scale tied to a varying dynamical scale (e.g., the Rossby radius of deformation). As an example, \autoref{fig1} shows unfiltered and filtered relative vorticity, where the filter scale is set to a model grid scale of 4$^\circ$. `GCM-Filters` also allows for anisotropic, i.e., direction-dependent, filtering.
Finally, the filter shape -- currently: either Gaussian or Taper -- determines how sharply the filter separates scales above and below the target filter scale.

![(Left) Snapshot of unfiltered surface relative vorticity  $\zeta = \partial_x v - \partial_y u$ from a global 0.1$^\circ$ simulation with MOM6 [@adcroft2019MOM6]. (Right) Relative vorticity filtered to 4$^\circ$, obtained by applying `GCM-Filters` to the field $\zeta$ on the left. The plots are made with `matplotlib` [@Hunter2007] and `cartopy` [@Cartopy].\label{fig1}](filtered_vorticity.png){ width=100% }

# Statement of Need

Spatial filtering is commonly used as a scientific tool for analyzing gridded data. An example of an existing spatial filtering tool in python is the `ndimage.gaussian_filter` function in `SciPy` [@2020SciPy-NMeth], implemented as a sequence of convolution filters. While being a valuable tool for image processing (or blurring), `SciPy`'s Gaussian filter is of limited use for GCM data; it assumes a regular and rectangular Cartesian grid, employs a simple boundary condition, and the definitions of filter scale and shape have little or no flexibility. The python package `GCM-Filters` is specificially designed to filter GCM data, and seeks to solve a number of challenges for the user:

1. GCM data comes on irregular curvilinear grids with spatially varying grid-cell geometry.
2. Continental boundaries require careful and special treatment when filtering ocean GCM output.
3. Earth Science applications benefit from configurable filters, where the definition of filter scale and shape is flexible.
4. GCM output is often too large to process in memory, requiring distributed and / or delayed execution.

The `GCM-Filters` algorithm [@grooms2021diffusion] applies a discrete Laplacian to smooth a field through an iterative process that resembles diffusion. The discrete Laplacian takes into account the varying grid-cell geometry and uses a no-flux boundary condition, mimicking how diffusion is internally implemented in GCMs. The no-flux boundary conditions ensures that the filter preserves the integral: $\int_{\Omega} \bar{f}(x,y) \,dA = \int_{\Omega} f (x,y)\, dA$, where $f$ is the original field, $\bar{f}$ the filtered field, and $\Omega$ the ocean domain. Conservation of the integral is a desirable filter property for many physical quantities, for example energy or ocean salinity. More details on the filter properties can be found in @grooms2021diffusion.

An important goal of `GCM-Filters` is to enable computationally efficient filtering. The user can employ `GCM-Filters` on either CPUs or GPUs, with `NumPy` [@harris2020array] or `CuPy` [@cupy2017learningsys] input data. `GCM-Filters` leverages `Dask` [@dask] and `Xarray` [@hoyer2017xarray] to support filtering of larger-than-memory datasets and computational flexibility.

# Usage

The main `GCM-Filters` class that the user will interface with is the `gcm_filters.Filter` object. When creating a filter object, the user specifies how they want to smooth their data, including the desired filter shape and filter scale. At this stage, the user also picks the grid type that matches their GCM data, given a predefined list of grid types. Each grid type has an associated discrete Laplacian, and requires different *grid variables* that the user must provide (the latter are usually available to the user as part of the GCM output). Currently, `GCM-Filters` provides a number of different grid types and associated discrete Laplacians:

* Grid types with **scalar Laplacians** that can be used for filtering scalar fields, for example temperature or vorticity (see \autoref{fig1}). The currently implemented grid types are compatible with different ocean GCM grids including MOM5 [@mom5], MOM6 [@adcroft2019MOM6] and the POP2 [@pop2] tripole grid.
* Grid types with **vector Laplacians** that can be used for filtering vector fields, for example horizontal velocity $(u,v)$. The currently implemented grid type is compatible with ocean GCM grids that use an Arakawa C-grid convention; examples include MOM6 [@adcroft2019MOM6] and the MITgcm [@mitgcm].

Atmospheric model grids are not yet supported, but could be implemented in `GCM-Filters`. Users are encouraged to contribute more grid types and Laplacians via pull requests.
While we are excited to share `GCM-Filters` at version `0.2.1`, we plan to continue improving and maintaining the package for the long run and welcome new contributors from the broader community.

# Acknowledgements

This work was supported by the National Science Foundation grants OCE 1912302, OCE 1912325, OCE 1912332, OCE 1912420, GEO 1912357, and the NOAA grant CVP NA19OAR4310364.
Busecke received support from the Gordon and Betty Moore Foundation.
This research is supported in part by the generosity of Eric and Wendy Schmidt by recommendation of Schmidt Futures, as part of its Virtual Earth System Research Institute (VESRI).

# References
