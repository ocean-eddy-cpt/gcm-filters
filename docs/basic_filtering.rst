Basic Filtering
================

The Filter Object
------------------


The core object in GCM-Filters is the :py:class:`gcm_filters.Filter` object. Its full documentation below enumerates all possible options.

.. autofunction:: gcm_filters.Filter


Details related to ``filter_scale``, ``filter_shape``, ``transition_width``, and ``n_steps`` can be found in the :doc:`theory`.
The following sections explain the options for ``grid_type`` and ``grid_vars`` in more detail.

Grid types
----------

To define a filter, we need to pick a grid and associated Laplacian that matches our data.
The currently implemented grid types are:

.. ipython:: python

    import gcm_filters
    list(gcm_filters.GridType)

This list will grow as we implement more Laplacians.

The following table provides an overview of these different grid type options: what grid they are suitable for, whether they handle land (i.e., continental boundaries), what boundary condition the Laplacian operators use, and whether they come with a scalar or vector Laplacian. You can also find links to example usages.

+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``GridType``                   | Grid                                                            | Handles land | Boundary condition | Laplacian type   | Example                                  |
+================================+=================================================================+==============+====================+==================+==========================================+
| ``REGULAR``                    | Cartesian grid                                                  | no           | periodic           | Scalar Laplacian |                                          |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``REGULAR_WITH_LAND``          | Cartesian grid                                                  | yes          | periodic           | Scalar Laplacian | see below                                |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``IRREGULAR_WITH_LAND``        | locally orthogonal grid                                         | yes          | periodic           | Scalar Laplacian | :doc:`examples/example_filter_types`;    |
|                                |                                                                 |              |                    |                  | :doc:`examples/example_tripole_grid`     |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``MOM5U``                      | Velocity-point on Arakawa B-Grid                                | yes          | periodic           | Scalar Laplacian |                                          |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``MOM5T``                      | Tracer-point on Arakawa B-Grid                                  | yes          | periodic           | Scalar Laplacian |                                          |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``TRIPOLAR_POP_WITH_LAND``     | locally orthogonal grid                                         | yes          | tripole            | Scalar Laplacian | :doc:`examples/example_tripole_grid`     |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+
| ``VECTOR_C_GRID``              | `Arakawa C-Grid <https://en.wikipedia.org/wiki/Arakawa_grids>`_ | yes          | periodic           | Vector Laplacian | :doc:`examples/example_vector_laplacian` |
+--------------------------------+-----------------------------------------------------------------+--------------+--------------------+------------------+------------------------------------------+

Grid types with scalar Laplacians can be used for filtering scalar fields (such as temperature), and grid types with vector Laplacians can be used for filtering vector fields (such as velocity).

Grid types for simple fixed factor filtering
++++++++++++++++++++++++++++++++++++++++++++

The remaining grid types are for a special type of filtering: **simple fixed factor filtering** to achieve a fixed *coarsening* factor (see also the :doc:`theory`). If you specify one of the following grid types for your data, ``gcm_filters`` will internally transform your original (locally orthogonal) grid to a uniform Cartesian grid with `dx = dy = 1`, and perform fixed factor filtering on the uniform grid. After this is done, ``gcm_filters`` transforms the filtered field back to your original grid.
In practice, this coordinate transformation is achieved by area weighting and deweighting (see :doc:`theory`). This is why the following grid types have the suffix ``AREA_WEIGHTED``.

+-----------------------------------------------+-------------------------+--------------+--------------------+------------------+--------------------------------------+
| ``GridType``                                  | Grid                    | Handles land | Boundary condition | Laplacian type   | Example                              |
+===============================================+=========================+==============+====================+==================+======================================+
| ``REGULAR_AREA_WEIGHTED``                     | locally orthogonal grid | no           | periodic           | Scalar Laplacian |                                      |
+-----------------------------------------------+-------------------------+--------------+--------------------+------------------+--------------------------------------+
| ``REGULAR_WITH_LAND_AREA_WEIGHTED``           | locally orthogonal grid | yes          | periodic           | Scalar Laplacian | :doc:`examples/example_filter_types`;|
|                                               |                         |              |                    |                  | :doc:`examples/example_tripole_grid` |
+-----------------------------------------------+-------------------------+--------------+--------------------+------------------+--------------------------------------+
| ``TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED``  | locally orthogonal grid | yes          | tripole            | Scalar Laplacian | :doc:`examples/example_tripole_grid` |
+-----------------------------------------------+-------------------------+--------------+--------------------+------------------+--------------------------------------+


Grid variables
--------------

Each grid type from the above two tables has different *grid variables* that must be provided as Xarray `DataArrays <http://xarray.pydata.org/en/stable/user-guide/data-structures.html>`_. For example, let's assume we are on a Cartesian grid (with uniform grid spacing equal to 1), and we want to use the grid type ``REGULAR_WITH_LAND``. To find out what the required grid variables for this grid type are, we can use this utility function.

.. ipython:: python

    gcm_filters.required_grid_vars(gcm_filters.GridType.REGULAR_WITH_LAND)

``wet_mask`` is a binary array representing the topography on our grid. Here the convention is that the array is 1 in the ocean (“wet points”) and 0 on land (“dry points”).

.. ipython:: python

    import numpy as np
    import xarray as xr

    ny, nx = (128, 256)

    mask_data = np.ones((ny, nx))
    mask_data[(ny // 4):(3 * ny // 4), (nx // 4):(3 * nx // 4)] = 0
    wet_mask = xr.DataArray(mask_data, dims=['y', 'x'])

.. ipython:: python
    :okwarning:

    @savefig wet_mask.png
    wet_mask.plot()

We have made a big island.

Creating the Filter Object
--------------------------

We create a filter object as follows.

.. ipython:: python

    filter = gcm_filters.Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.TAPER,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={'wet_mask': wet_mask}
    )
    filter

The string representation for the filter object in the last line includes some of the parameters it was initiliazed with, to help us keep track of what we are doing.
We have created a Taper filter that will filter our data by a fixed factor of 4.

Applying the Filter
-------------------

We can now apply the filter object that we created above to some data. Let's create a random 3D cube of data that matches our grid.

.. ipython:: python

    nt = 10
    data = np.random.rand(nt, ny, nx)
    da = xr.DataArray(data, dims=['time', 'y', 'x'])
    da

We now mask our data with the ``wet_mask``.

.. ipython:: python

   da_masked = da.where(wet_mask)

.. ipython:: python
    :okwarning:

    @savefig data.png
    da_masked.isel(time=0).plot()

Now that we have some data, we can apply our filter. We need to specify which dimension names to apply the filter over. In this case, it is ``y``, ``x``.

.. warning:: The dimension order matters! Since some filters deal
    with anisotropic grids, the latitude / y dimension must appear first
    in order to obtain the correct result. That is not an issue for this simple
    (isotropic) toy example but needs to be kept in mind for applications on
    real GCM grids.

.. ipython:: python

    %time da_filtered = filter.apply(da_masked, dims=['y', 'x'])

.. ipython:: python

    da_filtered

Let's visualize what the filter did.

.. ipython:: python
    :okwarning:

    @savefig data_filtered.png
    da_filtered.isel(time=0).plot()


Using Dask
-----------

Up to now, we have filtered *eagerly*; when we called ``.apply``, the results were computed immediately and stored in memory.
``GCM-Filters`` is also designed to work seamlessly with Dask array inputs. With `dask <https://dask.org/>`_, we can filter *lazily*, deferring the filter computations and possibly executing them in parallel.
We can do this with our synthetic data by converting them to dask.

.. ipython:: python
    :okwarning:

    da_dask = da_masked.chunk({'time': 2})
    da_dask

We now filter our data lazily.

.. ipython:: python
    :okwarning:

    da_filtered_lazy = filter.apply(da_dask, dims=['y', 'x'])
    da_filtered_lazy

Nothing has actually been computed yet.
We can trigger computation as follows:

.. ipython:: python

    %time da_filtered_computed = da_filtered_lazy.compute()

Here we got only a very modest speedup because our example data are too small. For bigger data, the performance benefit will be more evident.
