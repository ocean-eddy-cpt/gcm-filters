Grid Types and Filter Object
============================

The core object in GCM-Filters is the ``gcm_filters.Filter`` object.
When creating a ``gcm_filters.Filter`` object, we need to specify how we want to smooth the data, including the filter scale, filter shape, and all relevant grid parameters.

Grid types
----------

To define a filter, we need to pick a grid and associated Laplacian that matches our data.
The currently implemented grid types are:

.. ipython:: python

    list(gcm_filters.GridType)

This list will grow as we implement more Laplacians.

The following table provides an overview of

+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+
| ``GridType``                   | Grid             | Handles land | Boundary condition | Laplacian type   | Example                              |
+================================+==================+==============+====================+==================+======================================+
| ``REGULAR``                    | Cartesian grid   | no           | periodic           | Scalar Laplacian |                                      |
+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+
| ``REGULAR_WITH_LAND``          | Cartesian grid   | yes          | periodic           | Scalar Laplacian | :doc:`examples/example_tripole_grid` |
+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+
| ``IRREGULAR_WITH_LAND``        | any Arakawa grid | yes          | periodic           | Scalar Laplacian | `here`                               |
+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+
| ``TRIPOLAR_REGULAR_WITH_LAND`` | Cartesian grid   | yes          | tripole            | Scalar Laplacian | `here`                               |
+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+
| ``TRIPOLAR_POP_WITH_LAND``     | Arakawa C-Grid   | yes          | tripole            | Scalar Laplacian | `here`                               |
+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+
| ``VECTOR_C_GRID``              | Arakawa C-Grid   | yes          | periodic           | Vector Laplacian | `here`                               |
+--------------------------------+------------------+--------------+--------------------+------------------+--------------------------------------+

Grid types with scalar Laplacians can be used for filtering scalar fields (such as temperature), and grid types with vector Laplacians can be used for filtering vector fields (such as velocity).

.. note::

    ``REGULAR``, ``REGULAR_WITH_LAND``, and ``TRIPOLAR_REGULAR_WITH_LAND`` can also be used for non-Cartesian grids, but only for simple fixed factor filtering - a simplified filtering method where the filter scale is tied to the local grid scale. Simple fixed factor filtering transforms a locally orthogonal grid to a Cartesian grid and filters the transformed data with a fixed factor, see more details here.

Grid variables
--------------

Each grid type from the above list has different “grid variables” that must be provided. To find out what these are, we can use this utility function.

.. ipython:: python

    gcm_filters.required_grid_vars(gcm_filters.GridType.REGULAR_WITH_LAND)

So if we use this grid type, we have to include a ``wet_mask`` grid variable. This is a binary array representing the topography on our grid. Here the convention is that the array is 1 in the ocean (“wet points”) and 0 on land (“dry points”).

.. ipython:: python

    import xarray as xr

    ny, nx = (128, 256)

    mask_data = np.ones((ny, nx))
    mask_data[(ny // 4):(3 * ny // 4), (nx // 4):(3 * nx // 4)] = 0
    wet_mask = xr.DataArray(mask_data, dims=['y', 'x'])


Creating the Filter Object
--------------------------

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


Applying the Filter
-------------------
