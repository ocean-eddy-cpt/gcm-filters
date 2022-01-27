Filter Theory
=============

.. ipython:: python
    :suppress:

    import gcm_filters
    import numpy as np

The theory behind ``gcm-filters`` is described here at a high level.
For a more detailed treatment, see `Grooms et al. (2021) <https://doi.org/10.1029/2021MS002552>`_.

Filter Scale and Shape
----------------------

Any low-pass spatial filter should have a target length scale such that the filtered field keeps the part of the signal with length scales larger than the target length scale, and smoothes out smaller scales. In the context of this package the target length scale is called ``filter_scale``.

A spatial filter can also have a *shape* that determines how sharply it separates scales above and below the target length scale.
The filter shape can be thought of in terms of the kernel of a convolution filter

.. math:: \bar{f} = \int G(x - x')f(x') dx'

where :math:`f` is the function being filtered, :math:`G` is the filter kernel, and :math:`x'` is a dummy integration variable. (We note, however, that our filter is not *exactly* the same as a convolution filter. So our filter with a Gaussian target does not exactly produce the same results as a convolution against a Gaussian kernel on the sphere.)

This package currently has two filter shapes: ``GAUSSIAN`` and ``TAPER``.

.. ipython:: python

    list(gcm_filters.FilterShape)

For the ``GAUSSIAN`` filter the ``filter_scale`` equals :math:`\sqrt{12}\times` the standard deviation of the Gaussian.
\I.e. if you want to use a Gaussian filter with standard deviation L, then you should set ``filter_scale`` equal to L :math:`\times\sqrt{12}`.
This strange-seeming choice makes the Gaussian kernel have the same effect on large scales as a boxcar filter of width ``filter_scale``.
Thus ``filter_scale`` can be thought of as the "coarse grid scale" of the filtered field.

We can create a Gaussian filter as follows.

.. ipython:: python

    gaussian_filter = gcm_filters.Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    gaussian_filter

Once the filter has been constructed, the method ``plot_shape`` can be used to plot the shape of the target filter and the approximate filter.

.. ipython:: python
    :okwarning:

    @savefig gaussian_shape.png
    gaussian_filter.plot_shape()

The distinction between the target filter and the approximate filter will be discussed below.

.. note:: ``plot_shape`` does not plot the shape of the filter *kernel*. Instead, it plots the frequency response of the filter for each wavenumber :math:`k`.
    In other words, the plot shows how the filter attenuates different scales in the data.
    Length scales are related to wavenumbers by :math:`\ell = 2\pi/k`.
    The filter leaves large scales unchanged, so the plot shows values close to 1 for small :math:`k`.
    The filter damps out small scales, so the plots shows values close to 0 for large :math:`k`.

The definition of the ``TAPER`` filter is more complex, but the ``filter_scale`` has the same meaning: it corresponds to the width of a qualitatively-similar boxcar filter.

.. ipython:: python

    taper_filter = gcm_filters.Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.TAPER,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    taper_filter

.. ipython:: python
    :okwarning:

    @savefig taper_shape.png
    taper_filter.plot_shape()


The plot above shows that the ``TAPER`` filter is more scale-selective than the Gaussian filter; it does a better job of leaving scales larger than the filter scale unchanged, and removing scales smaller than the filter scale.
The drawbacks of the ``TAPER`` filter are that it requires higher computational cost for the same filter scale (due to a higher number of necessary filter steps, see below), and it can produce negative values for the filtered field even when the unfiltered field is positive.

The Taper filter has a tunable parameter ``transition_width`` that controls how sharply the filter separates scales above and below the filter scale.
``transition_width`` = 1 would be the same as a complete *projection* onto the large scales, leaving the small scales completely zeroed out.
This would require a very high computational cost, and is not at all recommended!
The default is ``transition_width`` = :math:`\pi`.
Larger values for ``transition_width`` reduce the cost and the likelihood of producing negative values from positive data, but make the filter less scale-selective. In the example below, we choose ``transition_width`` = :math:`2\pi`.

.. ipython:: python

    wider_taper_filter = gcm_filters.Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.TAPER,
        transition_width=2*np.pi,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    wider_taper_filter

.. ipython:: python
    :okwarning:

    @savefig wider_taper_shape.png
    wider_taper_filter.plot_shape()

.. note:: The Taper filter is similar to the `Lanczos filter <https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml>`_.
    Both are 1 for a range of large scales and 0 for a range of small scales, with a transition in between.
    The difference is in the transition region: in the transition region the Lanczos filter is straight line connecting 1 and 0, while the Taper filter is a smoother cubic.
    The Lanczos filter is typically described in terms of its "half-power cutoff wavelength"; the Taper filter can be similarly described.
    The half-power cutoff wavelength for the Taper filter with a ``filter_scale`` of :math:`L` and a ``transition_width`` of :math:`X` is :math:`2LX/(X+1)`.


Filter Steps
------------

The filter goes through several steps to produce the final filtered field.
There are two different kinds of steps: *Laplacian* and *Biharmonic* steps.
At each Laplacian step, the filtered field is updated using the following formula

.. math:: \bar{f} \leftarrow \bar{f} + \frac{1}{s_{j}}\Delta \bar{f}

The filtered field is initialized to :math:`\bar{f}=f` and :math:`\Delta` denotes a discrete Laplacian.
At each Biharmonic step, the filtered field is updated using

.. math:: \bar{f}\leftarrow \bar{f}+\frac{2R\{s_j\}}{|s_j|^2} \Delta\bar{f} + \frac{1}{|s_j|^2}\Delta^2\bar{f}

where :math:`R\{\cdot\}` denotes the real part of a complex number.

The total number of steps, ``n_steps``, and the values of :math:`s_j` are automatically selected by the code to produce the desired filter scale and shape.
If the filter scale is much larger than the grid scale, many steps are required.
Also, the Taper filter requires more steps than the Gaussian filter for the same ``filter_scale``; in the above examples the Taper filters required ``n_steps`` = 16, but the Gaussian filter only ``n_steps`` = 5.

The code allows users to set their own ``n_steps``.
Biharmonic steps are counted as 2 steps because their cost is approximately twice as much as a Laplacian step.
So with ``n_steps`` = 3 you might get one Laplacian plus one biharmonic step, or three Laplacian steps.
(The user cannot choose how ``n_steps`` is split between Laplacian and Biharmonic steps; that split is set internally in the code.)

For example, the user might want to use a smaller number of steps to reduce the cost. The caveat is that the accuracy will be reduced, so the filter might not act as expected: it may not have the right shape or the right length scale. To illustrate this, we create a new filter with a smaller number of steps than the default ``n_steps`` = 16, and plot the result.

.. ipython:: python
    :okwarning:

    taper_filter_8steps = gcm_filters.Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.TAPER,
        n_steps=8,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    taper_filter_8steps

.. ipython:: python
    :okwarning:

    @savefig taper_8steps_shape.png
    taper_filter_8steps.plot_shape()


The example above shows that using ``n_steps`` = 8 still yields a very accurate approximation of the target filter, at half the cost of the default. The main drawback in this example is that the filter slightly *amplifies* large scales, which also implies that it will not conserve variance.

The example below shows what happens with ``n_steps`` = 4.

.. ipython:: python
    :okwarning:

    taper_filter_4steps = gcm_filters.Filter(
        filter_scale=4,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.TAPER,
        n_steps=4,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    taper_filter_4steps

.. ipython:: python
    :okwarning:

    @savefig taper_4steps_shape.png
    taper_filter_4steps.plot_shape()


.. warning::

    For this example of a Taper filter with a filter factor of 4, ``n_steps = 4`` is simply not enough to get a good approximation of the target filter. The ``taper_filter_4steps`` object created here will still "work" but it will not behave as expected; specifically, it will smooth more than expected - it will act like a filter with a larger filter scale.

The minimum number of steps is 3; if ``n_steps`` is not set by the user, or if it is set to a value less than 3, the code automatically changes ``n_steps`` to the default value.


Numerical Stability
-------------------

When the filter scale is much larger than the grid scale the filter can become unstable to roundoff errors.
The usual manifestation of these roundoff errors is high-amplitude small-scale noise in the filtered field.
(This problem is worse for the Taper filter than the Gaussian filter.)

.. tip::
    In such cases, the user has a few options to try to regain stability.

    1. If the data being filtered is single-precision, it might help to promote it to double precision (or higher) before filtering.
    2. If a user is encountering instability with the standard Gaussian filter, the user can try setting ``n_iterations`` to an integer greater than 1. Read more about this in :doc:`factored_gaussian`.
    3. The user can also try reducing ``n_steps``, but must not reduce it too much or the resulting filter will not behave as expected.
    4. Users might elect to *coarsen* their data before filtering, i.e. to reduce the resolution of the input data before applying the filter. This has the effect of increasing the grid size, and thus decreasing the gap between the filter scale and the grid scale.
    5. The final option is simply to use a different approach to filtering, not based on ``gcm-filters``.

:doc:`examples/example_numerical_instability` has an example of numerical instability, as well as examples of avoiding the instability by increasing the precision and coarsening the data.

Spatially-Varying Filter Scale
------------------------------

In the foregoing discussion the filter scale is fixed over the physical domain.
It is possible to vary the filter scale over the domain by introducing a *diffusivity* :math:`\kappa`.
(This diffusivity is nondimensional.)
The Laplacian steps are altered to

.. math:: \bar{f} \leftarrow \bar{f} + \frac{1}{s_{j}}\nabla\cdot(\kappa\nabla \bar{f})

and the Biharmonic steps are similarly altered by replacing :math:`\Delta` with :math:`\nabla\cdot(\kappa\nabla)`.
With :math:`\kappa` the *local* filter scale is :math:`\sqrt{\kappa}\times` ``filter_scale``.
For reasons given in `Grooms et al. (2021) <https://doi.org/10.1029/2021MS002552>`_, we require :math:`\kappa\le 1`, and at least one place in the domain where :math:`\kappa = 1`.
Thus, when using variable :math:`\kappa`, ``filter_scale`` sets the *largest* filter scale in the domain and the local filter scale can be reduced by making :math:`\kappa<1`.

Suppose, for example, that you want the local filter scale to be :math:`L(x,y)`.
You can achieve this in ``gcm-filters`` as follows.

1. Set ``filter_scale`` equal to the maximum of :math:`L(x,y)` over the domain. (Call this value :math:`L_{max}`).
2. Set :math:`\kappa` equal to :math:`L(x,y)^2/L_{max}^2`.

:doc:`examples/example_filter_types` has examples of filtering with spatially-varying filter scale.

Anisotropic Filtering
---------------------

It is possible to have different filter scales in different directions, and to have both the scales and directions vary over the domain.
This is achieved by replacing :math:`\kappa` in the previous section with a :math:`2\times2` symmetric and positive definite matrix (for a 2D domain), i.e. replacing :math:`\Delta` with :math:`\nabla\cdot(\mathbf{K}\nabla)`.
``gcm-filters`` currently only supports diagonal :math:`\mathbf{K}`, i.e. the principal axes of the anisotropic filter are aligned with the grid, so that the user only inputs one :math:`\kappa` for each grid direction, rather than a full :math:`2\times2` matrix.
Just like in the previous section, we require that each of these two :math:`\kappa` be less than or equal to 1, and the interpretation is also the same: the local filter scale in a particular direction is :math:`\sqrt{\kappa}\times` ``filter_scale``.

Suppose, for example, that you want to filter with a scale of 60 in the grid-x direction and a scale of 30 in the grid-y direction.
Then you would set ``filter_scale`` =  60, with :math:`\kappa_x = 1` to get a filter scale of 60 in the grid-x direction.
Next, to get a filter scale of 30 in the grid-y direction you would set :math:`\kappa_y=1/4`.

The :doc:`examples/example_filter_types` has examples of anisotropic filtering.

.. _Fixed factor filtering:

Fixed Factor Filtering
----------------------

:doc:`examples/example_filter_types` also shows methods designed specifically for the case where the user wants to set the local filter scale equal to a multiple :math:`m` of the local grid scale to achieve a fixed *coarsening* factor.
This can be achieved using the anisotropic diffusion described in the previous section.

An alternative way to achieve filtering with fixed coarsening factor :math:`m` is what we refer to as **simple fixed factor filtering**. This method is somewhat ad hoc, and *not* equivalent to fixed factor filtering via anisotropic diffusion. On the upside, simple fixed factor filtering is often significantly faster and yields very similar results in practice, as seen in :doc:`examples/example_filter_types`. The code handles simple fixed factor filtering as follows:

1. It multiplies the unfiltered data by the local grid cell area.
2. It applies a filter with ``filter_scale`` = :math:`m` *as if* the grid scale were uniform.
3. It divides the resulting field by the local grid cell area.

The first step is essentially a coordinate transformation where your original (locally orthogonal) grid is transformed to a uniform Cartesian grid with :math:`dx = dy = 1`. The third step is the reverse coordinate transformation.

.. note:: The three steps above are handled internally by ``gcm-filters`` if the user chooses one of the following grid types:

   * ``REGULAR_AREA_WEIGHTED``
   * ``REGULAR_WITH_LAND_AREA_WEIGHTED``
   * ``TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED``

   together with ``filter_scale`` = :math:`m` and ``dx_min`` = 1. (For simple fixed factor filtering, only ``dx_min`` on the transformed uniform grid matters; and here we have ``dx_min`` = 1). Read more about the different grid types in :doc:`basic_filtering`.


Filtering Vectors
-----------------

In Cartesian geometry the Laplacian of a vector field can be obtained by taking the Laplacian of each component of the vector field, so vector fields can be filtered as described in the foregoing sections.
On smooth manifolds, the Laplacian of a vector field is not the same as the Laplacian of each component of the vector field.
Users may wish to use a **vector Laplacian** to filter vector fields.
The filter is constructed in exactly the same way; the only difference is in how the Laplacian is defined.
Rather than taking a scalar field and returning a scalar field, the vector Laplacian takes a vector field as input and returns a vector field.
To distinguish this from the scalar Laplacian, we refer to the filter based on a scalar Laplacian as a *diffusion-based* filter and the filter based on a vector Laplacian as a *viscosity-based* filter.
:doc:`examples/example_vector_laplacian` has examples of viscosity-based filtering.
