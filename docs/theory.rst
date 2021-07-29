Filter Theory
=============

The theory behind ``gcm-filters`` is described here at a high level.
For a more detailed treatment, see `Grooms et al. (2021) <https://doi.org/10.1002/essoar.10506591.1>`_.

Filter Scale and Shape
----------------------

Any low-pass spatial filter should have a target length scale such that the filtered field keeps the part of the signal with length scales larger than the target length scale, and smoothes out smaller scales. In the context of this package the target length scale is called ``filter_scale``.

A spatial filter can also have a *shape* that determines how sharply it separates scales above and below the target length scale.
The filter shape can be thought of in terms of the kernel of a convolution filter

.. math:: \bar{f} = \int G(x - x')f(x') dx'

where :math:`f` is the function being filtered, :math:`G` is the filter kernel, and :math:`x'` is a dummy integration variable.
This package currently has two filter shapes: ``GAUSSIAN`` and ``TAPER``.
For the Gaussian filter the ``filter_scale`` equals :math:`\sqrt{12}\times` the standard deviation of the Gaussian.
\I.e. if you want to use a Gaussian filter with standard deviation L, then you should set ``filter_scale`` equal to L :math:`\times\sqrt{12}`.
This strange-seeming choice makes the Gaussian kernel have the same effect on large scales as a boxcar filter of width ``filter_scale``.
Thus ``filter_scale`` can be thought of as the "coarse grid scale" of the filtered field.

The definition of the "Taper" filter is more complex, but the ``filter_scale`` has the same meaning: it corresponds to the width of a qualitatively-similar boxcar filter.
The Taper filter is more scale-selective than the Gaussian filter; it does a better job of leaving scales larger than the filter scale unchanged, and removing scales smaller than the filter scale.
The drawback is that it requires higher computational cost for the same filter scale, and can produce negative values for the filtered field even when the unfiltered field is positive.

The Taper filter has a tunable parameter ``transition_width`` that controls how sharply the filter separates scales above and below the filter scale.
``transition_width`` = 1 would be the same as a complete *projection* onto the large scales, leaving the small scales completely zeroed out.
This would require a very high computational cost, and is not at all recommended!
The default is ``transition_width`` = :math:`\pi`.
Larger values reduce the cost and the likelihood of producing negative values from positive data, but make the filter less scale-selective.

Filter Steps
------------

The filter goes through several steps to produce the final filtered field.
There are two different kinds of steps: "Laplacian" and "Biharmonic" steps.
At each Laplacian step, the filtered field is updated using the following formula

.. math:: \bar{f} \leftarrow \bar{f} + \frac{1}{s_{j}}\Delta \bar{f}

The filtered field is initialized to :math:`\bar{f}=f` and :math:`\Delta` denotes a discrete Laplacian.
At each Biharmonic step, the filtered field is updated using

.. math:: \bar{f}\leftarrow \bar{f}+\frac{2R\{s_j\}}{|s_j|^2}\Delta\bar{f} + \frac{1}{|s_j|^2}\Delta^2\bar{f}

where :math:`R\{\cdot\}` denotes the real part of a complex number.

The total number of steps and the values of :math:`s_j` are automatically selected by the code to produce the desired filter scale and shape.
If the filter scale is much larger than the grid scale, many steps are required.
Also, the Taper filter requires more steps than the Gaussian filter for the same ``filter_scale``.

The code allows users to set their own number of steps ``n_steps``.
The minimum number of steps is 3; if ``n_steps`` is not set by the user, or if it is set to a value less than 3, the code automatically changes ``n_steps`` to the default value.
If the number of steps is set too low the filter will not behave as expected: it may not have the right shape or the right length scale.

Biharmonic steps are counted as 2 steps because their cost is approximately twice as much as a laplacian step.
So with ``n_steps`` = 3 you might get one Laplacian plus one biharmonic step, or three Laplacian steps.
(The user cannot choose how ``n_steps`` is split between Laplacian and Biharmonic steps; that split is set internally in the code.)

Once a filter object has been constructed, the method ``plot_shape`` can be used to plot the shape of the target filter and the approximate filter.
This can be particularly useful if the user is trying to reduce ``n_steps`` from its default value without introducing sigificant errors.
``plot_shape`` does not plot the shape of the filter *kernel*.
Instead, it plots the frequency response of the filter for each wavenumber :math:`k`.
Length scales are related to wavelengths by :math:`\ell = 2\pi/k`.
The filter leaves large scales unchanged, so ``plot_shape`` shows values close to 1 for small :math:`k`.
The filter damps out small scales, so ``plot_shape`` shows values close to 0 for large :math:`k`.
The :doc:`tutorial` gives examples of using ``plot_shape`` and interpreting the resulting plots.

Numerical Stability
-------------------

When the filter scale is much larger than the grid scale the filter can become unstable to roundoff errors.
The usual manifestation of these roundoff errors is high-amplitude small-scale noise in the filtered field.
(This problem is worse for the Taper filter than the Gaussian filter.)
In such cases user has a few options to try to regain stability.

1. If the data being filtered is single-precision, it might help to promote it to double precision (or higher) before filtering.
2. The user can also try reducing `n_steps`, but must not reduce it too much or the resulting filter will not behave as expected.
3. Users might elect to *coarsen* their data before filtering, i.e. to reduce the resolution of the input data before applying the filter. This has the effect of increasing the grid size, and thus decreasing the gap between the filter scale and the grid scale.
4. The final option is simply to use a different approach to filtering, not based on ``gcm-filters``.

:doc:`tutorial_numerical_instability` has an example of numerical instability, as well as examples of avoiding the instability by increasing the precision and coarsening the data.

Spatially-Varying Filter Scale
------------------------------

In the foregoing discussion the filter scale is fixed over the physical domain.
It is possible to vary the filter scale over the domain by introducing a "diffusivity" :math:`\kappa`.
(This "diffusivity" is nondimensional.)
The Laplacian steps are altered to

.. math:: \bar{f} \leftarrow \bar{f} + \frac{1}{s_{j}}\nabla\cdot(\kappa\nabla \bar{f})

and the Biharmonic steps are similarly altered by replacing :math:`\Delta` with :math:`\nabla\cdot(\kappa\nabla)`.
With :math:`\kappa` the *local* filter scale is :math:`\sqrt{\kappa}\times` ``filter_scale``.
For reasons given in `Grooms et al. (2021) <https://doi.org/10.1002/essoar.10506591.1>`_, we require :math:`\kappa\le 1`, and at least one place in the domain where :math:`\kappa = 1`.
Thus, when using variable :math:`\kappa`, ``filter_scale`` sets the *largest* filter scale in the domain and the local filter scale can be reduced by making :math:`\kappa<1`.

Suppose, for example, that you want the local filter scale to be :math:`L(x,y)`.
You can achieve this in ``gcm-filters`` as follows.

1. Set ``filter_scale`` equal to the maximum of :math:`L(x,y)` over the domain. (Call this value :math:`L_{max}`).
2. Set :math:`\kappa` equal to :math:`L(x,y)^2/L_{max}^2`.

The :doc:`tutorial_filter_types` has examples of filtering with spatially-varying filter scale.

Anisotropic Filtering
---------------------

It is possible to have different filter scales in different directions, and to have both the scales and directions vary over the domain.
This is achieved by replacing :math:`\kappa` in the previous section with a :math:`2\times2` symmetric and positive definite matrix (for a 2D domain), i.e. replacing :math:`\Delta` with :math:`\nabla\cdot(\mathbf{K}\nabla)`.
``gcm-filters`` currently only supports diagonal :math:`\mathbf{K}`, i.e. the principal axes of the anisotropic filter are aligned with the grid, so that the user only inputs one :math:`\kappa` for each grid direction, rather than a full :math:`2\times2` matrix.
Just like in the previous section, we require that each of these two :math:`\kappa` be less than or equal to 1, and the interpretation is also the same: the local filter scale in a particular direction is :math:`\sqrt{\kappa}\times` ``filter_scale``.

Suppose, for example, that you want to filter with a scale of 60 in the grid-x direction and a scale of 30 in the grid-y direction.
Then you would set ``filter_scale`` =  60, with :math:`\kappa_x = 1` to get a filter scale of 60 in the grid-x direction.
Next, to get a filter scale of 30 in the grid-y direction you would set :math:`\kappa_y=1/4`.

The :doc:`tutorial_filter_types` has examples of anisotropic filtering. The same tutorial also shows methods designed specifically for the case where the user wants to set the local filter scale equal to the local grid scale to achieve a fixed "coarsening" factor.
This can be achieved using the anisotropic diffusion described above, but it can also be achieved in a more efficient computational manner as follows.

1. Multiply the unfiltered data by the local grid cell area.
2. Apply the filter *as if* the grid scale were uniform, i.e. tell the filter that the grid spacings are all equal to 1.
3. Divide the resulting field by the local grid cell area.

This somewhat ad hoc method is not equivalent to the one described above, but in practice it yields very similar results and is often significantly faster.

Filtering Vectors
-----------------

In Cartesian geometry the Laplacian of a vector field can be obtained by taking the Laplacian of each component of the vector field, so vector fields can be filtered as described in the foregoing sections.
On smooth manifolds, the Laplacian of a vector field is not the same as the Laplacian of each component of the vector field.
Users may wish to use a vector Laplacian to filter vector fields.
The filter is constructed in exactly the same way; the only difference is in how the Laplacian is defined.
Rather than taking a scalar field and returning a scalar field, the vector Laplacian takes a vector field as input and returns a vector field.
To distinguish this from the scalar Laplacian, we refer to the filter based on a scalar Laplacian as a "diffusion-based" filter and the filter based on a vector Laplacian as a "viscosity-based" filter.
:doc:`tutorial_vector_laplacian` has examples of viscosity-based filtering.
