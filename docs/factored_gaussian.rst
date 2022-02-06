Factoring the Gaussian Filter
=============================

This section provides background on applying a Gaussian filter with a large filter scale by repeatedly applying Gaussian filters with a smaller filter scale.
This was not discussed in `Grooms et al. (2021) <https://doi.org/10.1029/2021MS002552>`_, so this section provides extra detail.

The :py:class:`gcm_filters.Filter` class has an argument ``n_iterations`` that will automatically split a Gaussian filter into a sequence of Gaussian filters with smaller scales, each of which is less sensitive to roundoff errors. If a user is encountering instability with the standard Gaussian filter, the user can try setting ``n_iterations`` to an integer greater than 1. In general ``n_iterations`` should be set to the smallest value that avoids numerical instability. The user should input their desired filter scale, and then the code will automatically select a smaller filter scale for each of the constituent filters in such a way that the final result achieves the filter scale input by the user.

.. ipython:: python

    factored_gaussian_filter_x2 = gcm_filters.Filter(
        filter_scale=40,
        dx_min=1,
        n_iterations=2,  # number of constituent filters
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    factored_gaussian_filter_x2

.. ipython:: python

    factored_gaussian_filter_x3 = gcm_filters.Filter(
        filter_scale=40,
        dx_min=1,
        n_iterations=3,  # number of constituent filters
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    factored_gaussian_filter_x3


.. note:: When ``n_iterations`` is greater than 1, ``n_steps`` is the number of steps in a single small-scale Gaussian filter. The total number of steps taken by the algorithm is the product of ``n_iterations`` and ``n_steps``. If ``n_steps`` is not set by the user, the code automatically changes ``n_steps`` to a default value that gives an accurate approximation to the small-scale Gaussian filters that comprise the factored filter. In the example above, the code determined a smaller ``n_steps`` for the second filter compared to the first filter because the filter scale for each of the constituent filters is smaller.

Applying the Taper filter repeatedly is not equivalent to applying it once with a different scale, so this method only works for the Gaussian filter.

Mathematical Background
-----------------------

Per equation (13) of `Grooms et al. (2021) <https://doi.org/10.1029/2021MS002552>`_, the data to be filtered can be thought of as a vector :math:`\vec{f}` that can be expanded in an orthonormal basis of eigenfunctions of the discrete Laplacian

.. math:: \vec{f} = \sum_i \hat{f}_i\vec{q}_i

where :math:`\vec{q}_i` are the orthonormal basis vectors.
Each basis vector is associated with a wavenumber :math:`k_i` and a length scale :math:`2\pi/k_i`.
The filtered data can also be expanded in the same basis

.. math:: \sum_i \hat{f}_ip(k_i^2)\vec{q}_i

The polynomial :math:`p` approximates the target filter.
If you filter the data :math:`N` times using the same filter, then the result has the following expansion

.. math:: \sum_i \hat{f}_i(p(k_i^2))^N\vec{q}_i

For a Gaussian filter, the target filter is

.. math:: p(k^2) \approx g_L(k) = \text{exp}\left\{-\frac{L^2}{24}k^2\right\}

where :math:`L` is the ``filter_scale``.
The Gaussian target has the nice property that :math:`g_L^N = g_{\sqrt{N}L}`, i.e. if you apply the Gaussian filter :math:`N` times with scale :math:`L`, the result is the same as if you applied the Gaussian filter once with scale :math:`\sqrt{N}L`.
The way the code uses this is that instead of filtering once with scale :math:`L`, it has the option to filter :math:`N` times with scale :math:`L/\sqrt{N}`.

Inexact Equivalence
-------------------

Unfortunately, because we're using a polynomial approximation rather than an exact Gaussian filter, filtering once with scale :math:`L` is not exactly the same as filtering :math:`N` times with scale :math:`L/\sqrt{N}`.
The difference between the filtered field obtained using a single Gaussian versus :math:`N` factored Gaussians is exactly

.. math:: \sum_i \hat{f}_i(p_1(k_i^2) - (p_N(k^2))^N)\vec{q}_i

where :math:`p_1(k^2)` is the polynomial that approximates :math:`g_{L}(k)` and :math:`p_N(k^2)` is the polynomial that approximates :math:`g_{L/\sqrt{N}}(k)`.
We can bound the 2-norm of this error as follows.
For :math:`p_1` we have the error

.. math:: g_L(k) = p_1(k^2) + e_1(k)
    :label: unfactored-error

and for :math:`p_N` we have the error

.. math:: g_{L/\sqrt{N}}(k) = p_N(k^2) + e_N(k)

The aforementioned property of the Gaussian implies that

.. math:: g_{L}(k) = (p_N(k^2) + e_N(k))^N = (p_N(k^2))^N + N (p_N(k^2))^{N-1} e_N(k) + \ldots + (e_N(k))^N
    :label: factored-error

(using the binomial expansion.)
Subtracting :eq:`factored-error` from :eq:`unfactored-error` gives us an expression for the difference between the polynomial approximation with scale :math:`L` and the factored approximation using :math:`N` filters each with scale :math:`L/\sqrt{N}`:

.. math:: p_1(k^2) - (p_N(k^2))^N = - e_1(k) + N p_N(k^2)^{N-1} e_N(k) + \ldots + e_N(k)^N \sim - e_1(k) + N p_N(k^2)^{N-1} e_N(k)

where the last expression is in the limit of small errors :math:`|e_1(k)|` and :math:`|e_N(k)|` with :math:`N` fixed.
The difference in the two filtered fields is thus

.. math:: \sum_i \hat{f}_i(p_1(k_i^2) - (p_N(k_i^2))^N)\vec{q}_i\sim\sum_i \hat{f}_i(- e_1(k_i) + N p_N(k_i^2)^{N-1} e_N(k_i))\vec{q}_i

and the squared norm of this asymptotic approximation is exactly

.. math :: \sum_i \hat{f}_i^2(- e_1(k) + N p_N(k^2)^{N-1} e_N(k))^2.

The default choice of ``n_steps`` implies that :math:`|e_1(k)|` and :math:`|e_N(k)|` are both less than about 0.01, and the approximating polynomial is approximately bounded between 0 and 1.
Together these imply that

.. math :: (- e_1(k) + N p_N(k^2)^{N-1} e_N(k))^2 < 0.0001 (1+N)^2

The squared norm of the difference in the filtered fields is thus approximately bounded by

.. math :: 0.0001 (1+N)^2 \sum_i \hat{f}_i^2 = 0.0001(1+N)^2\|\vec{f}\|^2

The norm of the difference divided by the norm of the unfiltered field is thus approximately bounded by :math:`0.01(1+N)`.
This is why :math:`N` should be chosen as small as possible while avoiding numerical instability: as :math:`N` increases the difference between applying the filter once vs :math:`N` times increases.

Closing Comments
----------------

Note that the same ideas can be used to bound the norm of the difference between the filtered field that would be obtained using the exact filter :math:`g`, and the filtered field obtained using the polynomial approximation with :math:`N=1`.
In this case the analysis is simpler and the result is that the norm of the difference divided by the norm of the unfiltered field is bounded by 0.01.
Since this doesn't rely on factoring the filter, this bound is true for both the Gaussian and Taper filters.
