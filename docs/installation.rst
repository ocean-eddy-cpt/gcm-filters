Installation
------------

Requirements
^^^^^^^^^^^^

GCM-Filters is compatible with python 3. It requires xarray_, numpy_, and dask_.

Installation from conda forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GCM-Filters can be installed via conda forge::

    conda install -c conda-forge gcm_filters


Installation from pip
^^^^^^^^^^^^^^^^^^^^^

GCM-Filters can also be installed with pip::

    pip install gcm_filters

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.


Installation from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^

GCM-Filters is under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/ocean-eddy-cpt/gcm-filters>`_
and install it::

    git clone https://github.com/ocean-eddy-cpt/gcm-filters.git
    cd gcm-filters
    python setup.py install

or simply::

    pip install git+https://github.com/ocean-eddy-cpt/gcm-filters.git

Users are encouraged to `fork <https://help.github.com/articles/fork-a-repo/>`_
GCM-Filters and submit issues_ and `pull requests`_.

.. _dask: http://dask.pydata.org
.. _numpy: https://numpy.org
.. _xarray: http://xarray.pydata.org
.. _issues: https://github.com/ocean-eddy-cpt/gcm-filters/issues
.. _`pull requests`: https://github.com/ocean-eddy-cpt/gcm-filters/pulls


How to run the example notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run the example notebooks in this documentation, you will need a few extra dependencies that you can install via::

   conda env create -f docs/environment.yml

   conda activate gcm-filters-docs
