
Contributor Guide
=================

Step-by-step
------------
This guide will take you through the necessary steps in order to contribute code to the repository.


1. Setup
^^^^^^^^^^^^^^^^^^
Fork the main repository, and clone your fork onto the machine of your choice.

Make sure to install and activate the environment with::

   conda create -n gcm-filters-env -c conda-forge --file requirements.txt --file requirements-dev.txt

If you find that `conda` takes a long time, try to install `mamba` with ``conda install mamba`` and then
and do::

   conda create -n gcm-filters-env -c conda-forge --file requirements.txt --file requirements-dev.txt


and finally activate the environment::

   conda activate gcm-filters-env

Before you can efficiently test your code you need to install this package in the now activated environment::

   pip install -e . --no-deps

For the linting also install `pre-commit <https://pre-commit.com>`_ now::

   pre-commit install

2. Write your code
^^^^^^^^^^^^^^^^^^
This one is pretty obvious 😁

3. Write and run tests
^^^^^^^^^^^^^^^^^^^^^^

Now figure out a good way to test the functionality of your code and run the test suite with::

   pytest

You will likely have to iterate here several times until all tests pass.

4. Linting and Docstrings
^^^^^^^^^^^^^^^^^^^^^^^^^
Once your tests pass, you want to make sure that all the code is formatted properly and has docstrings.

Run all the linters with::

   pre-commit run --all-files

Some things will automatically be reformatted, others need manual fixes. Follow the instructions in the terminal
until all checks pass.


Once you got everything to pass, you can stage and commit your changes and push them to the remote github repository. If you are not completely sure what this means,
you can find a git tutorial `here <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork>`_.
