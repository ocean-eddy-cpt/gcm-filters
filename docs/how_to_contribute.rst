
Contributor Guide
=================

Step-by-step
------------
This guide will take you through the necessary steps in order to contribute code to the repository.


1. Setup
^^^^^^^^
If you are not super familiar with the terminology of `forking`, `pull request` etc, here is a `git tutorial <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork>`_ to get you started.

Fork the main repository, and clone your fork onto the machine of your choice.

Make sure to install and activate the environment with::

   conda create -n gcm-filters-env -c conda-forge --file requirements.txt --file requirements-dev.txt

If you find that `conda` takes a long time, try to install `mamba` with ``conda install mamba`` and then
and do::

   mamba create -n gcm-filters-env -c conda-forge --file requirements.txt --file requirements-dev.txt


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


Once you got everything to pass, you can stage and commit your changes and push them to the remote github repository.

How to change the documentation
-------------------------------

In order to build the documentation locally you should build and activate the docs environment::

   mamba env create -f docs/environment.yml

   conda activate gcm-filters-docs

Then navigate to the docs folder and build the docs locally with::

   cd docs

   make html

Once that is done you can open the created html files in `docs/_build/index.html` with your webbrowser::

   open _build/index.html

You can then edit and rebuild the docs until you are satisfied and submit a PR as usual.
