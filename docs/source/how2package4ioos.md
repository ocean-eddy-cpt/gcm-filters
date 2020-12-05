# IOOS packaging guidelines

## Why not a cookie-cutter?

IOOS packages are developed by multiple partners and enforcing a
cookie-cutter would cause more harm than good.
That is why we prefer a set of guidelines instead.

This document describes these guidelines and how to implement them in an existing
project or a new one.

If you know enough about Python packaging just visit [https://github.com/ioos/ioos-python-package-skeleton](https://github.com/ioos/ioos-python-package-skeleton) and copy the bits that are relevant to your project.

## Project structure

Almost all python packages are structure as following:

```
|-docs
| |-source
| | |-_static
| |-build
|-tests
|-ioos_pkg_skeleton
|-notebooks
|-README.md
|-LICENSE.txt
```


Sometimes the `tests` folder goes inside the actual module.
We recommend that only if shipping the tests is important, e.g compiled modules.
If your module is pure Python it is fine to leave them outside of the package.

Always start writing tests and document from day 1!
We also recommend to write notebooks with examples that can be build as part of the docs.

Note that you should always have a README in your projects.
Markdown is a good format because it renders automatically on GitHub and is support on PyPI.

While IOOS does not recommend any particular license we prefer projects with OSI approved license,
the most common one is BSD-3-Clause.


## PEP 517/518

[PEP 517](https://www.python.org/dev/peps/pep-0517) species a build-system that is
independent of the format for source trees (your code).
The idea is to allow for other tools to become Python builds systems,
like `flit` and `poetry`,
via a minimum interface installation with `pip`.
PEP 517 really shines when combined with [PEP 518](https://www.python.org/dev/peps/pep-0518),
which specifies a minimum build system requirements via the `pyproject.toml` file.

```toml
[build-system]
requires = ["setuptools>=42", "wheel", "setuptools_scm[toml]>=3.4", "cython", "numpy"]
build-backend = "setuptools.build_meta"
```

When this file is present `pip` knows that it should install everything on requires before building the packages,
and knows that it should be built with `setuptools`.

The main advantages of using these PEPs together are:

- standardized non-executable config file;
- non-executable setup.py (safer installs without crazy workarounds);
- support many backends with one spec:
  - poetry, setuptools, pipenv(?), flit, conda, etc;
  - all should support pip installs.
- ensure that setup dependencies will be available at build time.


[This blog post](https://medium.com/@grassfedcode/pep-517-and-518-in-plain-english-47208ca8b7a6) contains a nice summary of these PEPs.

For IOOS packages we recommend to keep a bare bones `setup.py`,
for backwards compatibility, and to move all the package metadata to a `setup.cfg`,
while keeping the `pyproject.toml` only for the build information.

## `setup.py`

Most `setup.py` files can now be simplified to just the version handling and build call:

```python
from setuptools import setup

setup(
    use_scm_version={
        "write_to": "ioos_pkg_skeleton/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    }
)

```

The use of `versioneer` is deprecated because PEP 518 now ignores files in the root of the package.
One can work around it by adding the the root directory to the path in `setup.py`
but instead we recommend the use of `setuptools-scm`.
Just add `setuptools-scm` to your development/build dependencies and the lines above in your `setup.py` file.
The version will be automatically generated via tags and changes in your version control system.

PS: `setuptools-scm` can use a file or `pkg_resources` to get the version number.
We do not recommend `pkg_resources`because, depending on the number of packages installed,
that can lead to a significant overhead at import time.

## `setup.cfg`

While could use the `pyproject.toml` for most of your project configuration we recommend
to split that between the `setup.cfg` and `pyproject.toml` for readability.
The former will have the package metadata and tools configuration while the latter will specify the build system.

```cfg
[metadata]
name = ioos_pkg_skeleton
description = My Awesome module
author = AUTHOR NAME
author_email = AUTHOR@EMAIL.COM
url = https://github.com/ioos/ioos-python-package-skeleton
long_description = file: README.md
long_description_content_type = text/markdown
license = BSD-3-Clause
license_file = LICENSE.txt
classifiers =
    Development Status :: 5 - Production/Stable
    Intended Audience :: Science/Research
    Operating System :: OS Independent
    License :: OSI Approved :: BSD License
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    Topic :: Scientific/Engineering

[options]
zip_safe = False
install_requires =
    numpy
    requests
python_requires = >=3.6
packages = find:

[sdist]
formats = gztar

[check-manifest]
ignore =
    *.yml
    *.yaml
    .coveragerc
    docs
    docs/*
    *.enc
    notebooks
    notebooks/*
    tests
    tests/*

[flake8]
max-line-length = 105
select = C,E,F,W,B,B950
ignore = E203, E501, W503
exclude = ioos_pkg_skeleton/_version.py
```

The metadata and options fields are almost the same information that used to go in the `setup.py`.
In addition to the metadata we can use this file to write the configuration for
`flake8`, `check-manifest` and other tools.
We strongly recommend adding a `check-manifest` test to avoid shipping broken tarballs.
See more on tarball checks in the #travis-yaml section.

Note: `flake8` can be handled exclusively via #pre-commit-hooks.

## `MANIFEST.in`

Most of the problems we find with published tarballs is the lack of a required file at build time.
That is we why recommend `check-manifest` to help you write your `MANIFEST.in` file.
Here is an example that covers most cases:


```
include *.txt
include LICENSE # Please consider the Windows users and use .txt
include README.md

recursive-include ioos_pkg_skeleton *.py
```

## Do we still need a `requirements.txt` file?

Sadly yes, PEP 517/518 do not allow for non-python dependencies in the spec.
Even though the first two are already in the `setup.cfg` we cannot specify `libnetcdf`
and `libgdal` without an external file.

In order to make the package both `pip` and `conda` friendly we recommend to add the
external dependencies as comments and write a parser to read them in your CI,
or just duplicated them in the install section of your testing CI.

For example,

```
numpy
requests
#conda: libnetcdf
#conda: libgdal
```

One should also have a `requirements-dev.txt` with all the dependencies that are used to
build the package, build documents, and perform tests. For example:

```
# code style/consistency
black
flake8
flake8-builtins
flake8-comprehensions
flake8-mutable
flake8-print
isort
pylint
pytest-flake8
# checks and tests
check-manifest
pytest
pytest-cov
pytest-xdist
pre-commit
# documentation
doctr
nbsphinx
sphinx
# build
setuptools_scm
twine
wheel
```

## Continuous Integration

The easiest one to configure is Travis-CI,
while AzurePipelines are growing in popularity Travis-CI is easier to setup up and
maintain. The example below can be copied into your project with little modifications.

```yaml
language: minimal

sudo: false

env:
  global:
    - secure: "TOKEN"

matrix:
  fast_finish: true
  include:
    - name: "python-3.6"
      env: PY=3.6
    - name: "python-3.8"
      env: PY=3.8
    - name: "coding_standards"
      env: PY=3
    - name: "tarball"
      env: PY=3
    - name: "docs"
      env: PY=3

before_install:
  # Install miniconda and create TEST env.
  - |
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no --set show_channel_urls true
    conda update --quiet conda
    conda config --add channels conda-forge --force
    conda config --set channel_priority strict
    conda create --name TEST python=$PY --file requirements.txt --file requirements-dev.txt
    source activate TEST
    conda info --all

install:
  - pip install -e . --no-deps --force-reinstall

script:
  - if [[ $TRAVIS_JOB_NAME == python-* ]]; then
      cp -r tests/ /tmp ;
      pushd /tmp && pytest -n 2 -rxs --cov=ioos_pkg_skeleton tests && popd ;
    fi

  - if [[ $TRAVIS_JOB_NAME == 'tarball' ]]; then
      pip wheel . -w dist --no-deps ;
      check-manifest --verbose ;
      twine check dist/* ;
    fi

  - if [[ $TRAVIS_JOB_NAME == 'coding_standards' ]]; then
      pytest --flake8 -m flake8 ;
    fi

  - |
    if [[ $TRAVIS_JOB_NAME == 'docs' ]]; then
      set -e
      cp notebooks/tutorial.ipynb docs/source/
      pushd docs
      make clean html linkcheck
      popd
      if [[ -z "$TRAVIS_TAG" ]]; then
        python -m doctr deploy --build-tags --key-path github_deploy_key.enc --built-docs docs/_build/html dev
      else
        python -m doctr deploy --build-tags --key-path github_deploy_key.enc --built-docs docs/_build/html "version-$TRAVIS_TAG"
        python -m doctr deploy --build-tags --key-path github_deploy_key.enc --built-docs docs/_build/html .
      fi
    fi

```

This configuration sets a test matrix with multiple python versions, tarball tests, coding standards, and documentation builds.
The conda environment for the tests is created using the same requirement files as one would of with `pip` and the install section performs a simple `pip` installation to ensure everything works as expected on a user machine.

The test section will run all the items in the matrix if the conditions are met. Note that the documentation section will also build for latest version, development, and the tagged version.

Note that the tarball check is more than just the files in the manifest.
The command `pip wheel . -w dist --no-deps` creates the binary wheels and `twine check dist/*` ensures it is OK for publication on PyPI.

PS: one can create a local development environment using the same commands as the CI.
If you already have conda installed something like,

```
conda create --name TEST python=3.8 --file requirements.txt --file requirements-dev.txt
```

will create the testing env for Python 3.8.

## Configuring `pre-commit` locally and on GitHub Actions

With `pre-commit` we can run multiple checks every time we issue a new commit.
These checks can also be run on GitHub Actions as a CI.
This is useful when the contributors do not have `pre-commit` installed on their machine.

The configuration below can be dropped in the project root.
The checks selection are not comprehensive and not all of them are good for all the projects.
We will leave as an exercise to the reader to determine which ones are best for your project.

We do recommend `black` and `isort` for big projects with multiple contributors though because them help PR reviews by removing the code style from the equation.


```yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v3.1.0
  hooks:
    - id: trailing-whitespace
      exclude: tests/data
    - id: check-ast
    - id: debug-statements
    - id: end-of-file-fixer
    - id: check-docstring-first
    - id: check-added-large-files
    - id: requirements-txt-fixer
    - id: file-contents-sorter
      files: requirements-dev.txt

- repo: https://gitlab.com/pycqa/flake8
  rev: 3.7.9
  hooks:
    - id: flake8
      exclude: docs/source/conf.py
      args: [--max-line-length=105, --ignore=E203,E501,W503, --select=select=C,E,F,W,B,B950]

- repo: https://github.com/pre-commit/mirrors-isort
  rev: v4.3.21
  hooks:
  - id: isort
    additional_dependencies: [toml]
    args: [--project=ioos_pkg_skeleton, --multi-line=3, --lines-after-imports=2, --lines-between-types=1, --trailing-comma, --force-grid-wrap=0, --use-parentheses, --line-width=88]

- repo: https://github.com/asottile/seed-isort-config
  rev: v2.1.1
  hooks:
    - id: seed-isort-config

- repo: https://github.com/psf/black
  rev: stable
  hooks:
  - id: black
    language_version: python3

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v0.770
  hooks:
  - id: mypy
    exclude: docs/source/conf.py
    args: [--ignore-missing-imports]

```

In order to run them in every commit one must install them with:

```
pre-commit install
```

Two other handy commands are running to all files (if you are configuring it for an existing project):

```
pre-commit run --all-files
```

and ignoring it in a commit if you don't want it to run:

```
git commit ioos_pkg_skeleton/some-dot-pwhy.py --no-verify
```

## Github Actions

There is a lot that we can do with GitHub Actions, here we'll use it as an extra CI that will run the `pre-commit` checks. This is very useful when contributors do not have pre-commit configured in their machines.

This file should be in the `.github/workflows` folder.

```yaml
name: pre-commit

on:
  pull_request:
  push:
    branches: [master]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v1
    - uses: actions/setup-python@v1
    - name: set PY
      run: echo "::set-env name=PY::$(python --version --version | sha256sum | cut -d' ' -f1)"
    - uses: actions/cache@v1
      with:
        path: ~/.cache/pre-commit
        key: pre-commit|${{ env.PY }}|${{ hashFiles('.pre-commit-config.yaml') }}
    - uses: pre-commit/action@v1.0.0
```

## Summary

For IOOS project we divided the guidelines here in "Must Have" and "Nice to Have."

The Must Have list is:

```
- README
  - install instructions
- License
- docs
- unittest tests
- CIs
```

and the Nice to Have:

```
- automatic version number from tags (`setuptools-scm`)
- auto-publish docs and tarball (`sphinx` and `doctr`)
- tarball automated checks (`check-manifest`)
- standard style: `black`, lints (`flake8`), `isort`
- integration tests
- Windows Testing
- A package on both PyPI and conda-forge
```

## Extras

```
- CONTRIBUTING.rst
- .github/
```

Please check out https://www.pyopensci.org/

## TODO

- auto PyPI publication
https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/

- conda-forge publication
