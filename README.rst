.. -*- mode: rst -*-

|Travis|_ |Zenodo|_ |Codecov|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-hcp.png?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-hcp

.. |Zenodo| image:: https://zenodo.org/badge/53261823.svg
.. _Zenodo: https://zenodo.org/badge/latestdoi/53261823

.. |Codecov| image:: http://codecov.io/github/mne-tools/mne-hcp/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/mne-tools/mne-hcp?branch=master

MNE-CAMCAN
==========

We provide Python tools for seamless integration of MEG data from the `Cambridge Centre for Ageing and Neuroscience (Cam-CAN) Database  <http://www.cam-can.org>`_ into the Python ecosystem.
In only a few lines of code, complex data retrieval requests can be readily executed on the resources from this neuroimaging reference dataset.
Raw CAMCAN data are translated into actionable MNE objects that we know and love.
MNE-CAMCAN abstracts away difficulties due to diverging coordinate systems, distributed information, and file format conventions. Providing a simple and consistent access to HCP MEG data will facilitate emergence of standardized data analysis practices.
By building on the `MNE software package <http://martinos.org/mne/>`_, MNE-HCP naturally supplements a fast growing stack of Python data science toolkits.

Fast interface to MEG data
--------------------------


Scope and Disclaimer
--------------------
This code is under active research-driven development. The API is still changing,
but is getting closer to a stable release.

.. note::

    For now please consider the following caveats:

    - We only intend to support a subset of the files shipped with CAMCAN.
    - Specifically, for now it is not planned to support io and processing for any outputs of the CAMCAN source space pipelines.
    - This library breaks with some of MNE conventions in order to make the camcan outputs compatible with MNE.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_, which comes with the necessary dependencies. Alternatively, to install ``mne-camcan``, you first need to install its dependencies::

	$ pip install numpy matplotlib scipy scikit-learn mne joblib pandas

Then clone the repository::

	$ git clone http://github.com/mne-tools/mne-camcan

and finally run `setup.py` to install the package::

	$ cd mne-camcan/
	$ python setup.py install

If you do not have admin privileges on the computer, use the ``--user`` flag
with `setup.py`.

Alternatively, for a devoloper install based on symbolic links (which simplifies keeping up with code changes), do::

	$ cd mne-camcan/
	$ python setup.py develop

To check if everything worked fine, you can do::

	$ python -c 'import camcan'

and it should not give any error messages.

Dependencies
------------

The following main and additional dependencies are required to use MNE-camcan:

    - MNE-Python master branch
    - the MNE-Python dependencies, specifically
        - scipy
        - numpy
        - matplotlib
    - scikit-learn (optional)


Acknowledgements
================

This project is supported by the Amazon Webservices Research grant issued to Denis A. Engemann and Sheraz Khan.


We acknowledge support by Alex Gramfort and Eric Larson for discussions, inputs and help with finding the best way to map
CAMCAN data to the MNE world.
