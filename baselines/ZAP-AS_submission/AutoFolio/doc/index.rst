.. AutoFolio documentation master file, created by
   sphinx-quickstart on Mon Sep 14 12:36:21 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AutoFolio documentation!
=================================
AutoFolio is an algorithm selection tool,
i.e., selecting a well-performing algorithm for a given instance [Rice 1976].
In contrast to other algorithm selection tools,
users of AutoFolio are bothered with the decision which algorithm selection approach to use
and how to set its hyper-parameters.
AutoFolio uses one of the state-of-the-art algorithm configuration tools, namely SMAC [Hutter et al LION'16]
to automatically determine a well-performing algorithm selection approach
and its hyper-parameters for a given algorithm selection data.
Therefore, AutoFolio has a robust performance across different algorithm selection tasks.

.. note::

   For a detailed description of its main idea,
   we refer to

	`JAIR Journal Article <http://aad.informatik.uni-freiburg.de/papers/15-JAIR-Autofolio.pdf>`_
	
	@ARTICLE{lindauer-jair15a,
	  author    = {M. Lindauer and H. Hoos and F. Hutter and T. Schaub},
	  title     = {AutoFolio: An automatically configured Algorithm Selector},
	  volume    = {53},
	  journal   = {Journal of Artificial Intelligence Research},
	  year      = {2015},
	  pages     = {745-778}
	}


AutoFolio is mainly written in Python 3.5.

.. note::

    This version is a re-implementation of the original AutoFolio implementation
    and has not the same configuration space of the original implementation -- 
    e.g., the clustering approach was not re-implementation because the performance had not met our expectations;
    e.g., as a new approach we implemented pair-wise performance difference prediction approach.


Contents:
---------
.. toctree::
   :maxdepth: 2

   installation
   manual
   contact
   license



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

