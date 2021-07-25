KL-pgmpy
========

This repository contains the code implemented for computing the Kullback-Leibler divergence method between two
networks with different structure but defined on the same set of variables. The code uses **pgmpy** library version
0.13 although version 0.15 is already available at **pgmpy** web page.

The code for KL computation is stored in folder **pgmpy/kltools**. This folder contains the following files:
- bayesBall.py: code for making relevance analysis and determining the variables required for a certain query.
- factorsRepsitory.py: code and functions for storing and managing factors used during KL computation.
- operationsRespository: software for managing and storing the operations required for KL computation.
- qualitativeVariableEliminationKL.py: this file contains software employed for making symbolic propagation on a Bayesian network in order to determine the operations plan required for computing the posteriors for a given set of queries.
- qualitativeBayesianModel.py: software for storing and managing a symbolic version if the Bayesian network to analyze.
- qualitativeEliminationOrder.py: adds auxiliary functions for symbolic propagation.
- qualitativeFactor.py: more auxiliary functions for symbolic propagation.
- utilityFunctions.py: some functions of general use.
- variableEliminationKL.py: classes and functions for performing queries and computing KL divergence.

A folder named **checks** contains functions for testing purposes. The folder named **methodAnalysis** adds
utility methods for analyzing and computing KL methods. The two methods implemented are presented in the paper
titled **Computation of Kullback-Leibler Divergence in Bayesian Networks** in the following order: **method1** (based on evidence propagation) and **method2** (based on using repositories of operations and factors) for clarity reasons. However, in the implementation methods are named as **alt1** for alternative using repositories and **alt2** for method based on evidence propagation. Folder **methodAnalysis** contains the following files:
klComputationWithJoint.py: testing software for computing the KL divergence by means of the joint distribution. This method can be used only of small size networks.
- klAnalysisSingleNetAlt1-1Engine.py: it uses the main method of computation with the method using repositories of operations and folder. It uses a single repository in order to compute the whole set of queries and therefore it offers the more efficient computation. It computes KL divergence, executiion time and some interest measures about the performance of the algorithm.
- klAnalysisSingleNetAlt2.py: same as the previous one but for the method based on evidence propagtion. In this case the only additional information about the algorithm performance is the maximum size of factors required for computing the divergence.
- klComputationAlt1-1Engine.py: repeats a given number of times the algorithm in order to get an reliable estimation of the computation time. It uses the version with a single repository.
- klComputationAlt1-2Engine.py: the same as previous one, but using an alternative version using two repositories (one for base model and another for the alternative one).
- klComputationAlt2.py: same as previous ones, but using the second method based on evidence propagation.
- klComputationComparison1Engine.py and klComputationComparison2Engines.py: executes both methods for getting time comparisons as presented in the paper.

The simplest way of executing this software from the command line is as follows:
- go to base folder of the project
- set PYTHONPATH environment to this folder
- execute the python script using (for example, the one for klComputationComparison1Engine.py) using the command:

> python pgmpy/kltools/methodAnalysis/klComputationComparison1Engine.py asia

The argument of the call is the name of the netwotk of interest. Available base and alternative mthods are stored in **nets** folder.

pgmpy
=====
[![Build Status](https://api.travis-ci.com/pgmpy/pgmpy.svg?branch=dev)](https://travis-ci.com/pgmpy/pgmpy)
[![Appveyor](https://ci.appveyor.com/api/projects/status/github/pgmpy/pgmpy?branch=dev)](https://www.appveyor.com/)
[![codecov](https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg)](https://codecov.io/gh/pgmpy/pgmpy)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/78a8256c90654c6892627f6d8bbcea14)](https://www.codacy.com/gh/pgmpy/pgmpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pgmpy/pgmpy&amp;utm_campaign=Badge_Grade)
[![Downloads](https://img.shields.io/pypi/dm/pgmpy.svg)](https://pypistats.org/packages/pgmpy)
[![Join the chat at https://gitter.im/pgmpy/pgmpy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pgmpy/pgmpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

pgmpy is a python library for working with Probabilistic Graphical Models.  

Documentation  and list of algorithms supported is at our official site http://pgmpy.org/  
Examples on using pgmpy: https://github.com/pgmpy/pgmpy/tree/dev/examples  
Basic tutorial on Probabilistic Graphical models using pgmpy: https://github.com/pgmpy/pgmpy_notebook  

Our mailing list is at https://groups.google.com/forum/#!forum/pgmpy .

We have our community chat at [gitter](https://gitter.im/pgmpy/pgmpy).

Dependencies
=============
pgmpy has following non optional dependencies:
- python 3.6 or higher
- networkX
- scipy 
- numpy
- pytorch

Some of the functionality would also require:
- tqdm
- pandas
- pyparsing
- statsmodels
- joblib

Installation
=============
pgmpy is available both on pypi and anaconda. For installing through anaconda use:
```bash
$ conda install -c ankurankan pgmpy
```

For installing through pip:
```bash
$ pip install -r requirements.txt  # only if you want to run unittests
$ pip install pgmpy
```

To install pgmpy from the source code:
```
$ git clone https://github.com/pgmpy/pgmpy 
$ cd pgmpy/
$ pip install -r requirements.txt
$ python setup.py install
```

If you face any problems during installation let us know, via issues, mail or at our gitter channel.

Development
============

Code
----
Our latest codebase is available on the `dev` branch of the repository.

Contributing
------------
Issues can be reported at our [issues section](https://github.com/pgmpy/pgmpy/issues).

Before opening a pull request, please have a look at our [contributing guide](
https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md)

Contributing guide contains some points that will make our life's easier in reviewing and merging your PR.

If you face any problems in pull request, feel free to ask them on the mailing list or gitter.

If you want to implement any new features, please have a discussion about it on the issue tracker or the mailing
list before starting to work on it.

Testing
-------

After installation, you can launch the test form pgmpy
source directory (you will need to have the ``pytest`` package installed):
```bash
$ pytest -v
```
to see the coverage of existing code use following command
```
$ pytest --cov-report html --cov=pgmpy
```

Documentation and usage
=======================

The documentation is hosted at: http://pgmpy.org/

We use sphinx to build the documentation. To build the documentation on your local system use:
```
$ cd /path/to/pgmpy/docs
$ make html
```
The generated docs will be in _build/html

Examples
========
We have a few example jupyter notebooks here: https://github.com/pgmpy/pgmpy/tree/dev/examples
For more detailed jupyter notebooks and basic tutorials on Graphical Models check: https://github.com/pgmpy/pgmpy_notebook/

Citing
======
Please use the following bibtex for citing `pgmpy` in your research:
```
@inproceedings{ankan2015pgmpy,
  title={pgmpy: Probabilistic graphical models using python},
  author={Ankan, Ankur and Panda, Abinash},
  booktitle={Proceedings of the 14th Python in Science Conference (SCIPY 2015)},
  year={2015},
  organization={Citeseer}
}
```

License
=======
pgmpy is released under MIT License. You can read about our license at [here](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)

