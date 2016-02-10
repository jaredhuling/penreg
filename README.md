


## Introduction

Based on the following package: https://github.com/yixuan/ADMM

## Goals

* have capabilities for a wide range of penalties (lasso, group lasso, generalized lasso, overlapping group lasso, ridge) and combinations of them

* have capabilities for multiple models with any of the above penalties (gaussian, binomial, cox PH, etc)

* have capabilities for wide and tall data as well as sparse and dense matrices

* have some sort of unified interface to the fitting functions

* make the code as easily generalizable as possible

## Structure

Each routine (ie lasso, group lasso, genlasso, etc) should have the following files:

* **fitting_function.R** This function (see admm.lasso, admm.genlasso) is a wrapper for the c++ functions. This should return a class related to the routine type (ie class = "admm.genlasso")

* **predict.R** This function will be used for predicting based off of a model created from the above file. There should be one predict function for each fitting_function.R for example if admm.genlasso.R returns an object of type "admm.genlasso", then a function called predict.admm.genlasso.R should be created

* possibly more .R functions for each class (plot, summary, etc)

* **fittin_function.cpp** This file will contain a function to be directly called by R. examples include lasso.cpp and genlasso.cpp. It's primary job is to set up the data and then call the appropriate solver for each value of the tuning parameter

* **fitting_function_tall.h** This file is the routine which solves the optimization problem for data with nrow(x) > ncol(x) with admm (or even potentially some other algorithm). Typically with ADMM, high dimensional settings and low dimensional settings should be implemented differently for efficiency. 

* **fitting_function_wide.h** Same as the above but for high dimensional settings

* **fitting_function_sparse.cpp** and **fitting_function_tall_sparse.h** and **fitting_function_wide_sparse.h** Unless I figure out how to handle matrix types generically, when the x matrix itself is sparse I think code must be treated differently. ie we would need to set the data variable to be an Eigen::SparseMatrix<double> instead of an Eigen::MatrixXd. It is possible this can be handled with a generic type and then maybe we don't need extra files for sparse matrices.

The following files are more general:

* **DataStd.h** this centers/standardizes the data and provides code to recover the coefficients after fittin

* **FADMMBase.h** and/or **ADMMBase.h** these files set up the general class structure for admm problems. One of these two files will generally be used by the **fitting.h** files, for example, **FADMMBase.h** is used by **ADMMLassoTall.h** and **ADMMGenLassoTall.h** and **ADMMBase.h** is used by **ADMMLassoWide.h**. These files set up a class structure which is quite general. They **may** need to be modified in the future for problems that involve Newton-type iterations