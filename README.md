





## Introduction

Based on the following package: https://github.com/yixuan/ADMM

Install using the **devtools** package:

```r
devtools::install_github("jaredhuling/penreg")
```

or by cloning and building 

## Goals

* have capabilities for a wide range of penalties (lasso, group lasso, generalized lasso, overlapping group lasso, ridge) and combinations of them

* have capabilities for multiple models with any of the above penalties (gaussian, binomial, cox PH, etc)

* have capabilities for wide and tall data as well as sparse and dense matrices

* have some sort of unified interface to the fitting functions

* make the code as easily generalizable as possible

## Current Problems

* How to best organize code for ADMM algorithm fo fitting multiple penalties at once?

* How to best organize code for fitting different models with penalties. linear models are simple, but what about binomial and other nonlinear models. Is ADMM best or should we use Newton iterations with ADMM on the inside fitting weighted penalize least squares models

* How to best incorporate sparse design matrices (ie when x itself is sparse). Standardization of the data cannot be used for this case, because it will ruin the sparsity of the matrix

## Immediate to-do list

* Incorporate weighting of observations for lasso and genlasso

* Incorporate a penalty factor multiplier (for all the methods. currently complete for coord mcp and admm lasso) which is a vector multiplied by the tuning parameter value so that individual variables can have their penalty upweighted, downweighted, or not penalized at all. Could also be used for adaptive lasso, etc

* complete different families for all methods

* Make automatic rho choice better in terms of convergence

* After the above are complete, work on group lasso 

* make the block soft thresholding function more efficient

## Code Structure

Each routine (ie lasso, group lasso, genlasso, etc) should have the following files:

* **fitting_function.R** This function (see admm.lasso, admm.genlasso) is a wrapper for the c++ functions. This should return a class related to the routine type (ie class = "admm.genlasso"). This function should have the proper roxygen description commented directly above its definition. Make sure to use **@export** in order for it to be included in the namespace. When building the package in rstudio, make sure roxygen is used.

* **predict.R** This function will be used for predicting based off of a model created from the above file. There should be one predict function for each fitting_function.R for example if admm.genlasso.R returns an object of type "admm.genlasso", then a function called predict.admm.genlasso.R should be created

* possibly more .R functions for each class (plot, summary, etc)

* **fitting_function.cpp** This file will contain a function to be directly called by R. examples include lasso.cpp and genlasso.cpp. It's primary job is to set up the data and then call the appropriate solver for each value of the tuning parameter

* **fitting_function_tall.h** This file is the routine which solves the optimization problem for data with nrow(x) > ncol(x) with admm (or even potentially some other algorithm). Typically with ADMM, high dimensional settings and low dimensional settings should be implemented differently for efficiency. 

* **fitting_function_wide.h** Same as the above but for high dimensional settings

* **fitting_function_sparse.cpp** and **fitting_function_tall_sparse.h** and **fitting_function_wide_sparse.h** Unless I figure out how to handle matrix types generically, when the x matrix itself is sparse I think code must be treated differently. ie we would need to set the data variable to be an Eigen::SparseMatrix<double> instead of an Eigen::MatrixXd. It is possible this can be handled with a generic type and then maybe we don't need extra files for sparse matrices.

The following files are more general:

* **DataStd.h** this centers/standardizes the data and provides code to recover the coefficients after fittin

* **FADMMBase.h** and/or **ADMMBase.h** these files set up the general class structure for admm problems. One of these two files will generally be used by the **fitting.h** files, for example, **FADMMBase.h** is used by **ADMMLassoTall.h** and **ADMMGenLassoTall.h** and **ADMMBase.h** is used by **ADMMLassoWide.h**. These files set up a class structure which is quite general. They **may** need to be modified in the future for problems that involve Newton-type iterations

## Models

### Lasso

```r
library(glmnet)
library(penreg)
set.seed(123)
n <- 100
p <- 10
m <- 5
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, mean = 1.2, sd = 2), n, p)
y <- 5 + x %*% b + rnorm(n)

fit         <- glmnet(x, y, thresh = 1e-10, standardize = FALSE)
glmnet_coef <- coef(fit)[,floor(length(fit$lambda/2))]
admm.res    <- admm.lasso(x, y, standardize = FALSE, intercept = TRUE, 
                          lambda = fit$lambda, abs.tol = 1e-8, rel.tol = 1e-8)
admm_coef   <- admm.res$beta[,floor(length(fit$lambda/2))]

data.frame(glmnet = as.numeric(glmnet_coef),
           admm   = as.numeric(admm_coef))
```

```
##         glmnet        admm
## 1   4.82478472  4.82478467
## 2   0.21479611  0.21479630
## 3   0.80214687  0.80214691
## 4   0.41866035  0.41866057
## 5   0.96235001  0.96234996
## 6   0.83101295  0.83101306
## 7  -0.01913640 -0.01913657
## 8   0.05623805  0.05623766
## 9  -0.09109605 -0.09109590
## 10  0.12805350  0.12805339
## 11  0.02799320  0.02799327
```

### Lasso (logistic regression)

```r
library(glmnet)
library(penreg)
set.seed(123)
n <- 100
p <- 10
m <- 5
b <- matrix(c(runif(m, min = -0.1, max = 0.1), rep(0, p - m)))
x <- matrix(rnorm(n * p, mean = 1.2, sd = 2), n, p)
y <- rbinom(n, 1, prob = 1 / (1 + exp(-x %*% b)))

fit         <- glmnet(x, y, family = "binomial", standardize = FALSE, thresh = 1e-12)
glmnet_coef <- coef(fit)[,floor(length(fit$lambda/2))]
admm.res    <- admm.lasso(x, y, intercept = TRUE, family = "binomial", lambda = fit$lambda, irls.tol = 1e-8)
admm_coef   <- admm.res$beta[,floor(length(fit$lambda/2))]

data.frame(glmnet = as.numeric(glmnet_coef),
           admm   = as.numeric(admm_coef))
```

```
##          glmnet        admm
## 1  -0.387291651 -0.38730951
## 2  -0.126237875 -0.12623663
## 3   0.215399880  0.21540136
## 4   0.002287391  0.00228826
## 5   0.068306065  0.06830715
## 6   0.079371247  0.07937165
## 7  -0.107363235 -0.10736164
## 8   0.085989643  0.08599049
## 9   0.046324036  0.04632551
## 10  0.047907553  0.04790905
## 11  0.088285296  0.08828688
```

## Performance

### Lasso


```r
library(microbenchmark)
library(penreg)
library(glmnet)
# compute the full solution path, n > p
set.seed(123)
n <- 10000
p <- 500
m <- 50
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)

lambdas = glmnet(x, y)$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-10)}, # thresh must be very low for glmnet to be accurate
    "admm[lasso]"   = {res2 <- admm.lasso(x, y, lambda = lambdas, 
                                          intercept = TRUE, standardize = TRUE,
                                          abs.tol = 1e-8, rel.tol = 1e-8)},
    "cd[lasso]"     = {res3 <- cd.lasso(x, y, lambda = lambdas, 
                                        intercept = TRUE, standardize = TRUE,
                                        tol = 1e-5)},
    times = 5
)
```

```
## Unit: milliseconds
##           expr       min        lq      mean    median        uq      max
##  glmnet[lasso] 1022.6184 1027.2837 1198.5852 1155.5844 1374.6038 1412.836
##    admm[lasso]  767.0904  814.4545  866.1616  817.3545  895.7443 1036.164
##      cd[lasso] 2128.6531 2142.7474 2390.2293 2182.4856 2481.9032 3015.357
##  neval cld
##      5  a 
##      5  a 
##      5   b
```

```r
# difference of results
max(abs(coef(res1) - res2$beta))
```

```
## [1] 6.609837e-07
```

```r
max(abs(coef(res1) - res3$beta))
```

```
## [1] 6.848958e-07
```

```r
max(abs(res2$beta - res3$beta))
```

```
## [1] 8.204803e-08
```

```r
set.seed(123)
n <- 10000
p <- 100
m <- 25
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)

## Logistic Regression
y <- rbinom(n, 1, prob = 1 / (1 + exp(-x %*% b)))

lambdas = glmnet(x, y, family = "binomial")$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-11, standardize = FALSE, 
                                      lambda = lambdas, 
                                      family = "binomial")}, # thresh must be very low for glmnet to be accurate
    "admm[lasso]"   = {res2 <- admm.lasso(x, y, lambda = lambdas, 
                                          family = "binomial",
                                          intercept = TRUE, standardize = FALSE,
                                          irls.tol = 1e-6,
                                          abs.tol = 1e-6, rel.tol = 1e-6)},
    times = 5
)
```

```
## Unit: seconds
##           expr       min        lq      mean    median       uq      max
##  glmnet[lasso] 10.871893 10.972751 11.155293 11.182841 11.36326 11.38572
##    admm[lasso]  8.911481  9.594041  9.856075  9.669041 10.36360 10.74221
##  neval cld
##      5   b
##      5  a
```

```r
# difference of results (admm is actually quite a bit more precise than glmnet here)
max(abs(coef(res1) - res2$beta))
```

```
## [1] 6.657e-05
```

```r
mean(abs(coef(res1) - res2$beta))
```

```
## [1] 4.736351e-06
```

```r
# p > n
set.seed(123)
n <- 200
p <- 400
m <- 100
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- drop(x %*% b) + rnorm(n)

lambdas = glmnet(x, y)$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-14)},
    "admm[lasso]"   = {res2 <- admm.lasso(x, y, lambda = lambdas, 
                                          intercept = TRUE, standardize = TRUE,
                                          abs.tol = 1e-9, rel.tol = 1e-9)},
    "cd[lasso]"     = {res3 <- cd.lasso(x, y, lambda = lambdas, 
                                        intercept = TRUE, standardize = TRUE,
                                        tol = 1e-5)},
    times = 5
)
```

```
## Unit: milliseconds
##           expr       min        lq      mean    median        uq      max
##  glmnet[lasso]  862.9237  864.7782  920.0771  876.4073  878.6754 1117.601
##    admm[lasso] 4813.2545 4853.4155 5162.5523 5034.6162 5061.9813 6049.494
##      cd[lasso] 1748.0269 1748.8744 1830.9445 1771.3740 1801.5555 2084.891
##  neval cld
##      5 a  
##      5   c
##      5  b
```

```r
# difference of results
max(abs(coef(res1) - res2$beta))
```

```
## [1] 7.220397e-05
```

```r
max(abs(coef(res1) - res3$beta))
```

```
## [1] 7.273356e-05
```

```r
max(abs(res2$beta - res3$beta))
```

```
## [1] 1.897986e-05
```

```r
# p >> n
# ADMM is clearly not well-suited for this setting
set.seed(123)
n <- 100
p <- 1000
m <- 10
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- drop(x %*% b) + rnorm(n)

lambdas = glmnet(x, y)$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-12)},
    "admm[lasso]"   = {res2 <- admm.lasso(x, y, lambda = lambdas, 
                                          intercept = TRUE, standardize = TRUE,
                                          abs.tol = 1e-9, rel.tol = 1e-9)},
    times = 5
)
```

```
## Unit: milliseconds
##           expr        min         lq       mean     median        uq
##  glmnet[lasso]   76.10945   80.20753   83.54355   82.51591   88.4191
##    admm[lasso] 4200.14108 4301.50653 4387.28765 4340.10322 4416.9244
##         max neval cld
##    90.46577     5  a 
##  4677.76305     5   b
```

```r
# difference of results
max(abs(coef(res1) - res2$beta))
```

```
## [1] 0.0001833642
```

```r
mean(abs(coef(res1) - res2$beta))
```

```
## [1] 4.638761e-07
```
