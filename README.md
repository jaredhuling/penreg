





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
##           expr       min        lq      mean    median        uq       max
##  glmnet[lasso]  899.7733  908.0236  921.3106  908.0504  909.6120  981.0936
##    admm[lasso]  675.5999  681.5730  699.6475  700.4277  705.3199  735.3170
##      cd[lasso] 1644.9993 1651.2834 1710.7607 1689.4881 1723.4230 1844.6095
##  neval cld
##      5  b 
##      5 a  
##      5   c
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
##           expr      min       lq     mean   median       uq       max
##  glmnet[lasso] 9.153600 9.218419 9.704175 9.243971 9.768545 11.136340
##    admm[lasso] 7.802485 7.938068 8.165284 8.202543 8.433374  8.449947
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
##           expr       min        lq      mean    median        uq       max
##  glmnet[lasso]  785.5918  793.0325  805.3263  802.1128  810.0524  835.8421
##    admm[lasso] 4497.1962 4570.2749 4613.4047 4582.3039 4637.2231 4780.0253
##      cd[lasso] 1486.8709 1504.2903 1566.8702 1527.9714 1590.3347 1724.8839
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

lambdas = glmnet(x, y, standardize = FALSE)$lambda

# the glmnet threshold criterion had to be made extremely
# small for it not to have some coefficents which were poorly
# converged
microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-16, lambda = lambdas,
                                      standardize = FALSE)},
    "admm[lasso]"   = {res2 <- admm.lasso(x, y, lambda = lambdas, 
                                          intercept = TRUE, standardize = FALSE,
                                          abs.tol = 1e-8, rel.tol = 1e-8)},
    "cd[lasso]"     = {res3 <- cd.lasso(x, y, lambda = lambdas, 
                                        intercept = TRUE, standardize = FALSE,
                                        tol = 1e-5)},
    times = 5
)
```

```
## Unit: milliseconds
##           expr       min        lq      mean    median        uq       max
##  glmnet[lasso]  210.5572  220.2147  222.9123  221.6325  228.9390  233.2179
##    admm[lasso] 2418.7377 2451.2284 2473.5513 2451.3488 2506.8122 2539.6293
##      cd[lasso]  768.6828  778.6817  785.3402  783.1453  787.8998  808.2915
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
## [1] 1.524573e-05
```

```r
mean(abs(coef(res1) - res2$beta))
```

```
## [1] 3.713966e-08
```

```r
max(abs(coef(res1) - res3$beta))
```

```
## [1] 3.587655e-06
```

```r
mean(abs(coef(res1) - res3$beta))
```

```
## [1] 6.717902e-09
```

```r
max(abs(res2$beta - res3$beta))
```

```
## [1] 1.279021e-05
```

```r
mean(abs(res2$beta - res3$beta))
```

```
## [1] 3.756596e-08
```
