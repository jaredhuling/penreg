
#-------------------------------------------------------------------------------
# Estimation of the overlapping group lasso for
# least squares and generalized linear models
# 
# August 2015
# Jared Huling
#-------------------------------------------------------------------------------



#' Overlapping Group Lasso (oglasso)
#'
#' @param x input matrix or SparseMatrix of dimension nobs \times nvars. Each row is an observation, 
#' each column corresponds to a covariate
#' @param y numeric response vector of length nobs
#' @param group A list of length equal to the number of groups containing vectors of integers 
#' indicating the variable IDs for each group. For example, group=list(c(1,2), c(2,3), c(3,4,5)) specifies
#' that Group 1 contains variables 1 and 2, Group 2 contains variables 2 and 3, and Group 3 contains 
#' variables 3, 4, and 5. Can also be a matrix of 0s and 1s with the number of columns equal to the 
#' number of groups and the number of rows equal to the number of variables. A value of 1 in row i and 
#' column j indicates that variable i is in group j and 0 indicates that variable i is not in group j.
#' @param penalty Specification of penalty type. Choices include "gr.lasso", 
#' "gr.lasso.infty", "gr.mcp", and "gr.scad"
#' @param family "gaussian" for least squares problems, "binomial" for binary response
#' @param nlambda The number of lambda values. Default is 100.
#' @param lambda A user-specified sequence of lambda values. Left unspecified, the a sequence of lambda values is 
#' automatically computed, ranging uniformly on the log scale over the relevant range of lambda values.
#' @param lambda.min.ratio Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
#' value (i.e. the smallest value for which all coefﬁcients are zero). The default
#' depends on the sample size nobs relative to the number of variables nvars. If
#' nobs > nvars, the default is 0.0001, close to zero. If nobs < nvars, the default
#' is 0.01. A very small value of lambda.min.ratio will lead to a saturated ﬁt in
#' the nobs < nvars case.
#' @param alpha alpha value for ridge penalty. Defaults to 1 for no ridge penalty.
#' @param gamma Tuning parameter of the MCP penalty; defaults to 3.
#' @param group.weights A vector of values representing multiplicative factors by which each group's penalty is to 
#' be multiplied. Often, this is a function (such as the square root) of the number of predictors in each group. 
#' The default is to use the square root of group size for the group selection methods.
#' @param method Algorithm type for overlapping group lasso. Options include "admm" for the Alternating Direction
#' Method of Multipliers, "fista" for the Fast Iterative Shrinkage-Thresholding Algorithm. Defaults to "admm". 
#' "fista" is recommended when nvars >> nobs. More methods to come.
#' @param irls.tol convergence tolerance for IRLS iterations. Only used if family != "gaussian". Default is 10^{-7}.
#' @param eps convergence tolerance for ADMM/FISTA iterations for the relative dual and primal residuals. 
#' Default is 10^{-4}, which is typically adequate.
#' @param inner.tol convergence tolerance for inner iterations. Does not apply for method = "admm". Default is 10^{-3}.
#' @param irls.maxit integer. Maximum number of IRLS iterations. Only used if family != "gaussian". Default is 100.
#' @param outer.maxit integer. Maximum number of outer (ADMM or FISTA) iterations. Default is 500.
#' @param inner.maxit integer. Maximum number of inner iterations. Does not apply for method = "admm". Default is 250.
#' @return An object with S3 class "oglasso.fit" 
#' @export
#' @examples
#' n.obs <- 1e5
#' n.vars <- 150
#' 
#' true.beta <- rnorm(n.vars)
#' 
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' 
#' fit <- oglasso(x = x, y = y, group=list(c(1,2), c(2,3), c(3,4,5)), penalty = "gr.lasso")
oglasso <- function(x, y, group, 
                    family = c("gaussian", "binomial"), 
                    nlambda = 100L, lambda = NULL, lambda.min.ratio = 1e-4, 
                    group.weights = NULL,
                    standardize = TRUE, intercept = TRUE, dynamic.rho = TRUE,
                    irls.tol = 1e-7, eps = 1e-4, inner.tol = 1e-3, 
                    irls.maxit = 100L, outer.maxit = 500L, inner.maxit = 250L) {
    
    family <- match.arg(family)
    this.call = match.call()
    
    nvars <- ncol(x)
    nobs <- nrow(x)
    y <- drop(y)
    dimy <- dim(y)
    leny <- ifelse(is.null(dimy), length(y), dimy[1])
    stopifnot(leny == nobs)
    
    if (missing(group)) {
        stop("Must specify group structure.")
    } else if (is.matrix(group) | inherits(group, "Matrix")) {
        if (nvars != ncol(group)) {
            stop("group matrix must have same number of rows as variables. i.e. ncol(x) = nrow(group).")
        }
        group <- as(group, "CsparseMatrix")
        group <- as(group, "dgCMatrix")
    } else if (is.list(group) & !is.data.frame(group)) {
        tmp.vc <- numeric(nvars)
        group <- as(vapply(group, function(x) {tmp.vc[x] <- 1; return(tmp.vc) }, tmp.vc), "CsparseMatrix")
        group <- as(group, "dgCMatrix")
    } else {
        stop("Supply either a list or matrix. No data frames allowed.")
    }
    # Seek for variables which were not
    # included in any group and add them 
    # to a final group which will be 
    # unpenalized.
    rSg <- Matrix::rowSums(group) == 0
    addZeroGroup <- any(rSg)
    if (addZeroGroup) {
        group <- cBind(group, (1 * rSg))
    }
    group.idx <- as.integer(c(0, cumsum(Matrix::colSums(group))))
    ngroups <- ncol(group)
    if (!is.null(group.weights)) {
        stopifnot(length(group.weights) == ngroups)
        group.weights <- as.double(group.weights)
    } else {
        group.weights <- sqrt(Matrix::colSums(group))
    }
    if (addZeroGroup) {
        # Do not penalize variables in last group
        # which represents the group of variables
        # that were not put into any groups.
        group.weights[ngroups] <- 0
    }
    
    varnames <- colnames(x)
    if (is.null(varnames)) {
        varnames <- paste("V", seq(nvars), sep = "")
    }
    is.sparse = FALSE
    if (inherits(x, "sparseMatrix")) {
        is.sparse = TRUE
        x <- as(x, "CsparseMatrix")
        x <- as(x, "dgCMatrix")
    }
    alpha <- as.double(alpha)
    gamma <- as.double(gamma)
    irls.tol <- as.double(irls.tol)
    eps <- as.double(eps)
    inner.tol <- as.double(inner.tol)
    irls.maxit <- as.integer(irls.maxit)
    outer.maxit <- as.integer(outer.maxit)
    inner.maxit <- as.integer(inner.maxit)
    
    if (is.null(lambda)) {
        if (lambda.min.ratio >= 1) {
            stop("lambda.min.ratio should be less than 1")
        }
        lambda.min.ratio <- as.double(lambda.min.ratio)
        compute.lambda <- TRUE
        nlambda <- as.integer(nlambda)
        lambda <- numeric(1L)
    } else {
        if (any(lambda < 0)) {
            stop("lambdas should be non-negative")
        }
        compute.lambda <- FALSE
        lambda <- as.double(rev(sort(lambda)))
        nlambda <- as.integer(length(lambda))
    }
    
    fit <- oglasso.fit(family, is.sparse, x, y, group, penalty,
                       nlambda, lambda, lambda.min.ratio,
                       alpha, gamma, group.weights, 
                       group.idx, method, 
                       irls.tol, eps, inner.tol, irls.maxit, 
                       outer.maxit, inner.maxit, nvars, nobs, 
                       ngroups, compute.lambda,
                       standardize, intercept, as.logical(dynamic.rho))
    fit$call = this.call
    fit
}



oglasso.fit <- function(family, is.sparse, x, y, group, penalty, 
                        nlambda, lambda, lambda.min.ratio,
                        alpha, gamma, group.weights, 
                        group.idx, method, 
                        irls.tol, eps, inner.tol, irls.maxit, 
                        outer.maxit, inner.maxit, nvars, nobs, 
                        ngroups, compute.lambda, 
                        standardize, intercept, dynamic.rho) {
    
    if (is.sparse) {
        fit <- .Call("oglasso_fit_sparse", 
                     x_ = x,
                     y_ = y,
                     group_ = group,
                     family_ = family,
                     penalty_ = penalty,
                     nlambda_ = nlambda,
                     lambda_ = lambda,
                     lambda_min_ratio_ = lambda.min.ratio,
                     alpha_ = alpha,
                     gamma_ = gamma,
                     group_weights_ = group.weights,
                     group_idx = group.idx,
                     method_ = method,
                     irls_tol_ = irls.tol,
                     eps_ = eps,
                     inner_tol_ = inner.tol,
                     irls_maxit_ = irls.maxit,
                     outer_maxit_ = outer.maxit,
                     inner_maxit_ = inner.maxit,
                     nvars_ = nvars,
                     nobs_ = nobs,
                     ngroups_ = ngroups,
                     compute_lambda_ = compute.lambda,
                     intercept_ = intercept, 
                     dynamic_rho_ = dynamic.rho,
                     PACKAGE = "oglasso")
    } else {
        fit <- .Call("admm_oglasso_dense", 
                     x_ = x,
                     y_ = y,
                     group_ = group,
                     family_ = family,
                     penalty_ = penalty,
                     nlambda_ = nlambda,
                     lambda_ = lambda,
                     lambda_min_ratio_ = lambda.min.ratio,
                     alpha_ = alpha,
                     gamma_ = gamma,
                     group_weights_ = group.weights,
                     group_idx = group.idx,
                     method_ = method,
                     irls_tol_ = irls.tol,
                     eps_ = eps,
                     inner_tol_ = inner.tol,
                     irls_maxit_ = irls.maxit,
                     outer_maxit_ = outer.maxit,
                     inner_maxit_ = inner.maxit,
                     nvars_ = nvars,
                     nobs_ = nobs,
                     ngroups_ = ngroups,
                     compute_lambda_ = compute.lambda,
                     standardize_ = standardize,
                     intercept_ = intercept,
                     dynamic_rho_ = dynamic.rho,
                     PACKAGE = "oglasso")
    }
    class(fit) <- c(class(fit), "oglasso")
    fit
}

