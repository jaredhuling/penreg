
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
#' @param group.weights A vector of values representing multiplicative factors by which each group's penalty is to 
#' be multiplied. Often, this is a function (such as the square root) of the number of predictors in each group. 
#' The default is to use the square root of group size for the group selection methods.
#' @param rho ADMM parameter. must be a strictly positive value. By default, an appropriate value is automatically chosen
#' @param dynamic.rho TRUE/FALSE indicating whether or not the rho value should be updated throughout the course of the ADMM iterations
#' @param maxit integer. Maximum number of ADMM iterations. Default is 500.
#' @param abs.tol absolute convergence tolerance for ADMM iterations for the relative dual and primal residuals. 
#' Default is 10^{-4}, which is typically adequate.
#' @param rel.tol relative convergence tolerance for ADMM iterations for the relative dual and primal residuals. 
#' Default is 10^{-4}, which is typically adequate.
#' @param irls.maxit integer. Maximum number of IRLS iterations. Only used if family != "gaussian". Default is 100.
#' @param irls.tol convergence tolerance for IRLS iterations. Only used if family != "gaussian". Default is 10^{-5}.
#' @return An object with S3 class "oglasso.fit" 
#' @export
#' @examples
#' n.obs <- 1e4
#' n.vars <- 50
#' 
#' true.beta <- c(rep(0,2), 1, -1, rep(0, 8), 0.5, -0.5, 1, rep(0, 35))
#' 
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' 
#' groups <- c(list(c(1,2), c(2,3), c(3,4,5), 5:10, 6:12, 7:15), lapply(16:50, function(x) x))
#' 
#' fit <- admm.oglasso(x = x, y = y, group = groups)
admm.oglasso <- function(x, y, 
                    group, 
                    family           = c("gaussian", "binomial"), 
                    nlambda          = 100L, 
                    lambda           = NULL, 
                    lambda.min.ratio = NULL, 
                    group.weights    = NULL,
                    standardize      = TRUE, 
                    intercept        = TRUE, 
                    rho              = NULL,
                    dynamic.rho      = TRUE,
                    maxit            = 500L,
                    abs.tol          = 1e-5, 
                    rel.tol          = 1e-5, 
                    irls.tol         = 1e-5, 
                    irls.maxit       = 100L) 
{
    
    family <- match.arg(family)
    this.call = match.call()
    
    nvars <- ncol(x)
    nobs <- nrow(x)
    y <- drop(y)
    dimy <- dim(y)
    leny <- ifelse(is.null(dimy), length(y), dimy[1])
    stopifnot(leny == nobs)
    
    if(isTRUE(rho <= 0))
    {
        stop("rho should be positive")
    }
    
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
    
    if(is.null(lambda.min.ratio))
    {
        lambda.min.ratio <- ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
    }
    
    
    irls.tol    <- as.double(irls.tol)
    abs.tol     <- as.double(abs.tol)
    rel.tol     <- as.double(rel.tol)
    dynamic.rho <- as.logical(dynamic.rho)
    irls.maxit  <- as.integer(irls.maxit)
    rho         <- if(is.null(rho))  -1.0  else  as.numeric(rho)
    
    if (is.null(lambda)) {
        if (lambda.min.ratio >= 1 | lambda.min.ratio <= 0) {
            stop("lambda.min.ratio should be less than 1 and greater than 0")
        }
        lambda.min.ratio <- as.double(lambda.min.ratio)
        nlambda <- as.integer(nlambda)[1]
        lambda <- numeric(0L)
    } else {
        if (any(lambda < 0)) {
            stop("lambdas should be non-negative")
        }
        lambda <- as.double(rev(sort(lambda)))
        nlambda <- as.integer(length(lambda))
    }
    
    opts <- list(maxit       = maxit,
                 eps_abs     = abs.tol,
                 eps_rel     = rel.tol,
                 rho         = rho,
                 dynamic_rho = dynamic.rho,
                 irls_maxit  = irls.maxit,
                 irls_tol    = irls.tol)
    
    fit <- oglasso.fit(family, is.sparse, x, y, group,
                       nlambda, lambda, lambda.min.ratio,
                       alpha, gamma, group.weights, 
                       group.idx,
                       ngroups, 
                       standardize, intercept, opts)
    fit$call = this.call
    fit
}



oglasso.fit <- function(family, is.sparse, x, y, group,  
                        nlambda, lambda, lambda.min.ratio,
                        alpha, gamma, group.weights, 
                        group.idx,
                        ngroups,
                        standardize, intercept, opts) {
    
    if (is.sparse) {
        fit <- .Call("oglasso_fit_sparse", 
                     x_ = x,
                     y_ = y,
                     group_ = group,
                     family_ = family,
                     nlambda_ = nlambda,
                     lambda_ = lambda,
                     lambda_min_ratio_ = lambda.min.ratio,
                     group_weights_ = group.weights,
                     group_idx = group.idx,
                     ngroups_ = ngroups,
                     standardize_ = standardize,
                     intercept_ = intercept, 
                     opts_ = opts,
                     PACKAGE = "penreg")
    } else {
        fit <- .Call("admm_oglasso_dense", 
                     x_ = x,
                     y_ = y,
                     group_ = group,
                     family_ = family,
                     nlambda_ = nlambda,
                     lambda_ = lambda,
                     lambda_min_ratio_ = lambda.min.ratio,
                     group_weights_ = group.weights,
                     group_idx = group.idx,
                     ngroups_ = ngroups,
                     standardize_ = standardize,
                     intercept_ = intercept, 
                     opts_ = opts,
                     PACKAGE = "penreg")
    }
    class(fit) <- c(class(fit), "oglasso")
    fit
}

