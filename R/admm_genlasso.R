
#' Fitting A Generalized Lasso Model Using ADMM Algorithm
#' 
#' @description Estimation of a linear model with the lasso penalty. The function
#' \eqn{\beta} minimizes
#' \deqn{\frac{1}{2n}\Vert y-X\beta\Vert_2^2+\lambda\Vert\beta\Vert_1}{
#' 1/(2n) * ||y - X * \beta||_2^2 + \lambda * ||D\beta||_1}
#' 
#' where \eqn{n} is the sample size and \eqn{\lambda} is a tuning
#' parameter that controls the sparseness of \eqn{\beta}.
#' 
#' @param x The design matrix
#' @param y The response vector
#' @param D The specified penalty matrix 
#' @param intercept Whether to fit an intercept in the model. Default is \code{FALSE}. 
#' @param standardize Whether to standardize the design matrix before
#'                    fitting the model. Default is \code{FALSE}. Fitted coefficients
#'                    are always returned on the original scale.
#' @param lambda A user provided sequence of \eqn{\lambda}. If set to
#'                      \code{NULL}, the program will calculate its own sequence
#'                      according to \code{nlambda} and \code{lambda_min_ratio},
#'                      which starts from \eqn{\lambda_0} (with this
#'                      \eqn{\lambda} all coefficients will be zero) and ends at
#'                      \code{lambda0 * lambda_min_ratio}, containing
#'                      \code{nlambda} values equally spaced in the log scale.
#'                      It is recommended to set this parameter to be \code{NULL}
#'                      (the default).
#' @param nlambda Number of values in the \eqn{\lambda} sequence. Only used
#'                       when the program calculates its own \eqn{\lambda}
#'                       (by setting \code{lambda = NULL}).
#' @param lambda_min_ratio Smallest value in the \eqn{\lambda} sequence
#'                                as a fraction of \eqn{\lambda_0}. See
#'                                the explanation of the \code{lambda}
#'                                argument. This parameter is only used when
#'                                the program calculates its own \eqn{\lambda}
#'                                (by setting \code{lambda = NULL}). The default
#'                                value is the same as \pkg{glmnet}: 0.0001 if
#'                                \code{nrow(x) >= ncol(x)} and 0.01 otherwise.
#' @param maxit Maximum number of admm iterations.
#' @param abs.tol Absolute tolerance parameter.
#' @param rel.tol Relative tolerance parameter.
#' @param rho ADMM step size parameter. If set to \code{NULL}, the program
#'                   will compute a default one which has good convergence properties.
#' @references 
#' \url{https://projecteuclid.org/euclid.aos/1304514656}
#' 
#' \url{http://stanford.edu/~boyd/admm.html}
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#' 
#' D <- c(1, -1, rep(0, p - 2))
#' for (i in 1:20) {D <- rbind(D, c(rep(0, 2 * i), 1, -1, rep(0, p - 2 - 2 * i)))}
#' D <- rbind(D, diag(p))
#' 
#' ## fit lasso model with 100 tuning parameter values
#' res <- admm.genlasso(x, y, D = D)
#' 
#' 
#' @export
admm.genlasso <- function(x, 
                          y, 
                          D                = NULL,
                          lambda           = numeric(0), 
                          nlambda          = 100L,
                          lambda.min.ratio = NULL,
                          intercept        = FALSE,
                          standardize      = FALSE,
                          maxit            = 5000L,
                          abs.tol          = 1e-7,
                          rel.tol          = 1e-7,
                          rho              = NULL
                          )
{
    n <- nrow(x)
    p <- ncol(x)
    
    if (is.null(lambda.min.ratio)) 
    {
        ifelse(n < p, 0.01, 0.0001)
    }
    
    if (is.null(D)) {
        warning("D is missing, defaulting to regular lasso")
        D <- as(diag(p), "sparseMatrix")
    } else {
        D <- as(D, "sparseMatrix")
    }
    
    x = as.matrix(x)
    y = as.numeric(y)
    intercept = as.logical(intercept)
    standardize = as.logical(standardize)
    
    if (n != length(y)) {
        stop("number of rows in x not equal to length of y")
    }
    
    lambda_val = sort(as.numeric(lambda), decreasing = TRUE)
    
    if(any(lambda_val <= 0)) 
    {
        stop("lambda must be positive")
    }
    
    if(nlambda[1] <= 0) 
    {
        stop("nlambda must be a positive integer")
    }
    
    if(is.null(lambda.min.ratio))
    {
        lmr_val <- ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
    } else 
    {
        lmr_val <- as.numeric(lambda.min.ratio)
    }
    
    if(lmr_val >= 1 | lmr_val <= 0) 
    {
        stop("lambda.min.ratio must be within (0, 1)")
    }
    
    lambda           <- lambda_val
    nlambda          <- as.integer(nlambda[1])
    lambda.min.ratio <- lmr_val
    
    
    if(maxit <= 0)
    {
        stop("maxit should be positive")
    }
    if(abs.tol < 0 | rel.tol < 0)
    {
        stop("abs.tol and rel.tol should be nonnegative")
    }
    if(isTRUE(rho <= 0))
    {
        stop("rho should be positive")
    }
    
    maxit   <- as.integer(maxit)
    abs.tol <- as.numeric(abs.tol)
    rel.tol <- as.numeric(rel.tol)
    rho     <- if(is.null(rho))  -1.0  else  as.numeric(rho)
    
    res <- .Call("admm_genlasso", 
                 x, y, D, 
                 lambda,
                 nlambda, lambda.min.ratio,
                 standardize, intercept,
                 list(maxit   = maxit,
                      eps_abs = abs.tol,
                      eps_rel = rel.tol,
                      rho     = rho),
                 PACKAGE = "penreg")
    res
}

