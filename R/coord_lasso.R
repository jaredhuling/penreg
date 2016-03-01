

#' Fitting A Lasso Model Using the Coordinate Descent Algorithm
#' 
#' @description Estimation of a linear model with the lasso penalty. The function
#' \eqn{\beta} minimizes
#' \deqn{\frac{1}{2n}\Vert y-X\beta\Vert_2^2+\lambda\Vert\beta\Vert_1}{
#' 1/(2n) * ||y - X * \beta||_2^2 + \lambda * ||\beta||_1}
#' 
#' where \eqn{n} is the sample size and \eqn{\lambda} is a tuning
#' parameter that controls the sparsity of \eqn{\beta}.
#' 
#' @param x The design matrix
#' @param y The response vector
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
#' @param penalty.factor a vector with length equal to the number of columns in x to be multiplied by lambda. by default
#'                      it is a vector of 1s
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
#' @param tol convergence tolerance parameter.
#' @param rel.tol Relative tolerance parameter.
#' 
#' @references 
#' \url{http://stanford.edu/~boyd/admm.html}
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#' 
#' ## fit lasso model with 100 tuning parameter values
#' res <- cd.lasso(x, y)
#' 
#' 
#' @export
cd.lasso <- function(x, 
                     y, 
                     lambda           = numeric(0), 
                     penalty.factor,
                     nlambda          = 100L,
                     lambda.min.ratio = NULL,
                     family           = c("gaussian", "binomial"),
                     intercept        = FALSE,
                     standardize      = FALSE,
                     maxit            = 5000L,
                     tol              = 1e-7
)
{
    n <- nrow(x)
    p <- ncol(x)
    
    x = as.matrix(x)
    y = as.numeric(y)
    intercept = as.logical(intercept)
    standardize = as.logical(standardize)
    family <- match.arg(family)
    
    if (n != length(y)) {
        stop("number of rows in x not equal to length of y")
    }
    
    if (missing(penalty.factor)) {
        penalty.factor <- numeric(0)
    } else {
        if (length(penalty.factor) != p) {
            stop("penalty.factor must be same length as number of columns in x")
        }
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
    if(tol < 0)
    {
        stop("tol should be nonnegative")
    }
    
    maxit   <- as.integer(maxit)
    tol <- as.numeric(tol)
    
    if (family == "gaussian")
    {
        res <- .Call("coord_lasso", x, y, 
                     lambda,
                     penalty.factor,
                     nlambda, 
                     lambda.min.ratio,
                     standardize, intercept,
                     list(maxit = maxit,
                          tol   = tol),
                     PACKAGE = "penreg")
    } else if (family == "binomial")
    {
        stop("Binomial not implemented yet")
    }
    class(res) <- "cd.lasso"
    res
}
