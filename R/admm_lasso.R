

admm.lasso <- function(x, 
                       y, 
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
    
    res <- .Call("admm_lasso", x, y, lambda,
                 nlambda, lambda.min.ratio,
                 standardize, intercept,
                 list(maxit   = maxit,
                      eps_abs = abs.tol,
                      eps_rel = rel.tol,
                      rho     = rho),
                 PACKAGE = "penreg")
    res
}

