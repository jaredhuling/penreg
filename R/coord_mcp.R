

#' Fitting An MCP Model Using the Coordinate Descent Algorithm
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
#' @param gamma Sparsity parameter vector
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
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#' 
#' ## fit lasso model with 100 tuning parameter values
#' res <- cd.mcp(x, y, gamma = 4)
#' 
#' 
#' @export
cd.mcp <- function(x, 
                   y, 
                   lambda           = numeric(0), 
                   gamma            = 4,
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
        res <- .Call("coord_mcp", x, y, 
                     lambda,
                     gamma,
                     penalty.factor,
                     nlambda, 
                     lambda.min.ratio,
                     standardize, intercept,
                     list(maxit = maxit,
                          tol   = tol),
                     PACKAGE = "penreg")
        lambda <- res$lambda
        gamma  <- res$gamma
        names(res$coefficients) <- paste0("g", 1:length(gamma))
        parms <- array(NA, dim = c(2, length(gamma), length(lambda)))
        parms[1,,] <- matrix(rep(gamma, length(lambda)), nrow = 2, byrow = FALSE)
        parms[2,,] <- matrix(rep(lambda, length(gamma)), nrow = 2, byrow = TRUE)
        dimnames(parms) <- list(c("gamma", "lambda"),
                                paste0("g", 1:length(gamma)),
                                paste0("l", 1:length(lambda)))
        res$parms  <- parms
    } else if (family == "binomial")
    {
        stop("Binomial not implemented yet")
    }
    class(res) <- "cd.mcp"
    res
}

## Modified from Rahul Mazumder, Trevor Hastie, and Jerome Friedman's sparsenet package
## https://cran.r-project.org/web/packages/sparsenet/index.html
## this function is an internal prediction function for MCP code
mcppredict = function(object,newx,s=NULL, type = c("response", "coefficients", "nonzero"), exact=FALSE, ...){
    a0=t(as.matrix(object$intercept))
    gamma=object$gamma
    gammarize=function(x,gamma){
        attr(x,"gamma")=gamma
        x
    }
    rownames(a0) = "(Intercept)"
    nbeta = rbind2(a0, object$beta)
    
    if(!is.null(s)){
        vnames          = dimnames(nbeta)[[1]]
        dimnames(nbeta) = list(NULL,NULL)
        lambda          = object$lambda
        lamlist         = lambda.interp(lambda,s)
        nbeta           = nbeta[,lamlist$left,drop=FALSE] * lamlist$frac + 
            nbeta[,lamlist$right,drop=FALSE] * (1-lamlist$frac)
        dimnames(nbeta) = list(vnames, paste(seq(along=s)))
    }
    if(type=="coefficients")return(gammarize(nbeta, gamma))
    if(type=="nonzero")return(gammarize(nonzeroCoef(nbeta[-1,,drop=FALSE], bystep=TRUE), gamma))
    gammarize(as.matrix(cbind2(1,newx) %*% nbeta),gamma)
}

## Modified from Rahul Mazumder, Trevor Hastie, and Jerome Friedman's sparsenet package
## https://cran.r-project.org/web/packages/sparsenet/index.html
#' @title prediction function for coordinate descent MCP
#' @param object Fitted "cd.mcp" model object
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix. 
#'        This argument is not used for type=c("coefficients","nonzero")
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. 
#'         Default is the entire sequence used to create the model.
#' @param which.gamma Index or indices of gamma values at which predictions are to be made. 
#'        Default is all those used in the fit
#' @param type "response" returns fitted predictions at newx. Type "coefficients" computes the coefficients at 
#'        the requested values for s. Type "nonzero" returns lists of the indices of the nonzero coefficients for 
#'        each value of s.
#' @param exact By default (exact=FALSE) the predict function uses linear interpolation to make predictions for values 
#'        of s that do not coincide with those used in the fitting algorithm. Currently exact=TRUE is not implemented, 
#'        but prints an error message telling the user how to achieve the exact predictions. This is done my rerunning 
#'        the algorithm with the desired values interspersed (in order) with the values used in the original fit
#' @param ... Not used.
#'              
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#' 
#' ## fit lasso model with 100 tuning parameter values
#' res <- cd.mcp(x, y, gamma = 4)
#' 
#' coefs <- predict(res, type = "coef")
#' 
#' @export
predict.cd.mcp = function(object,newx,s=NULL,which.gamma=NULL,type=c("response","coefficients","nonzero"),exact=FALSE,...){
    type = match.arg(type)
    if(missing(newx)){
        if(!match(type,c("coefficients","nonzero"),FALSE))stop("You need to supply a value for 'newx'")
    }
    coeflist    = object$coefficients
    ngamma      = length(coeflist)
    coeflistseq = seq(along=coeflist)
    if(is.null(which.gamma))which.gamma = coeflistseq
    else   which.gamma = coeflistseq[match(which.gamma,coeflistseq,0)]
    if(length(which.gamma)>1){
        predlist        = as.list(which.gamma)
        names(predlist) = names(coeflist)[which.gamma]
        for(j in seq(along=which.gamma))predlist[[j]] = mcppredict(coeflist[[which.gamma[j]]], newx, s, type, ...)
        predlist
    }
    else mcppredict(coeflist[[which.gamma]], newx, s, type,...)
}

## Modified from Rahul Mazumder, Trevor Hastie, and Jerome Friedman's sparsenet package
## https://cran.r-project.org/web/packages/sparsenet/index.html
#' @title cross validation for coordinate descent MCP
#' @param x The design matrix
#' @param y The response vector
#' @param type.measure loss to use for cross-validation. can be one of "mse" for mean squared error or "mae"
#'        for mean absolute error 
#' @param ... Other arguments that can be passed to cd.mcp
#' @param nfolds number of folds for CV - default is 10. no smaller than 3
#' @param foldid an optional vector of values between 1 and nfold identifying whhat fold each observation is in. 
#' If supplied, nfold can be missing
#' @param trace.it If TRUE, then a printout that shows the progress is printed
#'              
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#' 
#' ## fit lasso model with 100 tuning parameter values
#' res <- cv.cd.mcp(x, y, gamma = 4)
#' 
#' 
#' @export
cv.cd.mcp = function(x, y, weights, 
                     lambda = numeric(0), 
                     gamma  = 4,
                     type.measure = c("mse", "mae"), ..., 
                     nfolds=10, foldid, trace.it = FALSE){
    this.call=match.call()
    type.measure=match.arg(type.measure)
    N=nrow(x)
    ngamma = length(gamma)
    if(missing(weights))weights=rep(1.0,N)else weights=as.double(weights)
    
    ###Fit the model once to get dimensions etc of output
    y             = drop(y) # we dont like matrix responses unless we need them
    cd.mcp.object = cd.mcp(x, y, lambda = lambda, gamma = gamma, ...)
    parms         = cd.mcp.object$parms
    lambda        = cd.mcp.object$lambda
    gamma         = cd.mcp.object$gamma
    nz            = if (ngamma > 1) sapply(predict(cd.mcp.object, type="nonzero"), function(x) sapply(x, length)) else
        matrix(sapply(predict(cd.mcp.object, type="nonzero"), length), ncol = 1)
    if (ngamma == 1) colnames(nz) <- "g1"
    dd            = dim(nz)
    ngamma        = dd[2]
    nlams         = matrix(0, nfolds, dd[2])
    predmat       = array(NA, c(N, dd))
    if(missing(foldid)) foldid=sample(rep(seq(nfolds),length=N)) else nfolds=max(foldid)
    if(nfolds<3)stop("nfolds must be bigger than 3; nfolds=10 recommended")
    outlist       = as.list(seq(nfolds))
    ###Now fit the nfold models and store them
    for(i in seq(nfolds)){
        which=foldid==i
        #fitobj = cd.mcp(x[!which,,drop=FALSE],y[!which],weights=weights[!which],parms=parms, ...)
        fitobj = cd.mcp(x[!which,,drop=FALSE], y[!which], 
                        #weights = weights[!which], #no weights yet
                        lambda = lambda,
                        gamma = gamma,
                        ...)
        preds  = predict(fitobj, x[which,,drop=FALSE])
        for(j in seq(ngamma)){
            nlami = length(fitobj$coef[[j]]$lambda)
            predmat[which,seq(nlami),j] = preds[[j]]
            nlams[i,j] = nlami
        }
        if(trace.it)cat(i)
    }
    
    
    N=length(y) - apply(is.na(predmat),c(2,3),sum)
    cvraw=switch(type.measure,
                 "mse"=(y-predmat)^2,
                 "mae"=abs(y-predmat)
    )
    
    cvm=apply(cvraw,c(2,3),weighted.mean,w=weights,na.rm=TRUE)
    for(j in seq(ngamma))cvraw[,,j]=scale(cvraw[,,j],cvm[,j],FALSE)
    cvsd=apply(cvraw^2,c(2,3),weighted.mean,w=weights,na.rm=TRUE)/(N-1)
    cvsd=sqrt(cvsd)
    obj = list(lambda     = t(parms[2,,]), 
               cvm        = cvm,
               cvsd       = cvsd,
               cvup       = cvm+cvsd,
               cvlo       = cvm-cvsd,
               nzero      = nz,
               name       = type.measure,
               cd.mcp.fit = cd.mcp.object,
               call       = this.call)
    whichmin      = argmin(cvm)
    obj$parms.min = parms[,whichmin[2],whichmin[1]]
    obj$which.min = whichmin
    which         = cvm< min(cvm)+cvsd
    nz[!which]    = 1e40
    whichcv       = argmin(nz)
    obj$parms.1se = parms[, whichcv[2], whichcv[1]]
    obj$which.1se = whichcv
    class(obj)    = "cv.cd.mcp"
    obj
}

## Modified from Rahul Mazumder, Trevor Hastie, and Jerome Friedman's sparsenet package
## https://cran.r-project.org/web/packages/sparsenet/index.html
#' @title prediction function for cv.cd.mcp
#' @param object Fitted "cv.cd.mcp" model object
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix. 
#'        This argument is not used for type=c("coefficients","nonzero")
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. 
#'         Default is the entire sequence used to create the model. Can also be one of
#'         c("parms.min", "parms.1se")
#' @param which.gamma Index or indices of gamma values at which predictions are to be made. 
#'        Default is all those used in the fit
#' @param type "response" returns fitted predictions at newx. Type "coefficients" computes the coefficients at 
#'        the requested values for s. Type "nonzero" returns lists of the indices of the nonzero coefficients for 
#'        each value of s.
#' @param exact By default (exact=FALSE) the predict function uses linear interpolation to make predictions for values 
#'        of s that do not coincide with those used in the fitting algorithm. Currently exact=TRUE is not implemented, 
#'        but prints an error message telling the user how to achieve the exact predictions. This is done my rerunning 
#'        the algorithm with the desired values interspersed (in order) with the values used in the original fit
#' @param ... Not used.
#'              
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#' 
#' ## fit lasso model with 100 tuning parameter values
#' res <- cd.mcp(x, y, gamma = 4)
#' 
#' coefs <- predict(res, type = "coef")
#' 
#' @export
predict.cv.cd.mcp = function(object, newx, which = c("parms.min", "parms.1se"), ...){
    which = match.arg(which)
    switch(which,
           parms.min = {lambda=object$parms.min[2]; which.gamma=object$which.min[2]},
           parms.1se = {lambda=object$parms.1se[2]; which.gamma=object$which.1se[2]}
    )
    predict(object$cd.mcp.fit, newx, s=lambda, which.gamma = which.gamma, ...)
}


