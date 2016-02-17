


admm.sparse.genridge.R <- function(x, y, lambda, lambda.ridge, D,
                                   rho = NULL, 
                                   abs.tol = 1e-5, rel.tol = 1e-5, 
                                   maxit = 500L, gamma = 4) {
    require(Matrix)
    require(rARPACK)
    #xtx <- crossprod(x)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    n <- length(y)
    p <- ncol(x)
    nridge <- nrow(D)
    stopifnot(p == ncol(D))
    #lambda <- lambda * n
    iters <- maxit
    
    if (length(lambda) > 1) {
        warning("only works for one lambda value 
                at a time now; using first lambda value")
        lambda <- lambda[1]
    }
    if (length(lambda.ridge) > 1) {
        warning("only works for one lambda value 
                at a time now; using first lambda value")
        lambda.ridge <- lambda.ridge[1]
    }
    xtxDtD <- crossprod(x) + lambda.ridge * crossprod(D)
    
    
    ## if rho value is not supplied, 
    ## compute one that is good
    if (is.null(rho)) {
        eigs <- eigs_sym(xtxDtD, k = 2, 
                         which = "BE", 
                         opts = list(maxitr = 500, 
                                     tol = 1e-4))$values
        rho <- eigs[1] ^ (1 / 3) * lambda ^ (2 / 3)
        
    }
    
    
    alpha <- z <- u <- numeric(p)
    A <- as(xtxDtD + rho * diag(p), "Matrix")
    
    for (i in 1:maxit) {
        q <- xty + rho * (z - u)
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        z <- soft.thresh(alpha + u, lambda / rho)
        
        u <- u + (alpha - z)
        loss.history[i] <- genridge.l1.loss.leastsquares(x, y, alpha, z, lambda, lambda.ridge, D)
        
        r_norm = sqrt(sum( (alpha - z)^2 ))
        s_norm = sqrt(sum( (-rho * (z - z.prev))^2 ))
        eps_pri = sqrt(p) * abs.tol + rel.tol * max(sqrt(sum(alpha ^ 2)), sqrt(sum((-z)^2 ) ))
        eps_dual = sqrt(p) * abs.tol + rel.tol * sqrt(sum( (rho * u)^2 ))
        
        
        if (r_norm < eps_pri & s_norm < eps_dual) {
            iters <- i
            break
        }
    }
    
    list(beta=z, iters = iters, loss.hist = loss.history[!is.na(loss.history)])
}


soft.thresh <- function(a, kappa) {
    pmax(0, a - kappa) - pmax(0, -a - kappa)
}

genridge.l1.loss.leastsquares <- function(x, y, beta, z, lambda, lambda.ridge, D)
{
    0.5 * sum((y - x %*% beta) ^ 2) + lambda * sum(abs(z)) + lambda.ridge * sum((D %*% beta) ^ 2)
}

l1.loss.logistic <- function(x, y, beta, z, lambda)
{
    
    sum(log( 1 + exp(x %*% beta))) - sum((y * x) %*% beta) + lambda * sum(abs(z))
}

l1.loss.logistic2 <- function(x, y, beta, z, lambda)
{
    prob <- drop(1 / (1 + exp( -1 * x %*% beta)))
    prob <- pmax(prob, 1e-5)
    prob <- pmin(prob, 1 - 1e-5)
    sum( y * log(prob) ) + sum( (1 - y) * log(1 - prob) ) - lambda * sum(abs(z))
}



admm.lasso.logistic.R <- function(x, y, lambda, rho, abs.tol = 1e-5, rel.tol = 1e-5, maxit = 500L, gamma = 4) {
    require(Matrix)
    require(rARPACK)
    #xtx <- crossprod(x)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    n <- length(y)
    p <- ncol(x)
    #lambda <- lambda * n
    iters <- maxit
    
    alpha <- z <- u <- numeric(p)
    
    for (i in 1:maxit) {
        
        # update alpha
        alpha <- logistic.x.update(x, xty, u, z, rho, alpha.0 = alpha)
        z.prev <- z
        
        # update z
        z <- soft.thresh(alpha + u, lambda / rho)
        
        # update lagrangian parameter
        u <- u + (alpha - z)
        
        loss.history[i] <- l1.loss.logistic(x, y, alpha, z, lambda)
        
        r_norm   = sqrt(sum( (alpha - z)^2 ))
        s_norm   = sqrt(sum( (-rho * (z - z.prev))^2 ))
        eps_pri  = sqrt(p) * abs.tol + rel.tol * max(sqrt(sum(alpha ^ 2)), sqrt(sum((-z)^2 ) ))
        eps_dual = sqrt(p) * abs.tol + rel.tol * sqrt(sum( (rho * u)^2 ))
        
        if(r_norm  > 10 * s_norm )
        {
            rho <- rho * (2)
            u <- u * 0.5
        } else if(s_norm  > 10 * r_norm )
        {
            rho <- rho * 0.5
            u <- u * 2
        }
        
        
        
        if (r_norm < eps_pri & s_norm < eps_dual) {
            iters <- i
            break
        }
    }
    
    list(beta=z, iters = iters, loss.hist = loss.history[!is.na(loss.history)])
}


# admm x update step for logistic regression
logistic.x.update <- function(x, xty, u, z, rho, alpha.0 = NULL)
{
    nvars <- ncol(x)
    if (is.null(alpha.0))
    {
        alpha.cur <- rep(0, nvars)
    } else 
    {
        alpha.cur <- alpha.0
    }
    niter <- 25
    tol <- 1e-7
    for (i in 1:niter)
    {
        prob <- drop(1 / (1 + exp( -x %*% alpha.cur)))
        grad <- drop(crossprod(x, prob) - xty + rho * (alpha.cur - z + u))
        W    <- prob * (1 - prob)
        HH   <- crossprod(x * sqrt(W), x) + rho * diag(nvars)
        dx   <- -1 * solve(HH, grad)
        dfx  <- crossprod(grad, dx)
        if (abs(dfx) < tol)
        {
            break
        }
        alpha.cur <- alpha.cur + 1 * dx
        
    }
    alpha.cur
}

