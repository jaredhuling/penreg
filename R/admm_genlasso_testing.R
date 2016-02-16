

admm.genlasso.R <- function(x, y, D, lambda, rho, abs.tol = 1e-5, rel.tol = 1e-5, maxit = 500L, gamma = 4) {
    library(Matrix)
    n <- length(y)
    p <- ncol(x)
    npen <- nrow(D)
    if (ncol(D) != p) {
        stop("D does not have correct number of columns")
    }
    xtx <- crossprod(x)
    DtD <- crossprod(D)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    
    #lambda <- lambda * n
    iters <- maxit
    
    alpha <- numeric(p)
    z <- u <- numeric(npen)
    A <- as(xtx + rho * DtD, "Matrix")
    
    for (i in 1:maxit) {
        q <- xty + rho * crossprod(D, (z - u))
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        Dalpha <- D %*% alpha
        z <- soft.thresh(Dalpha + u, lambda / rho)
        
        u <- u + (Dalpha - z)
        loss.history[i] <- l1.loss.leastsquares(x, y, alpha, z, lambda)
        
        r_norm = sqrt(sum( (Dalpha - z)^2 ))
        s_norm = sqrt(sum( (-rho * (z - z.prev))^2 ))
        eps_pri = sqrt(p) * abs.tol + rel.tol * max(sqrt(sum(Dalpha ^ 2)), sqrt(sum((-z)^2 ) ))
        eps_dual = sqrt(p) * abs.tol + rel.tol * sqrt(sum( (rho * u)^2 ))
        
        
        if (r_norm < eps_pri & s_norm < eps_dual) {
            iters <- i
            break
        }
    }
    
    list(beta=alpha, iters = iters, loss.hist = loss.history[!is.na(loss.history)], beta.aug = z)
}


soft.thresh <- function(a, kappa) {
    pmax(0, a - kappa) - pmax(0, -a - kappa)
}

l1.loss.leastsquares <- function(x, y, beta, z, lambda)
{
    0.5 * sum((y - x %*% beta) ^ 2) + lambda * sum(abs(z))
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


##
admm.genlasso.logistic.R <- function(x, y, D, lambda, rho, 
                                     abs.tol = 1e-5, rel.tol = 1e-5, 
                                     maxit = 500L, gamma = 4) {
    library(Matrix)
    #xtx <- crossprod(x)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    n <- length(y)
    p <- ncol(x)
    npen <- nrow(D)
    if (ncol(D) != p) {
        stop("D does not have correct number of columns")
    }
    #lambda <- lambda * n
    DtD <- crossprod(D)
    iters <- maxit
    
    alpha <- numeric(p)
    z <- u <- numeric(npen)
    
    for (i in 1:maxit) {
        
        # update alpha
        alpha <- logistic.gen.x.update(x, xty, u, z, rho, D, DtD, alpha.0 = alpha)
        z.prev <- z
        
        
        # update z
        Dalpha <- D %*% alpha
        z <- soft.thresh(Dalpha + u, lambda / rho)
        
        # update lagrangian parameter
        u <- u + (Dalpha - z)
        
        loss.history[i] <- l1.loss.logistic(x, y, alpha, z, lambda)
        
        r_norm   = sqrt(sum( (Dalpha - z)^2 ))
        s_norm   = sqrt(sum( (-rho * (z - z.prev))^2 ))
        eps_pri  = sqrt(p) * abs.tol + rel.tol * max(sqrt(sum(Dalpha ^ 2)), sqrt(sum((-z)^2 ) ))
        eps_dual = sqrt(p) * abs.tol + rel.tol * sqrt(sum( (rho * u)^2 ))
        
        if(r_norm / eps_pri > 10 * s_norm / eps_dual)
        {
            rho <- rho * 2
            u <- u * 0.5
        } else if(s_norm / eps_dual > 10 * r_norm / eps_pri)
        {
            rho <- rho / 2
            u <- u * 2
        }
        
        
        if (r_norm < eps_pri & s_norm < eps_dual) {
            iters <- i
            break
        }
    }
    
    list(beta=alpha, iters = iters, loss.hist = loss.history[!is.na(loss.history)], beta.aug = z)
}


# admm x update step for logistic regression
logistic.gen.x.update <- function(x, xty, u, z, rho, D, DtD, alpha.0 = NULL)
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
        grad <- drop(   crossprod(x, prob) - xty + rho * crossprod(D, D %*% alpha.cur  -z + u)   )
        W    <- prob * (1 - prob)
        HH   <- crossprod(x * sqrt(W), x) + rho * DtD
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


