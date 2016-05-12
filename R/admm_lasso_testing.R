
## admm for the lasso with matrix equilibriation-based constraint preconditioning
admm.lasso.prec.R <- function(x, y, lambda, rho = NULL, 
                              abs.tol = 1e-5, rel.tol = 1e-5, 
                              maxit = 500L, gamma = 4, stochastic = TRUE) {
    require(Matrix)
    require(rARPACK)
    xtx <- crossprod(x)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    n <- length(y)
    p <- ncol(x)
    #lambda <- lambda * n
    iters <- maxit
    
    if (length(lambda) > 1) {
        warning("only works for one lambda value 
                at a time now; using first lambda value")
        lambda <- lambda[1]
    }
    
    
    ##  this scaler works okay
    #scaling <- sqrt(apply(xtx, 2, function(xx) sqrt(max( abs(xx) ))))
    
    if (stochastic)
    {
        ## often better than sbin,
        ## but it is stochastic
        sc <- ssbin(xtx, maxit = 500)
    } else 
    {
        ## good, but not usually as
        ## good as ssbin, but will
        ## give same scaler each time (deterministic)
        sc <- sbin(xtx, maxit = 100)
    }
    
    
    ###works
    scaling <- 1/sqrt(sc) #1/sqrt(sc)
    scaling <- scaling / max(scaling)
    
    #B <- (diag(1/(sc) )) %*% xtx %*% diag(1/(sc))
    
    #WORKS
    ###B <- (diag((scaling) )) %*% xtx %*% diag((scaling))
    
    B <- (diag((1/scaling) )) %*% xtx %*% diag((1/scaling))
    
    ## if rho value is not supplied, 
    ## compute one that is good
    if (is.null(rho)) {
        eigs <- eigs_sym(B, k = 2, 
                         which = "BE", 
                         opts = list(maxitr = 500, 
                                     tol = 1e-4))$values
        #rho <- eigs[1] ^ (1 / 3) * lambda ^ (2 / 3)
        #eigs <- eigen(B)$values
        rho <- sqrt(eigs[1] * (eigs[length(eigs)] + 1e-1) )
    }
    
    ## scale by inverse of scaling factor 
    #scaling <- 1/(drop(equ$d1)) 
    
    #scaling <- rep(2.5, length(scaling))
    
    alpha <- z <- u <- numeric(p)
    A <- as(xtx + rho * diag(scaling ^ 2), "Matrix")
    
    for (i in 1:maxit) {
        q <- xty + rho * scaling ^ 2 * z - u * scaling
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        z <- soft.thresh(alpha + u / (rho * scaling), lambda / (rho * scaling ^ 2) )
        
        u <- u + rho * (scaling * drop(alpha - z))
        loss.history[i] <- l1.loss.leastsquares(x, y, alpha, z, lambda)
        
        r_norm = sqrt(sum( (scaling * (alpha - z) )^2 ))
        s_norm = sqrt(sum( (-rho * scaling * (z - z.prev))^2 ))
        eps_pri = sqrt(p) * abs.tol + rel.tol * max(sqrt(sum( (scaling * alpha) ^ 2)), sqrt(sum((-scaling * z)^2 ) ))
        eps_dual = sqrt(p) * abs.tol + rel.tol * sqrt(sum( (rho * scaling^2 * u)^2 ))
        
        
        if (r_norm < eps_pri & s_norm < eps_dual) {
            iters <- i
            break
        }
    }
    
    list(beta=z, beta.aug = alpha, iters = iters, loss.hist = loss.history[!is.na(loss.history)],
         scaling = scaling)
}

ssbin <- function(A, eps = 1e-2, maxit = 100)
{
    
    # ALGORITHMS FOR THE EQUILIBRATION OF MATRICES
    # AND THEIR APPLICATION TO
    # LIMITED-MEMORY QUASI-NEWTON METHODS
    # thesis of Andrew Michael Bradley, 2010
    
    n <- nrow(A)
    p <- ncol(A)
    d <- 1/sqrt(sqrt(apply(A, 2, function(xx) sqrt(max( abs(xx) )))))
    d <- d / min(d)
    
    #d <- rep(1, n)
    for (i in 1:maxit)
    {
        u <- runif(n)
        s <- u / sqrt(d)
        y <- A %*% s
        omega <- 2 ^ (-max(min(floor(log2(i)) - 1, 4), 1))
        d <- (1 - omega) * d / sum(d) + omega * y ^ 2 / sum(y ^ 2)
    }
    drop(1/sqrt(d))
}

sbin <- function(A, eps = 1e-2, maxit = 100)
{
    
    # Scaling by Binormalization (Livne & Golub 2003)
    n <- nrow(A)
    p <- ncol(A)
    #d <- 1/sqrt(sqrt(apply(A, 2, function(xx) sqrt(max( abs(xx) )))))
    #d <- d / min(d)
    
    x <- rep(1, n)
    
    B <- A ^ 2
    d <- diag(B)
    beta <- drop(B %*% x)
    avg <- sum(beta * x) / n
    e <- 1
    
    std <- sqrt(sum((x * beta-avg)^2)/n)/avg
    
    
    for (i in 1:maxit)
    {
        x_old <- x
        
        for (j in 1:n)
        {
            bi <- beta[j]
            di <- d[j]
            xi <- x[j]
            c2 <- (n-1) * di
            c1 <- (n-2)*(bi-di*xi)
            c0 <- -di*xi^2 + 2*bi*xi - n*avg
            if (-c0 < eps)
            {
                C = A
                f <- rep(1, n)
                avg_c = -1
                G = -1
                est_c = -1
                break
            } else
            {
                xi = (2*c0)/(-c1 - sqrt(c1*c1 - 4*c2*c0))
            }
            
            delta = xi - x[j]
            avg <- avg + (delta * crossprod(x, B[,j]) + delta*bi + di*delta^2)/n
            beta = beta + delta*B[,j]
            x[j] = xi
        }
        std_old = std
        e_old = e
        std = sqrt(sum((x * beta-avg)^2)/n)/avg
        
        e = sqrt(sum( (x-x_old)^2 ))/sqrt(sum(x ^ 2))
        conv_factor = std/std_old
        
        
    }
    avg_c = (std/std_initial)^(1/i)
    f = sqrt(x)
    
    drop(f )
}

ruiz.equilibriate <- function(A, pnorm = 2, eps1 = 1e-2, eps2 = 1e-2, maxit = 100, verbose = FALSE)
{
    n <- nrow(A)
    p <- ncol(A)
    d1 <- rep(1, n)
    d2 <- rep(1, p)
    B <- A
    colnorm <- apply(B, 2, function(xx) sqrt(sum(   xx ^2   )) )
    rownorm <- apply(B, 1, function(xx) sqrt(sum(   xx ^2   )) )
    
    r1 <- r2 <- 1e99
    
    pnorm <- 2
    iters <- maxit
    for (i in 1:maxit)
    {
        r1.prev <- r1
        r2.prev <- r2
        d1 <- d1 * 1 / sqrt(rownorm) #apply(B, 1, function(xx) 1/sqrt( sqrt(sum(   xx ^2   ))  ) )
        d2 <- d2 * (n/p) ^ (0.5 / pnorm) * 1 / sqrt(colnorm) #apply(B, 2, function(xx) 1/sqrt( sqrt(sum(   xx ^2   )) ) )
        B <- diag(d1) %*% A %*% diag(d2)
        colnorm <- apply(B, 2, function(xx) sqrt(sum(   xx ^2   )) )
        rownorm <- apply(B, 1, function(xx) sqrt(sum(   xx ^2   )) )
        
        r1 <- max(rownorm) / min(rownorm)
        r2 <- max(colnorm) / min(rownorm)
        if (verbose) cat("r1", r1, ", r2", r2, "\n")
        if (abs(r1 - r1.prev) <= eps1 & (r2 - r2.prev) <= eps2)
        {
            iters <- i
            break
        }
    }
    list(d1 = d1, d2 = d2, B = B, iters = iters)
}

admm.lasso.R <- function(x, y, lambda, rho = NULL, abs.tol = 1e-5, rel.tol = 1e-5, maxit = 500L, gamma = 4) {
    require(Matrix)
    require(rARPACK)
    xtx <- crossprod(x)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    n <- length(y)
    p <- ncol(x)
    #lambda <- lambda * n
    iters <- maxit
    
    if (length(lambda) > 1) {
        warning("only works for one lambda value 
                at a time now; using first lambda value")
        lambda <- lambda[1]
    }
    
    ## if rho value is not supplied, 
    ## compute one that is good
    if (is.null(rho)) {
        eigs <- eigs_sym(xtx, k = 2, 
                         which = "BE", 
                         opts = list(maxitr = 500, 
                                     tol = 1e-4))$values
        #rho <- eigs[1] ^ (1 / 3) * lambda ^ (2 / 3)
        rho <- sqrt(eigs[1] * eigs[length(eigs)])
    }
    
    
    alpha <- z <- u <- numeric(p)
    A <- as(xtx + rho * diag(p), "Matrix")
    
    for (i in 1:maxit) {
        q <- xty + rho * (z - u)
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        z <- soft.thresh(alpha + u, lambda / rho)
        
        u <- u + (alpha - z)
        loss.history[i] <- l1.loss.leastsquares(x, y, alpha, z, lambda)
        
        r_norm = sqrt(sum( (alpha - z)^2 ))
        s_norm = sqrt(sum( (-rho * (z - z.prev))^2 ))
        eps_pri = sqrt(p) * abs.tol + rel.tol * max(sqrt(sum(alpha ^ 2)), sqrt(sum((-z)^2 ) ))
        eps_dual = sqrt(p) * abs.tol + rel.tol * sqrt(sum( (rho * u)^2 ))
        
        
        if (r_norm < eps_pri & s_norm < eps_dual) {
            iters <- i
            break
        }
    }
    
    list(beta=z, beta.aud = alpha, iters = iters, loss.hist = loss.history[!is.na(loss.history)])
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


