
#' @export
admm.genlasso.R <- function(x, y, D, lambda, rho = NULL, abs.tol = 1e-5, rel.tol = 1e-5, maxit = 500L, gamma = 4, tau = 1) {
    require(Matrix)
    require(rARPACK)
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
        rho <- eigs[1] ^ (1 / 3) * lambda ^ (2 / 3)
        
    }
    
    alpha <- numeric(p)
    z <- u <- numeric(npen)
    A <- as(xtx + rho * DtD, "Matrix")
    
    for (i in 1:maxit) 
    {
        q      <- xty + rho * crossprod(D, (z - u))
        alpha  <- as.vector(solve(A, q))
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

#' @export
admm.genlasso2.R <- function(x, y, D, lambda, rho = NULL, abs.tol = 1e-5, rel.tol = 1e-5, maxit = 500L, gamma = 4) {
    require(Matrix)
    require(rARPACK)
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
        rho <- eigs[1] ^ (1 / 3) * lambda ^ (2 / 3)
        
    }
    
    alpha <- numeric(p)
    z <- u <- numeric(npen)
    A <- as(xtx + rho * DtD, "Matrix")
    
    epsCor <- 0.2
    
    for (i in 1:maxit) {
        q <- xty + rho * crossprod(D, (z - u / rho))
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        if (i > 1) Dalpha_prev <- Dalpha
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        Dalpha <- D %*% alpha
        z <- soft.thresh(Dalpha + u / rho, lambda / rho)
        
        #if (i == 2) 
        #{
        #    print(Dalpha_prev[1:5])
        #    print(Dalpha[1:5])
        #}
            
        
        deltau <- rho * (Dalpha - z)
        if (i > 1) deltaH <- Dalpha - Dalpha_prev
        
        if (i > 1)
        {
            crossHu <- -sum(drop(deltau) * drop(deltaH))
            alphaSD <- sum(deltau ^ 2) / crossHu
            ssH <- sum(deltaH ^ 2)
            
            if (ssH == 0)
            {
                alphaMG <- 0
            } else
            {
                alphaMG <- crossHu / ssH
            }
                
            
            if (2 * alphaMG > alphaSD)
            {
                alphak <- alphaMG
            } else
            {
                alphak <- alphaSD - alphaMG * 0.5
            }
        }
        
        deltaG <- (z - z.prev)
        
        if (i > 1 & i %% 2 == 0)
        {
            crossHz <- sum(drop(deltau) * drop(deltaG))
            betaSD <- sum(deltau ^ 2) / crossHz
            
            ssG <- sum(deltaG ^ 2)
            if (ssG == 0)
            {
                betaMG <- 0
            } else
            {
                betaMG <- crossHz / ssG
            }
                
            
            if (2 * betaMG > betaSD)
            {
                betak <- betaMG
            } else
            {
                betak <- betaSD - betaMG * 0.5
            }
            
            if (ssH > 0)
            {
                alphaCor <- crossHu / (sqrt(ssH) * sqrt(sum(deltau ^ 2)))   
            } else
            {
                alphaCor <- 0
            }
                
            if (ssG > 0)
            {
                betaCor  <- crossHz / (sqrt(ssG) * sqrt(sum(deltau ^ 2)))
            } else
            {
                betaCor <- 0
            }
            
            #cat("alpha cor:", alphaCor, "betaCor:", betaCor, "crossHu", crossHu, "crossHz", crossHz, "deltaHss", sum(deltaH ^ 2), "deltaGss", sum(deltaG ^ 2), "\n")
            
            rhoupdate <- TRUE
            if (alphaCor > epsCor & betaCor > epsCor)
            {
                rhok <- sqrt(alphak * betak)
            } else if (alphaCor > epsCor & betaCor <= epsCor)
            {
                rhok <- alphak
            } else if (alphaCor <= epsCor & betaCor > epsCor)
            {
                rhok <- betak
            } else
            {
                rhoupdate <- FALSE
            }
            
            
            
            if (rhoupdate)
            {
                oldrho <- rho
                rho <- rhok
                A <- as(xtx + rho * DtD, "Matrix")
                
                cat("updated rho:", rho, "old rho:", oldrho, "\n")
            }
            
        }
        
        
        u <- u + deltau
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



#' @export
ama.genlasso.R <- function(x, y, D, lambda, rho = NULL, abs.tol = 1e-5, rel.tol = 1e-5, maxit = 500L, gamma = 4) {
    require(Matrix)
    require(rARPACK)
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
        rho <- eigs[1] ^ (1 / 3) * lambda ^ (2 / 3)
        
    }
    
    alpha <- numeric(p)
    z <- u <- numeric(npen)
    A <- as(xtx, "Matrix")
    
    for (i in 1:maxit) {
        q <- xty - drop(crossprod(D, u))
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        Dalpha <- D %*% alpha
        z <- soft.thresh(Dalpha + u / rho, lambda / rho)
        
        u <- u + rho * (Dalpha - z)
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


alin <- function(x, y, D, lambda, gamma = 0.2, maxit = 100, tol = 1e-5)
{
    xtx <- crossprod(x)
    xty <- crossprod(x, y)
    
    n <- length(y)
    p <- ncol(x)
    npen <- nrow(D)
    
    
    
    
    beta.hat <- beta.tilde.h <- beta.tilde.f <- numeric(p)
    z <- u <- numeric(npen)
    iters <- maxit
    
    d <- diag(xtx)
    A <- as(xtx + diag(d), "Matrix")
    
    test.lhs.cur <- 1e99
    for (i in 1:maxit)
    {
        
        beta.prev <- beta.hat
        sf <- drop(xtx %*% beta.tilde.f - xty)
        
        tau <- beta.hat - sf / d
        beta.tilde.h <- soft.thresh(tau, lambda / d)
        
        test.lhs.prev <- test.lhs.cur
        
        test.lhs <- l1.loss.leastsquares(x, y, beta.tilde.h, beta.tilde.h, lambda)
        loss.beta.hat <- l1.loss.leastsquares(x, y, beta.hat, beta.hat, lambda)
        test.rhs <- (1 - gamma) * loss.beta.hat + gamma * test.lhs
        
        test.lhs.cur <- test.lhs
        
        if (test.lhs < test.rhs)
        {
            beta.hat <- beta.tilde.h
        }
        
        #if (i > 2 & test.lhs >= loss.beta.hat - tol)
        #    #if (all(abs(beta.hat - beta.tilde.h) <= tol * (1e-4 + sum(abs(beta.tilde.h)))) & 
        #    #    all(abs(beta.hat - beta.tilde.f) <= tol * (1e-4 + sum(abs(beta.tilde.f)))))
        #{
        #    iters <- i
        #    break
        #}
        
        sh <- -sf - d * (beta.tilde.h - beta.hat)
        
        delta <- as.vector(solve(A, xty - xtx %*% beta.hat - sh))
        
        beta.tilde.f <- beta.hat + delta
        
        test.lhs      <- l1.loss.leastsquares(x, y, beta.tilde.f, beta.tilde.f, lambda)
        loss.beta.hat <- l1.loss.leastsquares(x, y, beta.hat,     beta.hat,     lambda)
        test.rhs      <- (1 - gamma) * loss.beta.hat + gamma * test.lhs
        
        if (test.lhs < test.rhs)
        {
            beta.hat <- beta.tilde.f
        }
        
        
        #if (test.lhs >= loss.beta.hat - tol)
        if (all(abs(beta.hat - beta.tilde.h) <= tol * (1e-4 + sum(abs(beta.tilde.h)))) & 
            all(abs(beta.hat - beta.tilde.f) <= tol * (1e-4 + sum(abs(beta.tilde.f)))))
        {
            iters <- i
            break
        }
        
        
    }
    list(beta = beta.hat, iters = iters)
}




#' @export
admm.restart.genlasso.R <- function(x, y, D, lambda, rho = NULL, maxit = 100, tol = 1e-5)
{
    xtx <- crossprod(x) / n
    xty <- crossprod(x, y) / n
    dtd <- crossprod(D)
    
    n <- length(y)
    p <- ncol(x)
    npen <- nrow(D)
    
    nlam <- length(lambda)
    
    ## if rho value is not supplied, 
    ## compute one that is good
    if (is.null(rho)) {
        eigs <- eigs_sym(xtx, k = 2, 
                         which = "BE", 
                         opts = list(maxitr = 500, 
                                     tol = 1e-4))$values
        
        
    }
    
    alpha <- numeric(p)
    z <- u <- u.hat <- z.hat <- rep(0, npen)
    
    
    
    ak <- 1
    ck <- 1e10
    eta <- 0.99
    
    beta.mat <- array(NA, dim = c(p, nlam))
    z.mat    <- array(NA, dim = c(npen, nlam))
    iter.list <- numeric(nlam)
    for (l in 1:nlam)
    {
        iters <- maxit
        rho <- eigs[1] ^ (1 / 3) * lambda[l] ^ (2 / 3)
        A <- as(xtx + 1 * rho * dtd, "Matrix")
        
        u.hat <- z.hat <- rep(0, npen)
    
        for (i in 1:maxit)
        {
            q     <- xty + drop(crossprod(D, 1 * rho * z.hat - u.hat))
            alpha <- as.vector(solve(A, q))
            z.prev <- z
            
            Dalpha <- D %*% alpha
            
            z <- soft.thresh(Dalpha + u.hat / rho, lambda[l] / rho)
            
            u.prev <- u
            u <- u.hat + rho * (Dalpha - z)
            
            ck.prev <- ck
            ck <- sum( (u - u.hat) ^ 2 ) / rho + rho * sum((z - z.hat) ^ 2)
            
            if (ck < eta * ck.prev)
            {
                ak.prev <- 1 * ak
                ak <- (1 + sqrt(1 + 4 * ak ^ 2)) * 0.5
                u.hat <- u + (ak.prev / ak) * (u - u.prev)
                z.hat <- z + (ak.prev / ak) * (z - z.prev)
            } else 
            {
                ak <- 1
                u.hat  <- u.prev
                z.hat <- z.prev
                ck <- ck.prev / eta
            }
            
            
            if (i > 2 & all(abs(z - z.prev) < tol))
            {
                iters <- i
                break
            }
        }
        
        beta.mat[,l] <- alpha
        z.mat[,l]    <- z.hat
        
        iter.list[l] <- iters
    }
    list(beta = beta.mat, iters = iter.list, z = z.mat, lambda = lambda)
}


