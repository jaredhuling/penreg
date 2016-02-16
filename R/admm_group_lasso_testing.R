

admm.group.lasso.R <- function(x, y, groups, lambda, gr.weights = NULL,
                               rho, 
                               abs.tol = 1e-5, rel.tol = 1e-5, 
                               maxit = 500L, gamma = 4) {
    library(Matrix)
    unique.groups <- sort(unique(groups))
    n.groups <- length(unique.groups)
    gr.idx.list <- vector(mode = "list", length = n.groups)
    for (g in 1:n.groups) {
        gr.idx.list[[g]] <- which(groups == unique.groups[g])
    }
    if (is.null(gr.weights)) {
        gr.weights <- numeric(n.groups)
        for (g in 1:n.groups) {
            gr.weights[g] <- sqrt(length(gr.idx.list[[g]]))
        }
    } 
    xtx <- crossprod(x)
    xty <- crossprod(x, y)
    loss.history <- rep(NA, maxit)
    n <- length(y)
    p <- ncol(x)
    #lambda <- lambda * n
    iters <- maxit
    
    alpha <- z <- u <- numeric(p)
    A <- as(xtx + rho * diag(p), "Matrix")
    

    
    for (i in 1:maxit) {
        q <- xty + rho * (z - u)
        alpha <- as.vector(solve(A, q))
        z.prev <- z
        
        #z <- sign(alpha + u) * pmax(abs(alpha + u) - lambda / rho, 0) / (1 - 1/gamma)
        #z <- ifelse( abs(alpha + u) <= gamma * (lambda/rho), z, alpha + u)
        
        for (g in 1:n.groups) {
            gr.idx <- gr.idx.list[[g]]
            z[gr.idx] <- block.soft.thresh(alpha[gr.idx] + u[gr.idx], gr.weights[g] * lambda / rho)
        }
        
        u <- u + (alpha - z)
        loss.history[i] <- group.lasso.loss.leastsquares(x, y, alpha, z, lambda, gr.idx.list, gr.weights)
        
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

block.soft.thresh <- function(a, lambda) {
    a * pmax(0, 1 - lambda / sqrt(sum(a ^ 2)))
}

soft.thresh <- function(a, lambda) {
    pmax(0, a - lambda) - pmax(0, -a - lambda)
}

l1.loss.leastsquares <- function(x, y, beta, z, lambda)
{
    0.5 * sum((y - x %*% beta) ^ 2) + lambda * sum(abs(z))
}

group.lasso.loss.leastsquares <- function(x, y, beta, z, lambda, gr.idx.list, gr.weights)
{
    loss <- 0.5 * sum((y - x %*% beta) ^ 2)
    for (g in 1:length(gr.idx.list)) {
        loss <- loss + gr.weights[g] * sqrt(sum(z[gr.idx.list[[g]]] ^ 2))
    }
    loss
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
    library(Matrix)
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


