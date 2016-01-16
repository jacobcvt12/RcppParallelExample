normal.gibbs <- function(data, mu0=0, t20=100, nu0=1, s20=1,
                         burnin=1000, iter=1000, chains=1) {
    # approximate posterior distribution of mu and s2
    mcmc <- normal_gibbs(data, mu0, t20, nu0, s20, 
                         burnin=burnin, iter=iter, chains=chains)

    # get calculated parameters
    # each is a matrix where rows are the iterations 
    # and cols are the chains
    mu <- mcmc$mu
    s2 <- mcmc$s2

    message("mu: R.hat ", round(gelman.rubin(mu), 3))
    message("s2: R.hat ", round(gelman.rubin(s2), 3))

    return(mcmc)
}

gelman.rubin <- function(param) {
    # mcmc information
    n <- nrow(param) # number of iterations
    m <- ncol(param) # number of chains

    # calculate the mean of the means
    theta.bar.bar <- mean(colMeans(param))

    # within chain variance
    W <- mean(apply(param, 2, var))

    # between chain variance
    B <- n / (m - 1) * sum((colMeans(param) - theta.bar.bar) ^ 2)

    # variance of stationary distribution
    theta.var.hat <- (1 - 1 / n) * W + 1 / n * B

    # Potential Scale Reduction Factor (PSRF)
    R.hat <- sqrt(theta.var.hat / W)

    return(R.hat)
}
