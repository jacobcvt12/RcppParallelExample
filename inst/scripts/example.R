library(RcppParallel)

# generate data
mu.true <- 20
s2.true <- 9
data <- rnorm(1000, mu.true, sqrt(s2.true))

# run gibbs sampler
mcmc <- normal_gibbs(data, 0, 100, 1, 1)
mu <- mcmc$mu
s2 <- mcmc$s2
