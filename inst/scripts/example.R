library(RcppParallel)

# generate data
mu.true <- 20
s2.true <- 9
data <- rnorm(100000, mu.true, sqrt(s2.true))

# run gibbs sampler
mcmc <- normal.gibbs(data, chains=3)
