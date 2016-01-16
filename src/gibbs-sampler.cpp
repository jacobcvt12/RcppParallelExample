#include <RcppArmadillo.h>
#include <cmath>
#include <omp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

arma::vec rnormArma(int n, double mean=0.0, double variance=1.0) {
    arma::vec draws = arma::randn(n, 1);

    // multiply by sqrt of variance
    draws *= sqrt(variance);

    // add mean
    draws += mean;

    return draws;
}

// gibbs sampler for normal distribution
// [[Rcpp::export]]
Rcpp::List normal_gibbs(arma::vec data, double mu0, double t20, double nu0, double s20, 
                        int burnin=1000, int iter=1000, int chains=1) {
    // initialize parameters
    double data_mean = arma::mean(data);
    double data_var = arma::var(data);
    int n = data.size();
    double mu = data_mean;
    double s2 = data_var;

    // initialize chains
    arma::mat mu_chain(iter, chains);
    arma::mat s2_chain(iter, chains);

    #pragma omp parallel for num_threads(chains)
    for (int chain = 0; chain < chains; ++chain) {
        // burnin
        for (int b = 0; b < burnin; ++b) {
            // update mu
            double mu_n = (mu0 / t20 + n * data_mean * (1. / s2)) / (1. / t20 + n * (1 / s2));
            double t2_n = 1 / (1 / t20 + n / (s2));
            mu = arma::conv_to<double>::from(rnormArma(1, mu_n, t2_n));

            // update s2
            double nu_n = nu0 + n;
            double s2_n = (nu0 * s20 + (n-1) * data_var + n * pow(data_mean - mu, 2)) / nu_n;
            s2 = arma::conv_to<double>::from(arma::randg(1, arma::distr_param(nu_n / 2., 2. / (nu_n *s2_n))));
        }

        // burnin
        for (int s = 0; s < iter; ++s) {
            // update mu
            double mu_n = (mu0 / t20 + n * data_mean * (1. / s2)) / (1. / t20 + n * (1 / s2));
            double t2_n = 1 / (1 / t20 + n / (s2));
            mu = arma::conv_to<double>::from(rnormArma(1, mu_n, t2_n));

            // update s2
            double nu_n = nu0 + n;
            double s2_n = (nu0 * s20 + (n-1) * data_var + n * pow(data_mean - mu, 2)) / nu_n;
            s2 = 1. / arma::conv_to<double>::from(arma::randg(1, arma::distr_param(nu_n / 2., 2. / (nu_n *s2_n))));

            // store values
            mu_chain(s, chain) = mu;
            s2_chain(s, chain) = s2;
        }
    }


    return Rcpp::List::create(Rcpp::Named("mu")=mu_chain,
                              Rcpp::Named("s2")=s2_chain);
}
