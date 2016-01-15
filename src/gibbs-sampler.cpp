// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

arma::vec rnormArma(int n, double mean=0.0, double variance=1.0) {
    arma::vec draws = arma::randn(n, 1);

    // multiply by sqrt of variance
    draws *= sqrt(variance);

    // add mean
    draws += mean;

    return draws;
}

// [[Rcpp::export]]
Rcpp::List mixture_model(arma::vec y, int k=3,
                         int burnin=1000, int iter=1000, int chains=1) {

    // initialize parameters
    arma::vec mu = arma::randn(k, 1);
    arma::vec sigma_2(k);
    arma::vec w(k);
    arma::vec S(y.size());

    return Rcpp::List::create(Rcpp::Named("y")=y);
}
