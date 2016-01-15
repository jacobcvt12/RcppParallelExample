// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

arma::vec rnormArma(int n, double mean=0.0, double variance=1.0) {
    arma::vec draws = arma::randn(n, 1);

    // multiply by sqrt of variance
    draws *= sqrt(variance);

    // add mean
    draws += mean;

    return draws;
}

arma::vec rdirichletArma(int n, arma::vec alpha) {
    arma::vec Y(alpha.size());
    arma::vec X(alpha.size());

    // draw Y_k ~ gamma(a_k, 1)
    for (int k = 0; k < alpha.size(); ++k) {
        double a = alpha[k];
        double theta = 1.;
        Y[k] = arma::conv_to<double>::from(arma::randg(1, arma::distr_param(a, theta)));
    }

    // calculate V ~ gamma(\sum alpha_i, 1)
    double V = arma::sum(Y);

    // calculate X ~ Dir(alpha_1, ..., alpha_k)
    for (int k = 0; k < alpha.size(); ++k) {
        X[k] = Y[k] / V;
    }

    return X;
}

// [[Rcpp::export]]
std::vector<int> rmultinomArma(int n, arma::vec p) {
    arma::vec p_sum = arma::cumsum(p);
    arma::vec draws = arma::randu(n);
    std::vector<int> multi(p.size());

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < p.size(); ++k) {
            if (draws[i] <= p_sum[k]) {
                multi[k]++;
                break;
            }
        }
    }

    return multi;
}

// [[Rcpp::export]]
Rcpp::List mixture_model(arma::vec y, int k=3,
                         int burnin=1000, int iter=1000, int chains=1) {

    // initialize parameters
    arma::vec mu = rnormArma(k, 0.0, 1000.0);
    arma::vec sigma_2 = 1. / arma::randg(k, arma::distr_param(0.01, 0.01));
    arma::vec w(k);
    arma::vec S(y.size());

    return Rcpp::List::create(Rcpp::Named("y")=y);
}
