// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <algorithm>
#include <random>
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
std::vector<int> sampleArma(int n, int k) {
    std::vector<int> S(n);
    arma::vec prob_window(k);
    prob_window.fill(1. / (double) k);
    prob_window = arma::cumsum(prob_window);

    for (int i = 0; i < n; ++i) {
        S[i] = std::rand() % k;
    }

    return S;
}

// [[Rcpp::export]]
Rcpp::List mixture_model(arma::vec y, int k=3,
                         int burnin=1000, int iter=1000, int chains=1) {
    // accessory variables
    arma::vec probs(k);
    probs.fill(1. / (double) k);
    double b_0 = arma::mean(y);
    double B_0 = arma::var(y);
    double c_0 = 1.;
    double C_0 = arma::var(y);
    int N = y.size();

    // initialize parameters
    arma::vec mu = rnormArma(k, 0.0, 1000.0);
    arma::vec sigma2 = 1. / arma::randg(k, arma::distr_param(0.01, 0.01));
    arma::vec w(k);
    std::vector<int> S = sampleArma(y.size(), k);

    // initialize chains
    arma::mat mu_chain(iter, k);
    arma::mat sigma2_chain(iter, k);
    arma::mat w_chain(iter, k);

    // burnin
    for (int B = 0; B < burnin; ++B) {
        // mu full conditional
        for (int i = 0; i < k; ++i) {
            int N_k = std::count(S.begin(), S.end(), k);
            double y_k_mean = 0.0;

            for (int j = 0; j < N; ++j) {
                if (S[j] == k) {
                    y_k_mean += y[j];
                }
            }

            y_k_mean /= (double) N_k;
            double b_k = B_0 * b_0 / (N_k + B_0) + N_k * y_k_mean / (N_k + B_0);
            double B_k = sigma2[k] / (N_k + B_0);

            mu[k] = arma::conv_to<double>::from(rnormArma(1, b_k, B_k));
        }

        // sigma full conditional
        for (int i = 0; i < k; ++i) {
            int N_k = std::count(S.begin(), S.end(), k);
            double c_k = c_0 + 0.5 * N_k;
            double n = 0.0;
            double mean = 0.0;
            double M2 = 0.0;
            double y_k_mean = 0.0;

            for (int j = 0; j < N; ++j) {
                if (S[j] == k) {
                    n += 1;
                    double delta = y[j] - mean;
                    mean += delta / n;
                    M2 += delta * (y[j] - mean);
                    y_k_mean += y[j];
                }
            }

            y_k_mean /= (double) N_k;
            double s_k = M2 / (n - 1);
            double C_k = C_0 + 0.5 * (N_k * s_k + (N_k * B_0 / (N_k + B_0)) * pow(y_k_mean - b_0, 2.0));

            sigma2[k] = arma::conv_to<double>::from(1. / arma::randg(1, arma::distr_param(c_k, C_k)));
        }
        
    }

    // MCMC sampling
    for (int B = 0; B < burnin; ++B) {
    }

    return Rcpp::List::create(Rcpp::Named("y")=y);
}

// gibbs sampler for bivariate distribution
// f(x, y)=kx^2 exp(-xy^2-y^2+2y-4x), x>0 y \in (-\infty, \infty)
// [[Rcpp::export]]
Rcpp::List bivariate(arma::vec data, int burnin=1000, int iter=1000, int chains=1) {
    // initialize parameters
    arma::vec x = arma::randg(1, arma::distr_param(0.001, 0.001));
    arma::vec y = rnormArma(1, 0, 1000);

    // initialize chains
    arma::vec x_chain(iter);
    arma::vec y_chain(iter);

    // burnin
    for (int B = 0; B < burnin; ++B) {
        // solve for full conditional parameters
        double shape = 3;
        double scale = arma::conv_to<double>::from(1. / (y * y + 4));

        double mean = arma::conv_to<double>::from(1. / (x + 1));
        double variance = arma::conv_to<double>::from(1. / (2 * x + 2));

        x = arma::randg(1, arma::distr_param(shape, scale));
        y = rnormArma(1, mean, variance);
    }

    // sample from posterior
    for (int s = 0; s < iter; ++s) {
        // solve for full conditional parameters
        double shape = 3;
        double scale = arma::conv_to<double>::from(1. / (y * y + 4));

        double mean = arma::conv_to<double>::from(1. / (x + 1));
        double variance = arma::conv_to<double>::from(1. / (2 * x + 2));

        x = arma::randg(1, arma::distr_param(shape, scale));
        y = rnormArma(1, mean, variance);

        x_chain[s] = arma::conv_to<double>::from(x);
        y_chain[s] = arma::conv_to<double>::from(y);
    }

    return Rcpp::List::create(Rcpp::Named("x")=x_chain,
                              Rcpp::Named("y")=y_chain);
}
