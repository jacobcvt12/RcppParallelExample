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
    arma::vec mu_chain(iter);
    arma::vec s2_chain(iter);

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
        s2 = arma::conv_to<double>::from(arma::randg(1, arma::distr_param(nu_n / 2., 2. / (nu_n *s2_n))));

        // store values
        mu_chain[s] = mu;
        s2_chain[s] = s2;
    }


    return Rcpp::List::create(Rcpp::Named("mu")=mu_chain,
                              Rcpp::Named("s2")=s2_chain);
}
