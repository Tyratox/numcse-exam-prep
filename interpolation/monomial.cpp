#define WITHOUT_NUMPY 1

#include "matplotlibcpp.h"
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

/**
 * Evaluates a polynomial of degree c.size()-1 with coefficients c at all positions x and
 * stores the results in y. c[0] contains the coefficient of maximal degree
*/
void evaluate(VectorXd &y, const VectorXd &x, const VectorXd &c) {
    int d = c.size() - 1;
    int s = c.size();

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct(x) + c[i] * ones;
    }
}

void interpolate(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    MatrixXd V(n, n); //vandermonde matrix
    V.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < n; i++) {
        V.col(i) = V.col(i - 1).cwiseProduct(x);
    }

    c = V.lu().solve(y).reverse();
}

void polyfit(VectorXd &c, const VectorXd &x, const VectorXd &y, int d) {
    assert(x.size() == y.size());

    int n = y.size();

    MatrixXd V(n, d + 1); //vandermonde matrix
    V.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < d + 1; i++) {
        V.col(i) = V.col(i - 1).cwiseProduct(x);
    }

    c = V.householderQr().solve(y).reverse();
}

vector<double> toCpp(VectorXd v) {
    vector<double> vec(v.data(), v.data() + v.size());
    return vec;
}

int main() {

    plt::figure();
    plt::figure_size(2400, 2400);
    plt::xlabel("x");
    plt::ylabel("y");
    plt::xlim(0, 1);
    plt::ylim(-3, 3);

    int N = 10;

    VectorXd x = VectorXd::LinSpaced(N, 0, 1);
    VectorXd y = VectorXd::Random(N);

    vector<double> vecX = toCpp(x);
    vector<double> vecY = toCpp(y);

    VectorXd xSampling = VectorXd::LinSpaced(N * 100, 0, 1);

    VectorXd coefficients, yInterpolation;
    interpolate(coefficients, x, y);
    evaluate(yInterpolation, xSampling, coefficients);

    vector<double> vecXSampling = toCpp(xSampling);
    vector<double> vecYInterpolation = toCpp(yInterpolation);

    plt::named_plot("Input nodes", vecX, vecY, "bo");
    plt::named_plot("Polynomial interpolation for d=N-1", vecXSampling, vecYInterpolation, "r");

    string name = "Polyfit with d = ";

    for (int i = 0; i < N - 1; i++) {
        VectorXd c, yFit;
        polyfit(c, x, y, i);
        evaluate(yFit, xSampling, c);

        vector<double> vecYFit = toCpp(yFit);

        plt::named_plot(name + to_string(i), vecXSampling, vecYFit, "--");
    }

    plt::legend();
    plt::save("./monomial.png");

    return 0;
}