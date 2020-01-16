#define WITHOUT_NUMPY 1

#include "../matplotlibcpp.h"
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
void evaluate(VectorXd &y, const VectorXd &x, const VectorXd &c, const VectorXd &t) {
    assert(c.size() <= t.size());

    int s = c.size();
    int d = s - 1;

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct((x.array() - t[d - i]).matrix()) + c[i] * ones;
    }
}

void interpolate(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    MatrixXd A(n, n);
    A.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < n; i++) {
        A.col(i) = A.col(i - 1).cwiseProduct((x.array() - x[i - 1]).matrix());
    }

    c = A.template triangularView<Lower>().solve(y).reverse();
}

void dividedDifferences(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    c = y;

    for (int l = 0; l < n; l++) {
        for (int k = n - l; k < n; k++) {
            c[k] = (c[k] - c[k - 1]) / (x[k] - x[n - 1 - l]);
        }
    }

    c.reverseInPlace();
}

void polyfit(VectorXd &c, VectorXd &t, const VectorXd &x, const VectorXd &y, int d) {
    assert(x.size() == y.size());

    int n = y.size();

    MatrixXd A(n, d + 1);
    A.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < d + 1; i++) {
        A.col(i) = A.col(i - 1).cwiseProduct((x.array() - x[i - 1]).matrix());
    }

    c = A.householderQr().solve(y).reverse();
    t = x;
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
    dividedDifferences(coefficients, x, y);
    evaluate(yInterpolation, xSampling, coefficients, x);

    vector<double> vecXSampling = toCpp(xSampling);
    vector<double> vecYInterpolation = toCpp(yInterpolation);

    plt::plot(vecX, vecY, {{"label", "Input nodes"}, {"color", "blue"}, {"marker", "o"}});
    plt::plot(vecXSampling, vecYInterpolation, {{"label", "Polynomial interpolation for d=N-1"}, {"color", "red"}});

    string name = "Polyfit with d = ";

    for (int i = 0; i < N - 1; i++) {
        VectorXd c, t, yFit;
        polyfit(c, t, x, y, i);
        evaluate(yFit, xSampling, c, t);

        vector<double> vecYFit = toCpp(yFit);

        plt::plot(vecXSampling, vecYFit, {{"label", name + to_string(i)}, {"linestyle", "--"}});
    }

    plt::legend();
    plt::savefig("./newton.png");

    return 0;
}