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
void horner(VectorXd &y, const VectorXd &x, const VectorXd &c) {
    int d = c.size() - 1;
    int s = c.size();

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct(x) + c[i] * ones;
    }
}

void evaluate(VectorXd &y, const VectorXd &x, const VectorXd &c) {
    int d = c.size() - 1;
    int s = c.size();

    y = VectorXd::Zero(x.size());

    for (int i = 1; i < s; i++) {
        y += c[i] * x.array().pow(i).matrix();
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

int main() {
    int max = 100;
    double d = 1.0 / max;

    vector<double> xAxis(max);
    vector<double> normal(max);
    vector<double> hornerScheme(max);

    for (int i = 0; i < max; i++) {
        int N = i + 2;
        xAxis[i] = N;

        VectorXd x = VectorXd::LinSpaced(N, 0, 1);
        VectorXd y = VectorXd::Random(N);

        VectorXd c;
        interpolate(c, x, y);

        VectorXd res;

        auto t1 = chrono::high_resolution_clock::now();
        evaluate(res, x, c);
        auto t2 = chrono::high_resolution_clock::now();
        horner(res, x, c);
        auto t3 = chrono::high_resolution_clock::now();

        normal[i] = chrono::duration_cast<chrono::microseconds>((t2 - t1)).count();
        hornerScheme[i] = chrono::duration_cast<chrono::microseconds>((t3 - t2)).count();
    }

    plt::figure();
    plt::xlabel("x");
    plt::ylabel("y");

    plt::named_plot("Normal evaluation", xAxis, normal, "r");
    plt::named_plot("Horner scheme", xAxis, hornerScheme, "g");
    plt::legend();
    plt::save("./horner.png");

    return 0;
}