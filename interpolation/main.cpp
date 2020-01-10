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

void dividedDiff(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    c = y;

    for (int l = 0; l < n; l++) {
        for (int k = n - l; k < n; k++) {
            c[k] = (c[k] - c[k - 1]) / (x[k] - x[n - 1 - l]);
        }
    }

    // c.reverseInPlace();
}

void newton(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    MatrixXd A(n, n);
    A.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < n; i++) {
        A.col(i) = A.col(i - 1).cwiseProduct((x.array() - x[i - 1]).matrix());
    }

    c = A.template triangularView<Lower>().solve(y).reverse();
}

void monomial(VectorXd &c, const VectorXd &x, const VectorXd &y) {
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

    plt::figure();
    plt::xlabel("x");
    plt::ylabel("y");

    int M = 100;

    vector<double> xAxis(M);
    vector<double> time1(M);
    vector<double> time2(M);
    vector<double> time3(M);

    for (int i = 0; i < M; i++) {
        int N = i + 2;

        xAxis[i] = N;
        VectorXd x = VectorXd::LinSpaced(N, 0, 1);
        VectorXd y = VectorXd::Random(N);

        VectorXd c1, c2, c3;

        auto t1 = chrono::high_resolution_clock::now();
        monomial(c1, x, y);
        auto t2 = chrono::high_resolution_clock::now();
        newton(c2, x, y);
        auto t3 = chrono::high_resolution_clock::now();
        dividedDiff(c3, x, y);
        auto t4 = chrono::high_resolution_clock::now();

        time1[i] = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        time2[i] = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();
        time3[i] = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();
    }

    plt::named_plot("Monomial", xAxis, time1);
    plt::named_plot("Newton", xAxis, time2);
    plt::named_plot("Divided differences", xAxis, time3);

    plt::legend();
    plt::save("./interpolation.png");

    return 0;
}