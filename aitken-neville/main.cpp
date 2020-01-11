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

void evaluateNewton(VectorXd &y, const VectorXd &x, const VectorXd &c, const VectorXd &t) {
    assert(c.size() <= t.size());

    int s = c.size();
    int d = s - 1;

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct((x.array() - t[d - i]).matrix()) + c[i] * ones;
    }
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

void aitkenNeville(double &d, const VectorXd &t, VectorXd y, double x) {
    for (int l = 1; l < y.size(); l++) {
        for (int k = l - 1; k >= 0; k--) {
            y(k) = y(k + 1) + ((x - t(l)) / (t(l) - t(k))) * (y(k + 1) - y(k));
        }
    }

    d = y(0);
}

int main() {
    int max = 100;
    double d = 1.0 / max;

    vector<double> xAxis(max);
    vector<double> interpolation1(max);
    vector<double> interpolation2(max);
    vector<double> aNeville(max);
    vector<double> err1(max);
    vector<double> err2(max);
    vector<double> err3(max);

    auto f = [](double x) {
        return 3 * pow(sin(x), 3) + 2 * x - 5 * sin(x);
    };

    VectorXd s = VectorXd(1);
    s << 0.54321;

    double correct = f(s(0));

    for (int i = 0; i < max; i++) {
        int N = i + 2;
        xAxis[i] = N;

        VectorXd x = VectorXd::LinSpaced(N, 0, 1);
        VectorXd y(N);
        for (int j = 0; j < N; j++) {
            y(j) = f(x(j));
        }

        VectorXd c;
        VectorXd res1, res2;
        double res3;

        auto t1 = chrono::high_resolution_clock::now();
        interpolate(c, x, y);
        evaluate(res1, s, c);
        auto t2 = chrono::high_resolution_clock::now();
        newton(c, x, y);
        evaluateNewton(res2, s, c, x);
        auto t3 = chrono::high_resolution_clock::now();
        aitkenNeville(res3, x, y, s(0));
        auto t4 = chrono::high_resolution_clock::now();

        err1[i] = abs(res1(0) - correct);
        err2[i] = abs(res2(0) - correct);
        err3[i] = abs(res3 - correct);
        interpolation1[i] = chrono::duration_cast<chrono::microseconds>((t2 - t1)).count();
        interpolation2[i] = chrono::duration_cast<chrono::microseconds>((t3 - t2)).count();
        aNeville[i] = chrono::duration_cast<chrono::microseconds>((t4 - t3)).count();
    }

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("time");

    plt::plot(xAxis, interpolation1, {{"label", "monomial+eval"}, {"color", "red"}});
    plt::plot(xAxis, interpolation2, {{"label", "newton+eval"}, {"color", "blue"}});
    plt::plot(xAxis, aNeville, {{"label", "Aitken Neville"}, {"color", "green"}});
    plt::legend();
    plt::savefig("./aitken-neville.png");

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("y");
    plt::semilogy(xAxis, err1, {{"label", "Error monomtial+eval"}, {"color", "red"}});
    plt::semilogy(xAxis, err2, {{"label", "Error newton+eval"}, {"color", "green"}});
    plt::semilogy(xAxis, err3, {{"label", "Error Aitken-Neville"}, {"color", "blue"}});
    plt::legend();

    plt::savefig("./error.png");

    return 0;
}