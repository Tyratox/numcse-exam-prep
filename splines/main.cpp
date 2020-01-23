#define WITHOUT_NUMPY 1

#include "../matplotlibcpp.h"
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

vector<double> toCpp(VectorXd v) {
    return vector<double>(v.data(), v.data() + v.size());
}

template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

VectorXd hermloceval(VectorXd t, double t1, double t2, double y1, double y2, double c1, double c2) {
    const double h = t2 - t1, a1 = y2 - y1, a2 = a1 - h * c1, a3 = h * c2 - a1 - a2;
    t = ((t.array() - t1) / h).matrix();
    return (y1 + (a1 + (a2 + a3 * t.array()) * (t.array() - 1)) * t.array()).matrix();
}

int main() {

    int n = 10;
    VectorXd x(n + 1);
    for (int i = 0; i < x.size(); i++) {
        x[i] = -1 + 0.2 * i;
    }

    VectorXd y(n + 1);
    for (int i = 0; i < x.size(); i++) {
        y[i] = sin(5 * x[i]) * exp(x[i]);
    }

    int N = 100;

    vector<Triplet<double>> triplets;
    triplets.reserve(3 * (n - 1) + 2 * 2 /*natural cubic spline interpolant conditions*/);

    VectorXd h = x.tail(n) - x.head(n);               //n elements
    VectorXd b = (1.0 / h.array()).matrix();          //n elements
    VectorXd a = 2 * (b.tail(n - 1) + b.head(n - 1)); //n-1 elements

    for (int i = 0; i < (n - 1); i++) {
        triplets.push_back(Triplet<double>(i, i, b[i]));         //diagonal
        triplets.push_back(Triplet<double>(i, i + 1, a[i]));     //superdiagonal
        triplets.push_back(Triplet<double>(i, i + 2, b[i + 1])); //supersuperdiagonal
    }

    //natural spline interpolant conditions
    triplets.push_back(Triplet<double>(n - 1, 0, 2.0 / h[0]));
    triplets.push_back(Triplet<double>(n - 1, 1, 1.0 / h[0]));

    triplets.push_back(Triplet<double>(n, n - 1, 1.0 / h[n - 1]));
    triplets.push_back(Triplet<double>(n, n, 2.0 / h[n - 1]));

    SparseMatrix<double> A(n + 1, n + 1);
    A.setFromTriplets(triplets.begin(), triplets.end());

    SparseLU<SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    VectorXd r(n + 1);
    for (int i = 0; i < n - 1; i++) {
        r[i] = 3 * ((y[i + 1] - y[i]) / (h[i] * h[i]) + (y[i + 2] - y[i + 1]) / (h[i + 1] * h[i + 1]));
    }

    //natural cubic spline interpolant conditions
    r[n - 1] = 3 * ((y[1] - y[0]) / (h[0] * h[0]));
    r[n] = 3 * ((y[n] - y[n - 1]) / (h[n - 1] * h[n - 1]));

    VectorXd c = solver.solve(r);

    VectorXd evalX((x.size() - 1) * N);
    VectorXd evalY((x.size() - 1) * N);

    for (int i = 0; i < x.size() - 1; i++) {
        double t1 = x[i];
        double t2 = x[i + 1];

        double y1 = y[i];
        double y2 = y[i + 1];

        double c1 = c[i];
        double c2 = c[i + 1];

        VectorXd xl = VectorXd::LinSpaced(N, t1, t2);
        VectorXd yl = hermloceval(xl, t1, t2, y1, y2, c1, c2);
        evalX.segment(i * N, N) = xl;
        evalY.segment(i * N, N) = yl;
    }

    plt::figure();
    plt::plot(toCpp(x), toCpp(y), {{"label", "data points"}, {"marker", "o"}, {"linestyle", ""}});
    plt::plot(toCpp(evalX), toCpp(evalY), {{"label", "hermite polynomial"}});
    plt::savefig("./interpolation.png");

    return 0;
}