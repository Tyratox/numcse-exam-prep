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

void explicitEuler(MatrixXd &Y, const MatrixXd &A, const MatrixXd &Y0, const double &h) {
    //y' = f(y) i.e.
    //Y1 = Y0 + h * f(y0)

    Y = Y0 + A * Y0;
}

void implicitEuler(MatrixXd &Y, const MatrixXd &A, const MatrixXd &Y0, const double &h) {
    //y' = f(y) i.e.
    //Y1 = Y0 + h * f(y1) <=> Y1 = Y0 + h * Ay1 <=> Y1 - h * A * Y1 = Y0 <=> (I - h*A)Y1 = Y0
    int n = A.rows();
    int m = A.cols();

    Y = (MatrixXd::Identity(n, m) - h * A).lu().solve(Y0);
}

void midpoint(MatrixXd &Y, const MatrixXd &A, const MatrixXd &Y0, const double &h) {
    //y' = f(y) i.e.
    //Y1 = Y0 + h * f(1/2 y0 + 1/2y1) <=> Y1 = Y0 + h * 1/2A(Y0 + Y1) <=> Y1 - 1/2 * h * A * Y1 = Y0 + 1/2 * h * A * Y0
    int n = A.rows();
    int m = A.cols();

    Y = (MatrixXd::Identity(n, m) - 1 / 2 * h * A).lu().solve(Y0 + 1 / 2 * h * A * Y0);
}

template <typename Method>
MatrixXd testMethod(const string &name, const Method &m, MatrixXd Y, const MatrixXd &A, const int &N, const double &h, const MatrixXd &I) {
    cout << "Running 20 timesteps with method \"" << name << "\"" << endl;
    cout << setw(8) << "N" << setw(32) << "(Y^T * Y - I).norm()" << endl;

    vector<double> error(N);

    for (int i = 0; i < N; i++) {
        m(Y, A, Y, h);
        double e = (Y.transpose() * Y - I).norm();
        cout << setw(8) << i << setw(32) << e << endl;

        error[i] = e;
    }

    plt::semilogy(error, {{"label", name}});

    return Y;
}

int main() {

    int n = 3;

    MatrixXd M(n, n);
    M << 8, 1, 6,
        3, 5, 7,
        9, 9, 2;

    //A is skew symmetric, i.e. A = -A^T
    MatrixXd A(n, n);
    A << 0, 1, 1,
        -1, 0, 1,
        -1, -1, 0;

    MatrixXd I = MatrixXd::Identity(n, n);

    assert((A + A.transpose()).norm() <= A.norm() * numeric_limits<double>::epsilon());

    MatrixXd Y0 = M.householderQr().householderQ() * I;

    assert((Y0 * Y0.transpose() - I).norm() <= M.norm() * numeric_limits<double>::epsilon());

    double N = 20;
    double h = 0.01;

    plt::figure();
    plt::title("Loss of orthogonality");

    MatrixXd M_explicit = testMethod("Explicit euler", explicitEuler, Y0, A, N, h, I);
    MatrixXd M_implicit = testMethod("Implicit euler", implicitEuler, Y0, A, N, h, I);
    MatrixXd M_midpoint = testMethod("Midpoint method", midpoint, Y0, A, N, h, I);

    plt::legend();
    plt::savefig("./error.png");

    return 0;
}