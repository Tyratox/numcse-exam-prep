#define WITHOUT_NUMPY 1

#include "matplotlibcpp.h"
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

void randomArrowMatrix(MatrixXd &A, int n, int m) {
    A = MatrixXd::Zero(n, m);
    A.diagonal() = VectorXd::Random(m);
    A.row(n - 1) = VectorXd::Random(m);
    A.col(m - 1) = VectorXd::Random(n);
}

void solveNormalEquations(VectorXd &s, const MatrixXd &A, const VectorXd &b) {
    s = (A.transpose() * A).lu().solve(A.transpose() * b);
}

void solveSparse(VectorXd &s, SparseMatrix<double> &A, const VectorXd &b) {
    int n = A.rows();
    int m = A.cols();

    SparseMatrix<double> C(m + n, m + n);
    int nnz = A.nonZeros();

    /*if (nnz != (A.array() != 0).count()) {
        cout << n << " x " << m << " matrix contains "
             << (A.array() != 0).count() << " nnz elements instead of " << nnz
             << endl;
        cout << A;
    }*/

    vector<Triplet<double>> triplets;
    triplets.reserve(n + 2 * nnz);

    for (int i = 0; i < n; i++) {
        triplets.push_back(Triplet<double>(i, i, -1));
    }

    assert(triplets.size() == n);

    // column major
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            /*cout << it.row() << ", " << (n + it.col()) << " and "
                 << (n + it.col()) << ", " << it.row() << " inside a " << n
                 << " x " << m << " matrix" << endl;*/
            triplets.push_back(
                Triplet<double>(it.row(), n + it.col(), it.value()));
            triplets.push_back(
                Triplet<double>(n + it.col(), it.row(), it.value()));
        }
    }

    assert(triplets.size() == n + 2 * nnz);

    C.setFromTriplets(triplets.begin(), triplets.end());

    SparseLU<SparseMatrix<double>> solver;
    solver.analyzePattern(C);
    solver.factorize(C);
    VectorXd c = VectorXd(n + m);
    c << b, VectorXd::Zero(n);

    VectorXd d = solver.solve(c);
    s = d.tail(n);
}

void constructDensePoissonMatrix(MatrixXd &A, int n) {
    A = MatrixXd::Zero(n, n);
    A.diagonal() = VectorXd::Constant(A.cols(), -2);
    A.diagonal(1) = VectorXd::Constant(A.cols() - 1, 1);
    A.diagonal(-1) = VectorXd::Constant(A.cols() - 1, 1);
}

void constructSparsePoissonMatrix(SparseMatrix<double> &A, int n) {
    A = SparseMatrix<double>(n, n);
    int nnz = n + 2 * (n - 1);

    vector<Triplet<double>> t;
    t.reserve(nnz);

    for (int i = 0; i < n; i++) {
        t.push_back(Triplet<double>(i, i, -2));
        if (i > 0) {
            t.push_back(Triplet<double>(i, i - 1, 1));
        }
        if (i < n - 1) {
            t.push_back(Triplet<double>(i, i + 1, 1));
        }
    }
    A.setFromTriplets(t.begin(), t.end());
}

int main() {
    MatrixXd A;
    SparseMatrix<double> B;
    VectorXd b, s1, s2;

    cout << setw(16) << "Nodes" << setw(16) << "norm equations" << setw(16)
         << "sparse solver" << setw(16) << "difference" << endl;

    auto f = [](double d) { return M_PI * M_PI * sin(M_PI * d); };

    vector<double> x(30);
    vector<double> diff(30);
    vector<double> norm(30);
    vector<double> sparse(30);

    for (int i = 0; i < 30; i++) {
        int N = i * 5 + 5;
        double dx = 1.0 / (1 + N);

        x[i] = N;

        constructDensePoissonMatrix(A, N + 1);
        constructSparsePoissonMatrix(B, N + 1);

        A = A / (dx * dx);
        B = B / (dx * dx);

        b = VectorXd(N + 1);
        for (int i = 0; i < N + 1; i++) {
            b(i) = f(dx * i);
        }

        auto t1 = chrono::high_resolution_clock::now();
        solveNormalEquations(s1, A, b);
        auto t2 = chrono::high_resolution_clock::now();
        solveSparse(s2, B, b);
        auto t3 = chrono::high_resolution_clock::now();

        auto duration1 =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count();
        auto duration2 =
            std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2)
                .count();

        double d = (s1 - s2).norm();

        norm[i] = duration1;
        sparse[i] = duration2;
        diff[i] = d;

        cout << setw(16) << N << setw(16) << duration1 << setw(16) << duration2
             << setw(16) << d << endl;
    }

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("time");
    plt::named_plot("Normal equations", x, norm);
    plt::named_plot("Extended normal equations", x, sparse);
    plt::legend();
    plt::save("./sparse.png");

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("difference");
    plt::semilogy(x, diff);
    plt::save("./difference.png");

    return 0;
}