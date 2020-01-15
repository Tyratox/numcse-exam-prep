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

template <typename Function>
void rungeKutta(
    Vector2d &y1,
    const MatrixXd &A, const VectorXd &b,
    Function &f, const Vector2d &y0, double h) {

    assert(A.cols() == A.rows());
    assert(b.size() == A.cols());

    int s = A.cols();
    int d = y0.size();

    vector<Vector2d> k(s);
    y1 = y0;

    for (int i = 0; i < s; i++) {

        Vector2d subsum = y0;
        for (int j = 0; j < i; j++) {
            subsum += h * A(i, j) * k[j];
        }
        Vector2d ki = f(subsum);
        k[i] = ki;
        y1 += h * b[i] * ki;
    }
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

int main() {

    Matrix3d A;
    A << 0, 0, 0,
        1.0 / 3.0, 0, 0,
        0, 2.0 / 3.0, 0;

    Vector3d b;
    b << 0.25, 0, 0.75;

    Vector2d y10;
    y10 << 0.319465882659820, 9.730809352326228;

    Vector2d y0;
    y0 << 100, 5;

    double T = 10;
    double alpha1 = 3;
    double alpha2 = 2;
    double beta1 = 0.1;
    double beta2 = 0.1;

    auto f = [&alpha1, &alpha2, &beta1, &beta2](Vector2d &x) {
        Vector2d y = x;
        y(0) *= alpha1 - beta1 * x(1);
        y(1) *= beta2 * x(0) - alpha2;

        return y;
    };

    vector<double> x = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    vector<double> err(x.size());

    plt::figure();

    for (int i = 0; i < x.size(); i++) {
        int N = x[i];
        double h = T / N;

        Vector2d y1;
        Vector2d y = y0;

        vector<double> xs(N + 1);
        vector<double> ys(N + 1);

        for (int i = 1; i <= N; i++) {
            rungeKutta(y1, A, b, f, y, h);
            y = y1;

            xs[i] = y(0);
            ys[i] = y(1);
        }

        plt::plot(xs, ys, {{"label", string("State space for N=") + to_string(N)}});

        double e = (y1 - y10).norm();
        cout << setw(16) << N << setw(16) << e << endl;
        err[i] = e;
    }

    plt::legend();
    plt::savefig("./state-spaces.png");

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("error");
    plt::loglog(x, err);
    plt::savefig("./error.png");

    return 0;
}