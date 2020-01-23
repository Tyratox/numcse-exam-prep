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

Matrix2d Df(const Vector2d &z) {
    Matrix2d J;
    J << 0, 1, 2 + 6 * z[0] * z[0], 0;

    return J;
}

Matrix2d DF(const Vector2d &x, double h) {
    return h * Df(x) - Matrix2d::Identity();
}

Vector2d f(const Vector2d &z) {
    Vector2d zNew;
    zNew(0) = z[1];
    zNew(1) = 2 * z[0] * (1 + z[0] * z[0]);

    return zNew;
}

Vector2d F(const Vector2d &x, const Vector2d &zk, double h) {
    return zk + h * f(x) - x;
}

Vector2d newtonIteration(Vector2d x, double h, int n, double tol) {

    Vector2d zk = x; //freeze initial argument as it is needed for F
    Vector2d s;      //newton correction

    int i = 0;

    do {
        /*cout << "df" << endl
             << DF(x, h) << endl
             << "rhs" << endl
             << F(x, zk, h) << endl;*/

        x = x - DF(x, h).lu().solve(zk + h * f(x) - x);

        /*cout << "new norm " << x.norm() << endl
             << endl;*/

        /*cout << " x " << endl
             << x << endl
             << "with norm " << x.norm() << endl;*/
    } while (x.norm() > tol && ++i < n);

    return x;
}

Vector2d implicitEuler(Vector2d z0, double h, int N) {
    Vector2d z = z0;
    double t = 0;

    vector<double> x(N + 1);
    vector<double> y(N + 1);
    vector<double> yDerivative(N + 1);
    vector<double> ySolution(N + 1);
    vector<double> yDerivativeSolution(N + 1);
    x[0] = t;
    y[0] = z[0];
    yDerivative[0] = z[1];

    ySolution[0] = tan(t);
    yDerivativeSolution[0] = 1.0 / (cos(t) * cos(t));

    for (int i = 0; i < N; i++) {
        z = newtonIteration(z, h, 10, 1.0e-8);
        t += h;

        x[i + 1] = t;
        y[i + 1] = z[0];
        yDerivative[i + 1] = z[1];

        ySolution[i + 1] = tan(t);
        yDerivativeSolution[i + 1] = 1.0 / (cos(t) * cos(t));
    }

    plt::plot(x, y, {{"label", "implicit tan"}});
    plt::plot(x, yDerivative, {{"label", "implicit tan'"}});
    plt::plot(x, ySolution, {{"label", "tan(x)"}});
    plt::plot(x, yDerivativeSolution, {{"label", "tan'(x)"}});

    return z;
}

int main() {

    //ivp y'' = 2y * (1 + y^2) = 2y + 2y^3, y(0)=0, y'(0)=1

    double T = (M_PI / 2) - 0.1;
    double h = 0.01;

    plt::figure();
    plt::ylim(-1, 10);

    VectorXd s = implicitEuler(Vector2d(0, 1), h, T / h);

    cout << "solution" << endl
         << s << endl;

    plt::legend();
    plt::savefig("./plot.png");

    return 0;
}