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

template <typename Function, typename Jacobian>
VectorXd newtonCorrection(Function f, Jacobian fPrime, VectorXd x0) {
    return fPrime(x0).lu().solve(f(x0));
}

int main() {

    auto f = [](Vector2d x) {
        Vector2d y;
        const double x1 = x(0), x2 = x(1);
        y << x1 * x1 - 2 * x1 - x2 + 1, x1 * x1 + x2 * x2 - 1;

        return y;
    };

    auto fPrime = [](Vector2d x) {
        Matrix2d J;
        const double x1 = x(0), x2 = x(1);
        J << 2 * x1 - 2, -1, 2 * x1, 2 * x2;

        return J;
    };

    Vector2d x(2, 3), y = f(x), correction;

    double atol = numeric_limits<double>::epsilon();
    double rtol = numeric_limits<double>::epsilon();

    vector<double> newtonValue;
    vector<double> simplifiedNewtonValue;

    plt::figure();
    plt::ylim(-10, 10);
    plt::xlabel("iteration");
    plt::ylabel("x");

    int i = 0;

    newtonValue.push_back(y.norm());

    auto t1 = chrono::high_resolution_clock::now();
    do {
        correction = fPrime(x).lu().solve(f(x));
        x = x - correction;
        y = f(x);

        newtonValue.push_back(y.norm());
    } while (correction.norm() > atol && correction.norm() > rtol * x.norm());
    auto t2 = chrono::high_resolution_clock::now();

    cout << "Found zero of f at x = " << x << " after " << newtonValue.size() << " newton iterations in " << (chrono::duration_cast<chrono::microseconds>(t2 - t1).count()) << " microseconds" << endl;

    x = Vector2d(2, 3);
    y = f(x);
    simplifiedNewtonValue.push_back(y.norm());

    auto lu = fPrime(x).lu();

    t1 = chrono::high_resolution_clock::now();
    do {
        correction = lu.solve(f(x));
        x = x - correction;
        y = f(x);

        simplifiedNewtonValue.push_back(y.norm());
    } while (correction.norm() > atol && correction.norm() > rtol * x.norm());
    t2 = chrono::high_resolution_clock::now();

    cout << "Found zero of f at x = " << x << " after " << simplifiedNewtonValue.size() << " simplified newton iterations in " << (chrono::duration_cast<chrono::microseconds>(t2 - t1).count()) << " microseconds" << endl;

    plt::figure();
    plt::xlabel("iteration");
    plt::ylabel("error");
    plt::semilogy(newtonValue, {{"label", "newton iteration"}, {"color", "r"}});
    plt::semilogy(simplifiedNewtonValue, {{"label", "simplified newton"}, {"color", "g"}});
    plt::legend();
    plt::savefig("./error-multi.png");

    return 0;
}