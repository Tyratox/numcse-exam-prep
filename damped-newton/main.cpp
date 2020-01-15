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
VectorXd newtonSolve(Function f, Jacobian J, VectorXd x) {

    VectorXd xn;
    double lambda = 1.0;
    double lmin = 1E-3;
    double atol = numeric_limits<double>::epsilon();
    double rtol = numeric_limits<double>::epsilon();

    VectorXd correction;
    double correctionNorm, lambdaCorrectionNorm;

    do {
        auto jacobianFactored = J(x).lu();
        correction = jacobianFactored.solve(f(x));
        correctionNorm = correction.norm();

        lambda *= 2;
        do {
            lambda /= 2;
            if (lambda < lmin) {
                cerr << "lambda approaches zero" << endl;
                throw "lambda approaches zero";
            }

            xn = x - lambda * correction;
            lambdaCorrectionNorm = jacobianFactored.solve(f(xn)).norm();

        } while (lambdaCorrectionNorm > (1 - lambda / 2) * correctionNorm);

        x = xn;
        lambda = std::min(2.0 * lambda, 1.0);
    } while (lambdaCorrectionNorm > atol && lambdaCorrectionNorm > rtol * x.norm());

    return x;
}

int main() {

    auto f = [](double x) {
        return atan(x);
    };

    auto fPrime = [](double x) {
        return 1 / (1 + x * x);
    };

    double x = 20, xn, correction = 0, y = f(x);
    double lambda = 1.0;
    double lmin = 1E-3;
    double atol = numeric_limits<double>::epsilon();
    double rtol = numeric_limits<double>::epsilon();

    vector<double> point;
    vector<double> newtonValue;
    vector<double> lambdas;
    vector<double> lambdaX;
    vector<double> lambdaY;

    plt::figure();
    plt::ylim(-10, 10);
    plt::xlabel("iteration");
    plt::ylabel("x");

    int i = 0;

    point.push_back(x);
    newtonValue.push_back(y);

    double sn, stn;

    auto t1 = chrono::high_resolution_clock::now();
    do {
        correction = f(x) / fPrime(x);
        sn = abs(correction);

        lambda *= 2;
        do {
            lambda /= 2;
            if (lambda < lmin) {
                cerr << "lambda -> 0" << endl;
                throw 2;
            }

            xn = x - lambda * correction;
            stn = abs(f(xn) / fPrime(xn));
        } while (stn > (1 - lambda / 2) * sn);

        lambdaY.push_back(y);
        lambdaX.push_back(x);
        lambdas.push_back(lambda);
        x = xn;

        lambda = std::min(2.0 * lambda, 1.0);
        y = f(x);

        point.push_back(x);
        newtonValue.push_back(abs(y));
    } while (stn > atol && stn > rtol * abs(x));
    auto t2 = chrono::high_resolution_clock::now();

    cout << "Found zero of f at x = " << x << " after " << point.size() << " newton iterations in " << (chrono::duration_cast<chrono::microseconds>(t2 - t1).count()) << " microseconds" << endl;
    plt::plot(point, {{"label", "newton iteration"}, {"marker", "o"}, {"linestyle", ""}});

    plt::legend();
    plt::savefig("./fp-iteration.png");

    plt::figure();
    plt::xlabel("iteration");
    plt::ylabel("error");
    plt::semilogy(newtonValue, {{"label", "newton iteration"}, {"color", "r"}});
    plt::legend();
    plt::savefig("./error.png");

    int N = 1000;
    double h = 40.0 / N;
    vector<double> actanXSampling(N), actanYSampling(N);
    for (int i = 0; i < N; i++) {
        x = -20 + h * i;
        y = f(x);

        actanXSampling[i] = x;
        actanYSampling[i] = y;
    }

    plt::figure();
    plt::xlabel("iteration");
    plt::ylabel("lambda");
    plt::plot(actanXSampling, actanYSampling, {{"label", "$tan^{-1}(x)$"}, {"color", "b"}});
    plt::plot(lambdaX, lambdaY, {{"label", "arctan at evaluated x"}, {"color", "g"}});
    plt::plot(lambdaX, lambdas, {{"label", "chosen lambda for given x"}, {"color", "r"}});
    plt::legend();
    plt::savefig("./lambda.png");

    return 0;
}