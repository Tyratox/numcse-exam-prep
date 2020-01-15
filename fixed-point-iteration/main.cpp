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

template <typename Function, typename Derivative>
double newtonCorrection(Function f, Derivative fPrime, double x0) {
    return f(x0) / fPrime(x0);
}

template <typename Function>
double secantCorrection(Function f, double x0, double x1) {
    double fx1 = f(x1);
    return fx1 * (x1 - x0) / (fx1 - f(x0));
}

int main() {

    auto f = [](double x) {
        return x * exp(x) - 1;
    };

    auto fPrime = [](double x) {
        return exp(x) + x * exp(x);
    };

    double x0 = 0, x1 = 5, x = 2.5, correction = 0, y = f(x);
    double atol = numeric_limits<double>::epsilon();
    double rtol = numeric_limits<double>::epsilon();

    vector<double> point;
    vector<double> newtonValue;
    vector<double> secantValue;

    plt::figure();
    plt::ylim(-10, 10);
    plt::xlabel("iteration");
    plt::ylabel("x");

    int i = 0;

    point.push_back(x);
    newtonValue.push_back(y);

    auto t1 = chrono::high_resolution_clock::now();
    do {
        correction = f(x) / fPrime(x);
        x = x - correction;
        y = f(x);

        point.push_back(x);
        newtonValue.push_back(abs(y));
    } while (abs(correction) > atol && abs(correction) > rtol * abs(x));
    auto t2 = chrono::high_resolution_clock::now();

    cout << "Found zero of f at x = " << x << " after " << point.size() << " newton iterations in " << (chrono::duration_cast<chrono::microseconds>(t2 - t1).count()) << " microseconds" << endl;
    plt::plot(point, {{"label", "newton iteration"}, {"marker", "o"}, {"linestyle", ""}});

    //secant method

    x = x1;
    point.clear();

    double fx0 = f(x0);
    double fx1;

    point.push_back(x);
    secantValue.push_back(fx0);

    t1 = chrono::high_resolution_clock::now();
    do {
        fx1 = f(x);
        correction = fx1 * (x - x0) / (fx1 - fx0);
        x0 = x;
        fx0 = fx1;

        x = x - correction;

        point.push_back(x);
        secantValue.push_back(abs(fx1));
    } while (abs(correction) > atol && abs(correction) > rtol * abs(x));
    t2 = chrono::high_resolution_clock::now();

    cout << "Found zero of f at x = " << x << " after " << point.size() << " secant iterations in " << (chrono::duration_cast<chrono::microseconds>(t2 - t1).count()) << " microseconds" << endl;

    plt::plot(point, {{"label", "secant method"}, {"marker", "o"}, {"linestyle", ""}});

    plt::legend();
    plt::savefig("./fp-iteration.png");

    plt::figure();
    plt::xlabel("iteration");
    plt::ylabel("error");
    plt::semilogy(newtonValue, {{"label", "newton iteration"}, {"color", "r"}});
    plt::semilogy(secantValue, {{"label", "secant method"}, {"color", "g"}});
    plt::legend();
    plt::savefig("./error.png");

    return 0;
}