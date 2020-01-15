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

template <typename Function, typename State>
State explicitEuler(const Function &f, const State &y0, const double &h) {
    return y0 + h * f(y0);
}

int main() {

    auto f = [](double x) {
        return x;
    };

    auto solution = [](double x) {
        return exp(x);
    };

    int samples = 20;
    double M = 10;
    int solN = 1000;

    double h = (double)M / solN;
    vector<double> t(solN);
    vector<double> v(solN);

    vector<double> errorX(samples);
    vector<double> error(samples);

    double x = 0, y, e;

    for (int i = 0; i < solN; i++) {
        t[i] = x;
        v[i] = solution(x);
        x += h;
    }

    plt::figure();
    plt::plot(t, v, {{"label", "$e^x$"}});

    for (int i = 0; i < samples; i++) {
        int N = i * 10 + 10;

        x = 0;
        y = 1;
        e = 0;

        h = M / (double)N;

        vector<double> t(N + 1);
        vector<double> v(N + 1);

        t[0] = x;
        v[0] = y;

        for (int j = 1; j <= N; j++) {
            y = explicitEuler(f, y, h);
            x += h;
            t[j] = x;
            v[j] = y;

            e = max(abs(y - solution(x)), e);
        }

        errorX[i] = N;
        error[i] = e;
        plt::plot(t, v, {{"label", "eplicit euler"}, {"linestyle", "--"}});
    }

    plt::savefig("./explicit-euler.png");

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("error");
    plt::semilogy(errorX, error);
    plt::savefig("./error");

    return 0;
}