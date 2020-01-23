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

VectorXd hermloceval(VectorXd t, double t1, double t2, double y1, double y2, double c1, double c2) {
    const double h = t2 - t1, a1 = y2 - y1, a2 = a1 - h * c1, a3 = h * c2 - a1 - a2;
    t = ((t.array() - t1) / h).matrix();
    return (y1 + (a1 + (a2 + a3 * t.array()) * (t.array() - 1)) * t.array()).matrix();
}

template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

int main() {

    int n = 11;
    VectorXd x(n);
    for (int i = 0; i < n; i++) {
        x[i] = -1 + 0.2 * i;
    }

    VectorXd y(n);
    for (int i = 0; i < n; i++) {
        y[i] = sin(5 * x[i]) * exp(x[i]);
    }

    double a = x.minCoeff();
    double b = x.maxCoeff();

    int N = 100;

    VectorXd h = x.tail(n - 1) - x.head(n - 1);                     //n-1 elements, n-2 is the max index. h[0] = t[1]-t[0]
    VectorXd dx = (y.tail(n - 1) - y.head(n - 1)).cwiseQuotient(h); //n-1 elements
    VectorXd c(n);
    c[0] = dx[0];
    for (int i = 1; i < n - 1; i++) {
        //c[i] = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1]) * dx[i - 1] + (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]) * dx[i];
        if (sgn(dx[i - 1]) == sgn(dx[i])) {
            //harmonic mean
            c[i] = 3 * (h[i] + h[i - 1]) / (((2 * h[i] + h[i - 1]) / (dx[i - 1])) + (2 * h[i - 1] + h[i]) / (dx[i]));
        } else {
            c[i] = 0; //prevent over- and underflow
        }
    }
    c[n - 1] = dx[dx.size() - 1];

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