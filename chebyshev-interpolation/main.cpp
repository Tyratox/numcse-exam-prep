#define WITHOUT_NUMPY 1

#include "../matplotlibcpp.h"
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

VectorXd clenshaw(const VectorXd &a, const VectorXd &x) {
    const int d = a.size() - 1;  //polynomial degree
    MatrixXd A(d + 1, x.size()); //intermediate values

    //initialize matrix
    for (int i = 0; i < A.cols(); i++) {
        A.col(i) = a;
    }

    for (int i = d - 1; i > 0; i--) {
        A.row(i) += 2 * x.transpose().cwiseProduct(A.row(i + 1));
        A.row(i - 1) -= A.row(i + 1);
    }

    return A.row(0) + x.transpose().cwiseProduct(A.row(1));
}

VectorXd getChebyshevNodes(double a, double b, int N) {
    VectorXd t = VectorXd::LinSpaced(N, 0, N - 1);
    int n = N - 1;

    t = (((((t.array() * 2 + 1) / (2 * (n + 1))) * M_PI).cos() + 1) * 0.5 * (b - a) + a).matrix();

    return t;
}

void evaluate(VectorXd &y, const VectorXd &x, const VectorXd &c, const VectorXd &t) {
    assert(c.size() <= t.size());

    int s = c.size();
    int d = s - 1;

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct((x.array() - t[d - i]).matrix()) + c[i] * ones;
    }
}

void interpolate(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    MatrixXd A(n, n);
    A.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < n; i++) {
        A.col(i) = A.col(i - 1).cwiseProduct((x.array() - x[i - 1]).matrix());
    }

    c = A.template triangularView<Lower>().solve(y).reverse();
}

vector<double> toCpp(VectorXd v) {
    return vector<double>(v.data(), v.data() + v.size());
}

int main() {

    //runge's example
    auto f = [](double x) {
        return 1.0 / (1 + x * x);
    };

    double a = -1;
    double b = 1;
    int N = 20;

    plt::figure();
    plt::figure_size(1200, 780);
    plt::xlabel("x");
    plt::ylabel("N");
    plt::xlim(-1.2, 1.2);
    plt::ylim(-1, 21);

    for (int i = 1; i <= N; i++) {
        VectorXd nodes = getChebyshevNodes(a, b, i);
        VectorXd y = VectorXd::Constant(i, i);

        plt::plot(toCpp(nodes), toCpp(y), {{"marker", "*"}, {"color", "r"}, {"linestyle", ""}});
    }

    plt::savefig("./chebyshev-nodes.png");

    plt::figure();

    a = -5;
    b = 5;
    N = 11;

    VectorXd xSampling = VectorXd::LinSpaced(1000, a, b);
    VectorXd ySampling(xSampling.size());
    for (int i = 0; i < ySampling.size(); i++) {
        ySampling[i] = f(xSampling[i]);
    }

    plt::plot(toCpp(xSampling), toCpp(ySampling), {{"label", "$1/(1+x^2)$"}});

    //interpolate with N equidistant nodes

    VectorXd x = VectorXd::LinSpaced(N, a, b);
    VectorXd y(x.size());
    for (int i = 0; i < y.size(); i++) {
        y[i] = f(x[i]);
    }

    VectorXd c;
    interpolate(c, x, y);
    evaluate(ySampling, xSampling, c, x);

    plt::plot(toCpp(xSampling), toCpp(ySampling), {{"label", "equidistant nodes"}});

    x = getChebyshevNodes(a, b, N);
    for (int i = 0; i < y.size(); i++) {
        y[i] = f(x[i]);
    }

    interpolate(c, x, y);
    evaluate(ySampling, xSampling, c, x);

    plt::plot(toCpp(xSampling), toCpp(ySampling), {{"label", "chebyshev nodes"}});

    plt::legend();
    plt::savefig("./interpolation.png");

    return 0;
}