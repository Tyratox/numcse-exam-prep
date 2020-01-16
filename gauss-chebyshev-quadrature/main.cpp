#define WITHOUT_NUMPY 1

#include "../matplotlibcpp.h"
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

VectorXd getChebyshevNodes(double a, double b, int N) {
    VectorXd t = VectorXd::LinSpaced(N, 0, N - 1);
    int n = N - 1;

    t = (((((t.array() * 2 + 1) / (2 * (n + 1))) * M_PI).cos() + 1) * 0.5 * (b - a) + a).matrix();

    return t;
}

VectorXd evaluate(const VectorXd &x, const VectorXd &c) {
    int d = c.size() - 1;
    int s = c.size();

    VectorXd ones = VectorXd::Ones(x.size());

    VectorXd y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct(x) + c[i] * ones;
    }

    return y;
}

double evaluate(double x, const VectorXd &c) {
    int s = c.size();
    int d = s - 1;

    double y = c[d];

    for (int i = 1; i < s; i++) {
        y = y * x + c[s - i];
    }

    return y;
}

int main() {

    double a = -1;
    double b = 1;
    int N = 8; //>= polynomial degree / 2

    auto f = [](double x) {
        return pow(x, 8) + 7 * pow(x, 7) + 3 * pow(x, 6) + 2 * pow(x, 5) - 10 * pow(x, 4) - 3 * pow(x, 3) + x * x + 2 * x + 7;
    };

    //we're integrating \int_-1, 1: f(x)/sqrt(1-x*x)

    double solution = 635 * M_PI / 128;

    cout << "The exact solution is " << solution << endl;

    cout << setw(16) << "N" << setw(16) << "value" << setw(16) << "error" << endl;

    for (int i = 1; i <= N; i++) {
        VectorXd nodes = getChebyshevNodes(a, b, i);
        VectorXd weights = VectorXd::Constant(i, M_PI / i);

        double sum = 0;
        for (int j = 0; j < nodes.size(); j++) {
            sum += weights[j] * f(nodes[j]);
        }

        cout << setw(16) << i << setw(16) << sum << setw(16) << abs(sum - solution) << endl;
    }

    return 0;
}