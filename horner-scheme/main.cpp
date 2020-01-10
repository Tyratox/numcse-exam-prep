#define WITHOUT_NUMPY 1

#include "matplotlibcpp.h"
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

/**
 * Evaluates a polynomial of degree c.size()-1 with coefficients c at all positions x and
 * stores the results in y. c[0] contains the coefficient of maximal degree
*/
void horner(VectorXd &y, const VectorXd &x, const VectorXd &c) {
    int d = c.size() - 1;
    int s = c.size();

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[0] * ones;

    for (int i = 1; i < s; i++) {
        y = y.cwiseProduct(x) + c[i] * ones;
    }
}

void evaluate(VectorXd &y, const VectorXd &x, const VectorXd &c) {
    int s = c.size();
    int d = s - 1;

    y = VectorXd::Zero(x.size());

    for (int i = 0; i < s; i++) {
        y += c[i] * x.array().pow(d - i).matrix();
    }
}

void interpolate(VectorXd &c, const VectorXd &x, const VectorXd &y) {
    assert(x.size() == y.size());
    int n = y.size();

    MatrixXd V(n, n); //vandermonde matrix
    V.col(0) = VectorXd::Ones(n);
    for (int i = 1; i < n; i++) {
        V.col(i) = V.col(i - 1).cwiseProduct(x);
    }
    c = V.lu().solve(y).reverse();
}

vector<double> toCpp(VectorXd v) {
    vector<double> vec(v.data(), v.data() + v.size());
    return vec;
}

int main() {

    int max = 100;
    double d = 1.0 / max;

    vector<double> xAxis(max);
    vector<double> normal(max);
    vector<double> hornerScheme(max);

    auto f = [] (double x){
        return 3*pow(sin(x), 3) + 2*x - 5*sin(x);
    };

    for (int i = 0; i < max; i++) {
        int N = i + 2;
        xAxis[i] = N;

        VectorXd x = VectorXd::LinSpaced(N, 0, 1);
        VectorXd xSampling = VectorXd::LinSpaced(N*1000, 0, 1);
        VectorXd y(N);
        for(int j=0;j<N;j++){
            y(j) = f(x(j));
        }

        VectorXd c;
        interpolate(c, x, y);

        VectorXd res1, res2;

        auto t1 = chrono::high_resolution_clock::now();
        evaluate(res1, xSampling, c);
        auto t2 = chrono::high_resolution_clock::now();
        horner(res2, xSampling, c);
        auto t3 = chrono::high_resolution_clock::now();

        if (i % 10 == 0) {
            //something's wrong?
            plt::figure();
            plt::xlabel("x");
            plt::ylabel("y");

            plt::named_plot("exact", toCpp(x), toCpp(y), "ro");
            plt::named_plot("normal evaluation", toCpp(xSampling), toCpp(res1), "g");
            plt::named_plot("horner scheme", toCpp(xSampling), toCpp(res2), "b");
            plt::legend();
            plt::save(string("./error-") + to_string(N) + string(".png"));
        }

        assert(res1.size() == res2.size());
        assert((res1-res2).norm() < 1E-7);

        normal[i] = chrono::duration_cast<chrono::microseconds>((t2 - t1)).count();
        hornerScheme[i] = chrono::duration_cast<chrono::microseconds>((t3 - t2)).count();
    }

    plt::figure();
    plt::xlabel("N");
    plt::ylabel("time");

    plt::named_plot("Normal evaluation", xAxis, normal, "r");
    plt::named_plot("Horner scheme", xAxis, hornerScheme, "g");
    plt::legend();
    plt::save("./horner.png");

    return 0;
}