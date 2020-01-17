#define WITHOUT_NUMPY 1

#include "../matplotlibcpp.h"
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

void evaluate(VectorXd &y, const VectorXd &c, const VectorXd &x) {
    int s = c.size();
    int d = s - 1;

    VectorXd ones = VectorXd::Ones(x.size());

    y = c[s - 1] * ones;

    for (int i = s - 2; i >= 0; i--) {
        y = y.cwiseProduct(x) + c[i] * ones;
    }
}

void polyfit(VectorXd &c, const VectorXd &x, const VectorXd &y, unsigned int d) {
    assert(x.size() == y.size());
    int n = x.size();

    MatrixXd V(n, d + 1); //Vandermonde
    V.col(0) = VectorXd::Ones(n);

    for (int i = 1; i <= d; i++) {
        V.col(i) = V.col(i - 1).cwiseProduct(x);
    }

    c = V.householderQr().solve(y);
}

vector<double> toCpp(VectorXd v) {
    return vector<double>(v.data(), v.data() + v.size());
}

MatrixXd bestApproximation(const MatrixXd &A, int k) {
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    return (svd.matrixU().leftCols(k)) * (svd.singularValues().head(k).asDiagonal()) * (svd.matrixV().leftCols(k).transpose());
}

MatrixXd getTrendVectors(const MatrixXd &A) {
    JacobiSVD<MatrixXd> svd(A, ComputeThinU);
    return svd.matrixU() * svd.singularValues();
}

int main() {

    VectorXd x(10);
    x << -1, -0.75, -0.4, -0.3, -0.15, 0.18, 0.3, 0.6, 0.8, 1;

    VectorXd y(10);
    y << 0, -0.5, -1.0, -1.1, -1.1, -1.0, -0.5, 0.1, 1.0, 2.0;

    VectorXd y2(10);
    y2 << -1, -1, -2.0, -1.5, -2.1, -1.5, 0.5, 0.7, 1.1, 2.2;

    //center data around 0
    /*B.row(0) = (B.row(0).array() - (B.row(0).mean())).matrix();
    B.row(1) = (B.row(1).array() - (B.row(1).mean())).matrix();*/

    plt::figure();
    plt::plot(toCpp(x), toCpp(y), {{"label", "data points"}, {"marker", "o"}, {"linestyle", ""}});

    VectorXd c;
    polyfit(c, x, y, 2);

    VectorXd samplingX = VectorXd::LinSpaced(1000, x.minCoeff(), x.maxCoeff());
    VectorXd samplingY;
    evaluate(samplingY, c, samplingX);

    plt::plot(toCpp(samplingX), toCpp(samplingY), {{"label", "interpolated polynomial"}});

    MatrixXd B(10, 2);
    B << x, y;

    MatrixXd B1 = getTrendVectors(B);
    for (int i = 0; i < B1.cols(); i++) {
        plt::plot(toCpp(x), toCpp(B1.col(i)), {{"label", "principal component"}});
    }

    plt::legend();
    plt::savefig("./curve-fitting.png");

    return 0;
}