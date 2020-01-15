#define WITHOUT_NUMPY 1

#include "../matplotlibcpp.h"
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

int main() {

    auto f = [](double d) {
        return exp(d);
    };

    double h = 0.1;
    double x = 0;

    cout << setw(25) << "h" << setw(25) << "a" << setw(25) << "error" << std::endl;
    cout.precision(15);

    vector<double> xValues(15);
    vector<double> yValues(15);

    for (int i = 0; i < 16; i++) {
        double a = (f(x + h) - f(x)) / h;
        double error = abs(a - 1);

        cout << setw(25) << h << setw(25) << a << setw(25) << error << std::endl;

        xValues.push_back(h);
        yValues.push_back(error);

        h /= 10;
    }

    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);

    // Plot line from given x and y data. Color is selected automatically.
    plt::loglog(xValues, yValues);
    // Add graph title
    plt::title("Cancellation error for difference quotient");
    // Enable legend.
    plt::legend();
    // Save the image (file format is determined by the extension)
    plt::savefig("./difference_quotient.png");

    return 0;
}