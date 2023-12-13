#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <tuple>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <complex>


using namespace std;
using namespace Eigen;



tuple<double, double, double, double> quadrature(double a, double b, VectorXd y, double D, bool& interval);

tuple<double, double, double, double> cotes5(double a, double b, VectorXd y, double D, bool& interval);

tuple<double, double, double, double> cotes9(double a, double b, VectorXd y, double D, bool& interval);

tuple<double, double, double, double> cotes17(double a, double b, VectorXd y, double D, bool& interval);

tuple<double, double, double, double> rule33(double a, double b, VectorXd y, double D,  bool& interval);

#endif
