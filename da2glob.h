#ifndef DA2GLOB_H
#define DA2GLOB_H

#include <tuple>
#include <vector>
#include <complex>
#include <functional>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


tuple<double,int> da2glob(function<double(double)> f, const VectorXd& a, double tolabs = 0, double tolrel = 0.001, bool step_info = false, bool trace=false);



#endif
