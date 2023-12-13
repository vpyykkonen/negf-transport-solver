#ifndef PADE_FREQUENCIES_H
#define PADE_FREQUENCIES_H

#include <tuple>
#include <Eigen/Dense>
#include <string>

using namespace std;
using namespace Eigen;

tuple<double,double> pade_frequency(int n, int n_approx);

void save_data(string path, const VectorXd& freqs, const VectorXd& residues);

tuple<bool, VectorXd,VectorXd> load_data(string base_path, int n_approx);

#endif
