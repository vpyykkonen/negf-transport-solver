#ifndef SCF_SOLVER_H
#define SCF_SOLVER_H

#include <string>
#include <vector>
#include <Eigen/Dense>

#include "ScfMethod.h"

using namespace std;
using namespace Eigen;

typedef complex<double> dcomp;

class ScfSolver
{
    private:
    VectorXcd* X;
    vector<ScfMethod*> methods;
    string save_path;
    int iterations;
    bool converged;

    public:
    ScfSolver(VectorXcd* X, vector<ScfMethod*>& methods, const string save_path);
    ScfSolver(VectorXcd* X, const string scf_cfg_path, const string save_path);
    ~ScfSolver();
    void iterate(function<VectorXcd(const VectorXcd&)> FX);

    void set_X(VectorXcd* X){this->X = X;}
    VectorXcd* get_X(){return this->X;}
    int get_iterations(){return this->iterations;}
    bool get_converged(){return this->converged;}

    void add_method(ScfMethod* method){methods.push_back(method);}
    void remove_method(int idx){methods.erase(methods.begin()+idx);}
    void reset_iterations(){iterations=0;}
};


#endif
