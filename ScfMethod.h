#ifndef SCF_METHOD_H
#define SCF_METHOD_H

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <functional>
#include <string>

using namespace std;
using namespace Eigen;

typedef complex<double> dcomp;

class ScfMethod
{
    protected:
    VectorXcd* X;
    int iter;
    int max_iter;
    double error_abs;
    double error_rel;
    double tol_abs;
    double tol_rel;
    bool converged;
    string save_path;
    int saving_frequency;
    
    public:
    virtual void iterate(function<VectorXcd(const VectorXcd&)> FX ) = 0;

    ScfMethod(){X = NULL; iter = 0; max_iter = 0; error_abs = 1; error_rel = 1; tol_abs = 0.01; tol_rel = 0.01; converged = true; save_path = ""; saving_frequency = 0;}
    ScfMethod(VectorXcd* x, int it, int m_it, double tol_abs, double tol_rel, string save_path, int saving_frequency)
        : X{x}, iter{it}, max_iter{m_it}, tol_abs{tol_abs}, tol_rel{tol_rel}, save_path{save_path}, saving_frequency{saving_frequency}
    {error_abs = 1.0; error_rel = 1.0; converged = false;}
    ScfMethod(const ScfMethod& s){X = s.X; iter = s.iter; max_iter = s.max_iter; error_abs = s.error_abs; error_rel = s.error_rel; tol_abs = s.tol_abs; tol_rel = s.tol_rel; converged = s.converged;save_path = s.save_path;}
    virtual ~ScfMethod(){}
    
    VectorXcd* get_X(){return X;};
    int get_iter(){return iter;}
    int get_max_iter(){return max_iter;}
    double get_error_abs(){return error_abs;}
    double get_error_rel(){return error_rel;}
    double get_tol_abs(){return tol_abs;}
    double get_tol_rel(){return tol_rel;}
    string get_save_path(){return save_path;}
    int get_saving_frequency(){return saving_frequency;}
    bool get_converged(){return converged;}

    void set_X(VectorXcd* X){this->X = X;}
    void set_iter(int iter){this->iter = iter;}
    void set_max_iter(int max_iter){this->max_iter = max_iter;}
    void set_tol_abs(double tol_abs){this->tol_abs = tol_abs;}
    void set_tol_rel(double tol_rel){this->tol_rel = tol_rel;}
    void set_save_path(string save_path){this->save_path = save_path;}
    void set_saving_frequency(int saving_frequency){this->saving_frequency = saving_frequency;}
    
    void save_iteration(const string path);
};

class Mixing : public ScfMethod
{
    private:
    double alpha; // mixing parameter

    public:
    //Mixing():ScfMethod(){ X = VectorXcd::Ones(1); alpha = 1.0; iter = 0; max_iter = 0; error = 1.0; tol = 1.0; converged = true;}
    Mixing():ScfMethod{}{alpha = 1.0;}
    Mixing(VectorXcd* X, double alph, int iter, int max_iter, double tol_abs, double tol_rel, string save_path, int saving_frequency)
        :ScfMethod{X,iter,max_iter,tol_abs,tol_rel,save_path,saving_frequency}, alpha{alph} {}
    Mixing(const Mixing& b): ScfMethod(b) {alpha=b.alpha;}
    ~Mixing(){}
    void iterate(function<VectorXcd(const VectorXcd&)> FX );

    void set_alpha(double alpha){this->alpha = alpha;};
    double get_alpha(){return alpha;}
};

class AdaptiveMixing : public ScfMethod
{
    private:
    double alpha; // mixing parameter
    double max_alpha; // upper bound for alpha
    double min_alpha; // lower bound for alpha

    public:
    //Mixing():ScfMethod(){ X = VectorXcd::Ones(1); alpha = 1.0; iter = 0; max_iter = 0; error = 1.0; tol = 1.0; converged = true;}
    AdaptiveMixing():ScfMethod{}{alpha = 1.0; max_alpha = 1.0; min_alpha = 1.0;}
    AdaptiveMixing(VectorXcd* X, double alph, double max_alph, double min_alph, int iter, int max_iter, double tol_abs, double tol_rel,string save_path, int saving_frequency)
        :ScfMethod{X,iter,max_iter,tol_abs,tol_rel,save_path,saving_frequency}, alpha{alph},max_alpha{max_alph},min_alpha{min_alph} {}
    AdaptiveMixing(const AdaptiveMixing& b): ScfMethod(b) {alpha=b.alpha;max_alpha = b.max_alpha; min_alpha = b.min_alpha;}
    ~AdaptiveMixing(){}
    void iterate(function<VectorXcd(const VectorXcd&)> FX );
};

class BroydenGood : public ScfMethod
{
    private:
    MatrixXcd* inv_Jacobian;

    public:
    BroydenGood():ScfMethod{}{ inv_Jacobian = NULL;}
    BroydenGood(VectorXcd* X, MatrixXcd* Jinv, int iter, int max_iter, double tol_abs, double tol_rel,string save_path, int saving_frequency)
        :ScfMethod{X,iter,max_iter,tol_abs,tol_rel,save_path,saving_frequency}, inv_Jacobian{Jinv} {}
    BroydenGood(const BroydenGood& b):ScfMethod(b){inv_Jacobian = b.inv_Jacobian;}
    ~BroydenGood(){}

    void iterate(function<VectorXcd(const VectorXcd&)> FX );
    void set_inv_Jacobian(MatrixXcd* Jinv){inv_Jacobian = Jinv;}
};


class BroydenBad : public ScfMethod
{
    private:
    MatrixXcd* inv_Jacobian;

    public:
    BroydenBad():ScfMethod{}{ inv_Jacobian = NULL;}
    BroydenBad(VectorXcd* X, MatrixXcd* Jinv, int iter, int max_iter, double tol_abs, double tol_rel, string save_path, int saving_frequency)
        :ScfMethod{X,iter,max_iter,tol_abs,tol_rel,save_path,saving_frequency}, inv_Jacobian{Jinv} {}
    BroydenBad(const BroydenBad& b):ScfMethod(b){inv_Jacobian = b.inv_Jacobian;}
    ~BroydenBad(){}

    void iterate(function<VectorXcd(const VectorXcd&)> FX );
    void set_inv_Jacobian(MatrixXcd* Jinv){inv_Jacobian = Jinv;}
};



class PulayPeriodic : public ScfMethod
{
    private:
    double alpha; // mixing parameter
    int m; // memory
    int p; // period

    public:
    PulayPeriodic(): ScfMethod{}{alpha = 1.0; m = 1; p = 1;}
    PulayPeriodic(VectorXcd* X, double alpha, int m, int p, int iter, int max_iter, double tol_abs, double tol_rel, string save_path,int saving_frequency)
        :ScfMethod{X,iter,max_iter,tol_abs,tol_rel, save_path, saving_frequency}, alpha{alpha}, m{m}, p{p} {}
    PulayPeriodic(const PulayPeriodic& b):ScfMethod(b){alpha=b.alpha; m = b.m; p= b.p;}
    ~PulayPeriodic(){}
    void iterate(function<VectorXcd(const VectorXcd&)> FX );
};



#endif
