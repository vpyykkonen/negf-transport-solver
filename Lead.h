#ifndef LEAD_H
#define LEAD_H

#include <complex>
#include <tuple>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef complex<double> dcomp;

class Lead
{
    private:
    double tL;
    double muL;
    dcomp Delta;
    double T;

    public:
    Lead() { tL = 30; Delta = 1; muL = 0 ; T = 0;}

    Lead(double t, dcomp delta, double mu, double temp) { tL = t; Delta = delta; muL = mu; T = temp;}

    Lead(const Lead& l){ tL = l.tL; Delta = l.Delta, muL = l.muL; T = l.T;}

    ~Lead(){;}

    void set_tL(const double tL){this->tL = tL;}
    void set_Delta(const dcomp Delta){this->Delta = Delta;}
    void set_muL(const double muL){this->muL = muL;}
    void set_T(const double T){this->T = T;}
    void set_phi(const double phi){this->Delta = polar(abs(this->Delta),phi);}

    double get_tL(){return this->tL;}
    dcomp get_Delta(){return this->Delta;}
    double get_muL(){return this->muL;}
    double get_T(){return this->T;}
    double get_phi(){return arg(this->Delta);}

    tuple<Matrix2cd,Matrix2cd> get_gR_and_gl(double E, dcomp ieta);
    tuple<dcomp,dcomp> get_non_nambu_gR_and_gl(double E, dcomp ieta);

    Matrix2cd get_g(dcomp omega);

    VectorXcd GRLL(const VectorXcd& Gvec, double E, dcomp ieta, Matrix2cd g);
    tuple<Matrix2cd,Matrix2cd> get_gR_and_gl(double E, dcomp ieta, double tol);



};

#endif 
