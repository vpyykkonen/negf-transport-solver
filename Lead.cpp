#include "Lead.h"
#include <iostream>
#include <complex>
#include <tuple>

#include <Eigen/Dense>
#include "ScfMethod.h"
//#include <Eigen/KroneckerProduct>

#include <limits>
#include "fd_dist.h"
#include "pade_frequencies.h"
using namespace std;
using namespace Eigen;
typedef complex<double> dcomp;

tuple<Matrix2cd,Matrix2cd> Lead::get_gR_and_gl(double E, dcomp ieta)
{
    Matrix2cd gR,gl;
    double fd = 0.0;
    if( abs(this->Delta) < numeric_limits<double>::epsilon()){
        dcomp phi_p = acos((E+this->muL+ieta)/(2.0*this->tL));
        dcomp phi_h = acos((E-this->muL+ieta)/(2.0*this->tL));
        //dcomp phi_p = acos(E/(2.0*this->tL));
        //dcomp phi_h = acos(E/(2.0*this->tL));
        //dcomp phi_p = acos((E+ieta)/(2.0*this->tL));
        //dcomp phi_h = acos((E+ieta)/(2.0*this->tL));
        //dcomp phi_p = acos((E+this->muL)/(2.0*this->tL));
        //dcomp phi_h = acos((E-this->muL)/(2.0*this->tL));
        gR << exp(-1.0i*phi_p)/this->tL , 0.0 ,
           0.0 , exp(-1.0i*phi_h)/this->tL;
        double fd_p = fd_dist(E,this->muL,this->T);
        double fd_h = fd_dist(E,-this->muL,this->T);

        gl << fd_p*(conj(gR(0,0))-gR(0,0)) , 0.0,
           0.0, fd_h*(conj(gR(1,1))-gR(1,1));

        //gl = fd*(gR.adjoint()-gR);
        return make_tuple(gR,gl);
    }

    //gR << -(E+ieta - this->muL), this->Delta,
    //   conj(this->Delta), -(E+ieta + this->muL);
    //gR *= 1.0/(this->tL*sqrt(pow(abs(this->Delta),2) - (E+this->muL+ieta)*(E-this->muL+ieta)));
    gR << -(E+ieta), this->Delta,
       conj(this->Delta), -(E+ieta);
    gR *= 1.0/(this->tL*sqrt(pow(abs(this->Delta),2) - (E+ieta)*(E+ieta)));

    fd = fd_dist(E,0,this->T);
    gl = fd*(gR.adjoint()-gR);

    return make_tuple(gR,gl);
}

// Get g on complex plane, 
// Matsubara Gm or Pade Gp with appropriate imaginary frequencies
// Retarded GR/GA with E +- i eta
Matrix2cd Lead::get_g(dcomp omega)
{
    Matrix2cd g;
    if( abs(this->Delta) < numeric_limits<double>::epsilon()){
        dcomp phi_p = acos((omega+this->muL)/(2.0*this->tL));
        dcomp phi_h = acos((omega-this->muL)/(2.0*this->tL));
        //dcomp phi_p = acos((omega)/(2.0*this->tL));
        //dcomp phi_h = acos((omega)/(2.0*this->tL));
        g << exp(-1.0i*phi_p)/this->tL , 0.0 ,
           0.0 , exp(-1.0i*phi_h)/this->tL;
        return g;
    }

    //g << -(omega - this->muL), this->Delta,
    //   conj(this->Delta), -(omega + this->muL);
    //g *= 1.0/(this->tL*sqrt(pow(abs(this->Delta),2) - (omega+this->muL)*(omega-this->muL)));
    g << -omega, this->Delta,
       conj(this->Delta), -omega;
    g *= 1.0/(this->tL*sqrt(pow(abs(this->Delta),2) - omega*omega));
    return g;
}

tuple<dcomp,dcomp> Lead::get_non_nambu_gR_and_gl(double E, dcomp ieta)
{
    dcomp phi_p = acos((E+this->muL+ieta)/(2.0*this->tL));
    dcomp gR = exp(-1.0i*phi_p)/this->tL;

    double fd = fd_dist(E,this->muL,this->T);
    dcomp gl = fd*(conj(gR)-gR);

    return make_tuple(gR,gl);
}

VectorXcd Lead::GRLL(const VectorXcd& Gvec, double E, dcomp ieta, Matrix2cd g)
{
    Matrix2cd Gin,Gout;
    Matrix2cd Sigmaz;
    //Sigmaz << 1.0, 0.0, 0.0, -1.0;
    VectorXcd Goutvec(4);


    // - signs for G(1,0),G(1,1) to take the sigmazs into account
    Gin << Gvec(0), Gvec(1), -Gvec(2), -Gvec(3);
    Gout = g + pow(this->tL,2)*g*Gin*Gin;
    Goutvec << Gout(0,0), Gout(0,1), Gout(1,0), Gout(1,1);
    return Goutvec;
}

tuple<Matrix2cd,Matrix2cd> Lead::get_gR_and_gl(double E, dcomp ieta, double tol)
{
    using namespace std::placeholders;

    Matrix2cd gR, gl, g;
    VectorXcd gRvec(4);
    int iter = 0, max_iter = 1000000;
    MatrixXcd Jinv = -0.344*MatrixXcd::Identity(4,4);

    // Initial guess
    gR << E+ieta-this->muL , this->Delta , conj(this->Delta) , E+ieta+this->muL;
    gR *= 1.0/((E+ieta-this->muL)*(E+ieta+this->muL)-pow(abs(Delta),2));

    g << E+ieta-this->muL , this->Delta , conj(this->Delta) , E+ieta+this->muL;
    g *= 1.0/((E+ieta-this->muL)*(E+ieta+this->muL)-pow(abs(Delta),2));


    gRvec << gR(0,0), gR(0,1), gR(1,0), gR(1,1);
    BroydenGood solv(&gRvec, &Jinv, iter, max_iter, tol, tol,"",0);

    solv.iterate(std::bind(&Lead::GRLL, this, _1, E, ieta, g));
    cout << solv.get_iter() << endl;
    cout << solv.get_error_rel() << endl;
    cout << Jinv;

    gR << gRvec(0), gRvec(1), gRvec(2), gRvec(3);

    double fd = fd_dist(E,0,this->T);
    gl = fd*(gR.adjoint()-gR);

    return make_tuple(gR,gl);
}





            
    


    


