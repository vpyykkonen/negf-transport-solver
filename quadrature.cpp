#include <tuple>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <complex>

#include "file_io.h"
#include "quadrature.h"

using namespace std;
using namespace Eigen;



tuple<double, double, double, double> quadrature(double lp, double rp, VectorXd y, double C, bool& interval)
{
    int points = y.size();
    if(points == 5){
        return cotes5(lp,rp,y,C,interval);
    }else if(points == 9){
        return cotes9(lp,rp,y,C,interval);
    }else if(points == 17){
        return cotes17(lp,rp,y,C,interval);
    }else if(points == 33){
        return rule33(lp,rp,y,C,interval);
    }else{
        interval = false;
        return make_tuple(0.0,0.0,0.0,0.0);
    }
}

tuple<double, double, double, double> cotes5(double a, double b, VectorXd y, double D, bool& interval)
{
    // Read weights for the considered quadratures and null-rules
    // w and nw are the weights and nullrules associated with Newton-Cotes 5 point rule.
    static VectorXd w = readRealMatrix("./da2glob_weights/w5.csv").transpose();
    static MatrixXd nw = readRealMatrix("./da2glob_weights/nw5.csv");
    static double eps = numeric_limits<double>::epsilon();
    // output parameters
    double err = 0, Qabs = 0, rmax = 2;
    double Q = 0;

    double h = (b-a)/2.0;
    VectorXd x = VectorXd::LinSpaced(5,a,b);
    //Q = ((h*w.transpose())*y)(0,0);
    //Qabs = ((h*w.cwiseAbs().transpose())*y.cwiseAbs())(0,0);
    Q = h*w.dot(y);
    Qabs = h*w.cwiseAbs().dot(y.cwiseAbs());
    // define local noise level
    double noise = 50*Qabs*eps;
    // Error estimates
    Vector4d E = (h*(nw*y)).cwiseAbs();

    double Emin = E.segment(1,3).minCoeff();
    Vector3d r;
    if( Emin == 0)
        rmax = 2;
    else{
        bool is_inf = false;
        for(int i = 0; i < 3; i++){
            r(i) = E(i)/E(i+1);
            is_inf |= isinf(r(i));
        }
        if(is_inf)
            rmax = 2;
        else
            rmax = r.maxCoeff();
    }
    if(rmax > 1)
        err = D*E.maxCoeff();
    else if ( 0.5 < rmax)
        err = D*rmax*E(1);
    else
        err = D*pow(2*rmax,3)*rmax*E(1);

    // If the highest degree null rules are on local noise level, put the error to zero
    if(E(0) < noise && E(1) < noise)
        err = 0;

    if(x(1) <= a || b <= x(3)){
        interval = 1;
        err = 0;
    }
    return make_tuple(err,Q,Qabs,rmax);
}

// Applies 9 point Newton-Cotes rule to approximate integral of function F(X)
// from a to b, stored in vector y. The constant D is used in error estimate.
// Interval signals if too small intervals has been found.
//
// Returns error, integral value, integral of absolute value of the function and rmax
//
tuple<double, double, double, double> cotes9(double a, double b, VectorXd y, double D, bool& interval)
{
    // Read weights for the considered quadratures and null-rules
    // w and nw are the weights and nullrules associated with Newton-Cotes 9 point rule.
    static VectorXd w = readRealMatrix("./da2glob_weights/w9.csv").transpose();
    static MatrixXd nw = readRealMatrix("./da2glob_weights/nw9.csv");
    static double eps = numeric_limits<double>::epsilon();
    // output parameters
    double err = 0, Qabs = 0, rmax = 2;
    double Q = 0;

    double h = (b-a)/2.0;
    VectorXd x = VectorXd::LinSpaced(9,a,b);
    //Q = ((h*w.transpose())*y)(0,0);
    //Qabs = ((h*w.cwiseAbs().transpose())*y.cwiseAbs())(0,0);
    Q = h*w.dot(y);
    Qabs = h*w.cwiseAbs().dot(y.cwiseAbs());
    // define local noise level
    double noise = 50*Qabs*eps;
    // Error estimates
    VectorXd e=h*(nw*y);
    Vector4d E;
    E << e.segment(0,2).norm(), e.segment(2,2).norm(), e.segment(4,2).norm(), e.segment(6,2).norm();
    double Emin = E.segment(1,3).minCoeff();
    Vector3d r;
    if( Emin == 0)
        rmax = 2;
    else{
        bool is_inf = false;
        for(int i = 0; i < 3; i++){
            r(i) = E(i)/E(i+1);
            is_inf |= isinf(r(i));
        }
        if(is_inf)
            rmax = 2;
        else
            rmax = r.maxCoeff();
    }
    if(rmax > 1)
        err = D*E.maxCoeff();
    else if ( 0.25 < rmax)
        err = D*rmax*E(0);
    else
        err = D*4*rmax*rmax*E(0);

    // If the highest degree null rules are on local noise level, put the error to zero
    if(E(0) < noise && E(1) < noise)
        err = 0;

    if(x(1) <= a || b <= x(7)){
        interval = 1;
        err = 0;
    }
    return make_tuple(err,Q,Qabs,rmax);
}

tuple<double, double, double, double> cotes17(double a, double b, VectorXd y, double D, bool& interval)
{
    // Read weights for the considered quadratures and null-rules
    // w and nw are the weights and nullrules associated with Newton-Cotes 17 point rule.
    static VectorXd w = readRealMatrix("./da2glob_weights/w17.csv").transpose();
    static MatrixXd nw = readRealMatrix("./da2glob_weights/nw17.csv");
    static double eps = numeric_limits<double>::epsilon();
    // output parameters
    double err = 0, Qabs = 0, rmax = 2;
    double Q = 0;

    double h = (b-a)/2.0;
    VectorXd x = VectorXd::LinSpaced(17,a,b);
    //Q = ((h*w.transpose())*y)(0,0);
    //Qabs = ((h*w.cwiseAbs().transpose())*y.cwiseAbs())(0,0);
    Q = h*w.dot(y);
    Qabs = h*w.cwiseAbs().dot(y.cwiseAbs());
    // define local noise level
    double noise = 50*Qabs*eps;
    // Error estimates
    VectorXd e=((h*nw)*y).cwiseAbs();
    VectorXd E(5);
    E << e.segment(0,3).norm(), e.segment(3,3).norm(),e.segment(6,3).norm(), e.segment(9,3).norm(), e.segment(12,3).norm();
    double Emin = E.segment(1,4).minCoeff();
    Vector4d r;
    if( Emin == 0)
        rmax = 2;
    else{
        bool is_inf = false;
        for(int i = 0; i < 4; i++){
            r(i) = E(i)/E(i+1);
            is_inf |= isinf(r(i));
        }
        if(is_inf)
            rmax = 2;
        else
            rmax = r.maxCoeff();
    }
    if(rmax > 1)
        err = D*E.maxCoeff();
    else if ( 0.125 < rmax)
        err = D*rmax*E(0);
    else
        err = D*pow(8*rmax,2/3)*rmax*E(0);

    // If the highest degree null rules are on local noise level, i put the error to zero
    if(E(0) < noise && E(1) < noise){
        err = 0.0;
        //cout << "below noise!" << endl;
    }
    // If the error is beaneath the noise level, put error to zero
    //cout << "17 rule" << endl;
    //cout << err << " " << noise << endl;
    //if( err < noise)

    if(x(1) <= a || b <= x(15)){
        interval = 1;
        err = 0;
    }
    return make_tuple(err,Q,Qabs,rmax);
}

tuple<double, double, double, double> rule33(double a, double b, VectorXd y, double D,  bool& interval)
{
    // Read weights for the considered quadratures and null-rules
    // w and nw are the weights and nullrules associated with 33 point rule (degree 27).
    static VectorXd w = readRealMatrix("./da2glob_weights/w33.csv").transpose();
    static MatrixXd nw = readRealMatrix("./da2glob_weights/nw33.csv");
    static double eps = numeric_limits<double>::epsilon();
    // output parameters
    double err = 0,  Qabs = 0;
    double Q = 0;

    double rmax = 2;
    double h = (b-a)/2.0;
    VectorXd x = VectorXd::LinSpaced(33,a,b);
    //Q = ((h*w.transpose())*y)(0,0);
    //Qabs = ((h*w.cwiseAbs().transpose())*y.cwiseAbs())(0,0);
    Q = h*w.dot(y);
    Qabs = h*w.cwiseAbs().dot(y.cwiseAbs());
    // define local noise level
    double noise = 50*Qabs*eps;
    // Error estimates
    VectorXd e=(h*(nw*y)).cwiseAbs();
    VectorXd E(5);
    E << e.segment(0,3).norm(), e.segment(3,3).norm(),e.segment(6,3).norm(), e.segment(9,3).norm(), e.segment(12,3).norm();
    double Emin = E.segment(1,4).minCoeff();
    Vector4d r;
    if( Emin == 0)
        rmax = 2;
    else{
        bool is_inf = false;
        for(int i = 0; i < 4; i++){
            r(i) = E(i)/E(i+1);
            is_inf |= isinf(r(i));
        }
        if(is_inf)
            rmax = 2;
        else
            rmax = r.maxCoeff();
    }
    if(rmax > 1)
        err = D*E.maxCoeff();
    else{
        err = D*e.segment(0,4).norm();
    }

    //cout << "Error: " << err << endl;
    //cout << "Noise: " << noise << endl;
    //cout << "E: " << E.transpose() << endl;

    // If the highest degree null rules are on local noise level, i put the error to zero
    if(E(0) < noise && E(1) < noise){
        err = 0.0;
        //cout << "below noise" << endl;
    }
    // If the error is beneath the noise level, set error to zero
    //cout << "33 rule" << endl;
    //cout << err << " " << noise << endl;
    //if(err < noise){
    //    cout << "below noise" << endl;
    //    err = 0.0;
    //}

    // If the interval is too small, then handle specfically
    if(x(1) <= a || b <= x(31)){
        interval = 1;
        err = 0.0;
        //cout << "interval too small" << endl;
    }
    //cout << err << endl;
    return make_tuple(err,Q,Qabs,rmax);
}
