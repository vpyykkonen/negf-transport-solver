#ifndef INTERVAL_H
#define INTERVAL_H

#include <complex>
#include <tuple>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;


class Interval
{
    private:
    double lp; // left limit
    double rp; //  right limit
    int points; // number of points
    int points_old; // number of points before upgrade or split 
    //(equal to points if no operation has been done)
  
    double integral; // local integral 
    double integral_abs; // local absolute value integral
    double error; // local error
    //int status; // which rule is applied (how many points)
    double rmax;

    bool interval;
    bool singular;

    VectorXd x; // Interval grid
    VectorXd y; // Function values within interval


    public:
    Interval(){points = 1;points_old = 0; lp = 0.0; rp = 0.0; integral = 0.0; integral_abs = 0.0; error = 0.0;rmax = 0; y = VectorXd::Zero(1); interval = false; singular = false;}
    Interval(double lp, double rp, int points, function<double(double)> f);
    Interval(VectorXd x, VectorXd y);
    Interval(const Interval& i){points = i.points; points_old = i.points_old; lp=i.lp;rp=i.rp;integral=i.integral; integral_abs=i.integral_abs;  error = i.error; rmax = i.rmax; x = i.x; y = i.y; interval = false; singular = false;}
    ~Interval(){}


    void set_limits_and_points(double lp,double rp,int points, function<double(double)> f);
    void set_points_old(int points_old){this->points_old = points_old;}

    double get_lp() const{return lp;}
    double get_rp() const{return rp;}
    int get_points() const{return points;}
    int get_points_old() const{return points_old;}
    double get_integral() const{return integral;}
    double get_integral_abs() const{return integral_abs;}
    double get_error()const {return error;}
    double get_rmax() const {return rmax;}
    bool get_interval() const {return interval;}
    bool get_singular() const {return singular;}
    VectorXd get_x() const {return x;}
    VectorXd get_y() const {return y;}

    void upgrade(function<double(double)> f);
    tuple<Interval,Interval> split();
   

};

VectorXd eval_func(function<double(double)> f, const VectorXd& x, bool & singular);
double eval_func(function<double(double)> f, double x, bool & singular);


#endif
