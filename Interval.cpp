#include <complex>
#include <tuple>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
//#include <omp.h>

#include "Interval.h"
#include "quadrature.h"

using namespace std;
using namespace Eigen;

//#define THREAD_NUM = 4


Interval::Interval(double lp, double rp, int points, function<double(double)> f)
    : lp{lp}, rp{rp}, points{points}, points_old{0}
{
    interval = false;
    singular = false;
    x = VectorXd::LinSpaced(points,lp,rp);
    y = eval_func(f,x,singular);
    tie(error,integral,integral_abs,rmax) = quadrature(lp,rp,y,32,interval); 

}

Interval::Interval(VectorXd x, VectorXd y)
    : lp{x(0)}, rp{x(x.size()-1)}, x{x}, y{y}
{
    interval = false;
    singular = false;
    points = y.size();
    points_old = 0;
    tie(error,integral,integral_abs,rmax) = quadrature(lp,rp,y,32,interval); 
}

void Interval::set_limits_and_points(double lp, double rp,int points, function<double(double)> f)
{
    this->lp = lp;
    this->rp = rp;
    points_old = this->points;
    this->points = points;

    VectorXd xnew = VectorXd::LinSpaced(points,lp,rp);
    VectorXd ynew = VectorXd::Zero(points);
    bool is_included;
    for(int i = 0; i < xnew.size(); i++){
        is_included = false;
        for(int j = 0; j < x.size(); j++){
            if(abs(xnew(i) - x(j))<1.0E-10){
                ynew(i) = y(j);
                is_included = true;
                break;
            }
        }
        if(!is_included)
            ynew(i) = eval_func(f,x(i),singular);
    }
    y = ynew;
    x = xnew;

    

}

void Interval::upgrade(function<double(double)> f)
{
    if (points == 33)
        return;
    points_old = points;
    int new_points = points-1;
    double cc = (lp+rp)/2;
    double hh = (rp-lp)/2;
    double llimit = cc - hh*((double)new_points-1.0)/new_points;
    double ulimit = cc + hh*((double)new_points-1.0)/new_points;
    VectorXd xnew = VectorXd::LinSpaced(new_points,llimit,ulimit);

    // Update x and y
    VectorXd ynew = eval_func(f,xnew,singular);
    points += new_points;
    VectorXd xold = x;  //
    VectorXd yold = y; // Previous values
    x = VectorXd::Zero(points);
    y = VectorXd::Zero(points);
    for(int i = 0; i < new_points; i++){
        x(2*i) = xold(i);
        y(2*i) = yold(i);
        x(2*i+1) = xnew(i);
        y(2*i+1) = ynew(i);
    }
    y(points-1) = yold(new_points);
    x(points-1) = xold(new_points);

    tie(error,integral,integral_abs,rmax) = quadrature(lp,rp,y,32,interval); 
}

tuple<Interval,Interval> Interval::split()
{
    if (points == 5)
        return make_tuple(*this,*this);
    int split_points = (points+1)/2;
    Interval L(x.head(split_points),y.head(split_points));
    Interval R(x.tail(split_points),y.tail(split_points));
    L.set_points_old(points);
    R.set_points_old(points);

    return make_tuple(L,R);
}


VectorXd eval_func(function<double(double)> f,const VectorXd& x,bool &singular)
{
    VectorXd y = VectorXd::Zero(x.size());
    //#pragma omp parallel for
    for(int j = 0; j < x.size(); j++){
        y(j) = f(x(j));
        if(isinf(real(y(j))) || isinf(abs(y(j)))){
            y(j) = 0;
            singular = true;
        }
        if(isnan(real(y(j))) || isnan(imag(y(j))))
            y(j) = 1.0;
    }
    return y;
}

double eval_func(function<double(double)> f,double x, bool &singular)
{
    double y;
    y = f(x);
    if(isinf(real(y)) || isinf(abs(y))){
        y = 0;
        singular = true;
    }
    if(isnan(real(y)) || isnan(imag(y)))
        y = 1.0;
    return y;
}

