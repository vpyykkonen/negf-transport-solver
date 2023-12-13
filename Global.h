#ifndef GLOBAL_H
#define GLOBAL_H

#include <complex>
#include <vector>
#include <tuple>
#include <functional>
#include "Interval.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Global
{
    private:
    int points;
    int n_intervals;
    double integral;
    double integral_abs;
    double error;
    bool inter_too_small;
    bool singular;

    vector<Interval> intervals;

    public:
    Global(){integral = 0; integral_abs = 0; error = 0; inter_too_small = false; singular = false; points = 0; n_intervals=0;}
    Global(const VectorXd& a, int points, function<double(double)> f);
    Global(const Global& g){integral = g.integral; integral_abs = g.integral_abs; error = g.error; inter_too_small = g.inter_too_small; points=g.points; n_intervals =g.n_intervals; intervals = g.intervals;}
    ~Global(){}

    int get_points() const{return points;}
    int get_n_intervals() const{return n_intervals;}
    double get_integral() const{return integral;}
    double get_integral_abs() const{return integral_abs;}
    double get_error() const{return error;}
    bool get_inter_too_small() const{return inter_too_small;}
    bool get_singular() const{return singular;}
    Interval& get_interval(int i){return intervals[i];}
    Interval* get_max_error_interval();

    double get_error_sum();
    double get_integral_sum();

    void add_interval(const Interval& inte);
    void add_globals(const Interval& inte);
    void remove_interval(int idx);
    void remove_globals(const Interval& inte);

    void print_intervals();
    void print_globals();
};

#endif
