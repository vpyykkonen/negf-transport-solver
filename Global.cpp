#include <complex>
#include <vector>
#include <algorithm>
#include <tuple>
#include <functional>
#include <Eigen/Dense>
#include "Global.h"
#include "Interval.h"


using namespace std;
using namespace Eigen;

Global::Global(const VectorXd& a, int inter_points, function<double(double)> f)
{
    points = 0;
    n_intervals = 0;
    integral = 0.0;
    integral_abs = 0.0;
    error = 0.0;
    singular = false;
    inter_too_small = false;
    int nint0 = a.size()-1;
    for(int i = 0; i < nint0; i++){
        Interval inte(a(i), a(i+1),inter_points,f);
        this->add_interval(inte);
    }
}

void Global::add_interval(const Interval& inte)
{
    intervals.push_back(inte);
    n_intervals ++;
    points += inte.get_points();
    integral += inte.get_integral();
    integral_abs += inte.get_integral_abs();
    error += inte.get_error();
    inter_too_small = inter_too_small | inte.get_interval();
    singular = singular | inte.get_singular();
}

void Global::add_globals(const Interval& inte)
{
    integral += inte.get_integral();
    integral_abs += inte.get_integral_abs();
    error += inte.get_error();
}

void Global::remove_interval(int i)
{
    n_intervals --;
    points -= intervals[i].get_points();
    integral -= intervals[i].get_integral();
    integral_abs -= intervals[i].get_integral_abs();
    error -= intervals[i].get_error();
    intervals.erase(intervals.begin()+i);
}

void Global::remove_globals(const Interval& inte)
{
    integral -= inte.get_integral();
    integral_abs -= inte.get_integral_abs();
    error -= inte.get_error();
}

double Global::get_error_sum()
{
    double error_sum = 0.0;
    for(auto it = intervals.begin(); it != intervals.end(); ++it){
        error_sum += it->get_error();
    }
    return error_sum;
}

double Global::get_integral_sum()
{
    double integral_sum = 0.0;
    for(auto it = intervals.begin(); it != intervals.end(); ++it){
        integral_sum += it->get_integral();
    }
    return integral;
}

void Global::print_intervals()
{
    cout << "Interval info: " << endl;
    for(auto it = intervals.begin(); it != intervals.end(); ++it){
        cout << "Interval nro. " << it-intervals.begin()+1 << endl;
        cout << "lp, rp " << it->get_lp() << " " << it->get_rp() << endl;
        cout << "points, old_points " << it->get_points() << " " << it->get_points_old() << endl;
        cout << "integral, error " << it->get_integral() << " " << it->get_error() << endl;
    }
}

void Global::print_globals()
{
    cout << "Global info: " << endl;
    cout << "n_ints: " << n_intervals << endl;
    cout << "n_points: " << points << endl;
    cout << "Integral: " << integral << endl;
    cout << "Error: " << error << endl;
    cout << "Error sum: " << this->get_error_sum() << endl;
}

Interval* Global::get_max_error_interval()
{
    double err = intervals[0].get_error();
    //int max_idx = 0;
    Interval* inte = &intervals[0];
    for(int j =1; j < n_intervals ; j ++){
        if( intervals[j].get_error() > err){
            //max_idx = j;
            inte = &intervals[j];
            err = inte->get_error();
        }
    }
    return inte;
}
