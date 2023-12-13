// Double adaptive global quadrature
// Inputs:
// function<double(double)> f function to be integrated
// VectorXd a Initial intervals as collection of subsequent points
// double tolabs Required absolute tolerance 
// double tolrel Required relative tolerance
// bool step_info Whether to  print information at each step
// bool trace Whether to print the intervals at the end
//
// Outputs:
// tuple<double,int> tuple of integral value and number of function evaluations
//
// Adapative structure:
// Start: Apply 9 point rule to the initial intervals
// Then follow the following rules to the interval with the largest error until global
// convergence:
// - If 5 point rule has been applied, apply 9 point rule.
// - If 9 points rule has been applied and previously either no rule or 5 point rule,
// split in half and apply the 5 point rule to both intervals. If error is not reduced but 
// rmax < 1/4, apply 17 point rule to the intervals, else store the splited intervals.
// - If 9 point rule has been applied and previously 17 point rule, apply 17 point rule
// - If 17 point rule has been applied and previously 9 point rule, split and
// apply 9 point rule to both parts. If the error is not reduced but rmax < 1/8, 
// apply 33 point rule to the original interval, else store the splited intervals.
// - If 17 point rule has been applied and previously 33 point rule, apply 33 point rule.
// - If 33 point rule has been applied, split to two 17 point intervals and store them.
// At each step, update also the global parameters
// Split can be done by the split function of Interval class. 
// Applying more accurate rule is done by the upgrade function of the Interval class.
#include "da2glob.h"

#include <tuple>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <complex>
#include <memory>

#include "file_io.h"
#include "Global.h"
#include "Interval.h"

using namespace std;
using namespace Eigen;


tuple<double,int> da2glob(function<double(double)> f, const VectorXd& a, double tolabs, double tolrel, bool step_info, bool trace)
{
    static double eps = numeric_limits<double>::epsilon();

    // Outputs
    double Q = 0;
    int nfun = 0;

    // Number of initial intervals
    int nint0 = a.size()-1;
    if(nint0 < 1)
        return make_tuple(Q,nfun);

    double noise = 0.0;
    double Lerr;
    double Rerr;

    // Maximum number of intervals and function evaluations
    int int_max = 100000;
    int nmax = 60000;

    // Initialize the Global object, containing the intervals and
    // global information using the 9-point rule
    unique_ptr<Global> g(new Global(a,9,f));

    // calculate the global noise level
    noise = 50*eps*g->get_integral_abs();

    Interval* inte; // Index for the current interval to be handled
    int count = 0;  // aux counter for printing information
    int nint_old = 0;
    while(g->get_error_sum() > max(max(abs(g->get_integral())*tolrel,tolabs),noise) && g->get_points() < nmax-3 && g->get_n_intervals() < int_max){
        // Find the interval with maximal error
        inte = g->get_max_error_interval();
        Interval inteL;
        Interval inteR;
        //if(step_info){
        //    count ++;
        //    nint_old = g->get_n_intervals();;
        //    cout << "Iteration no. " << count << endl;
        //    cout << "Number of intervals: " << nint_old << endl;
        //    cout << "Number of function evaluations: " << g->get_points() << endl;
        //    cout << "Interval info: ";
        //    cout << "lp, rp: " << inte->get_lp() << " " << inte->get_rp() << endl;
        //    cout << "points, old_points: " << inte->get_points() << " " << inte->get_points_old() << endl;
        //    cout << "integral, error:" << inte->get_integral() << " " << inte->get_error() << endl;
        //    cout << "rmax " << inte->get_rmax() << endl;
        //}
        // Substract old variable values since these will be updated (keep the Interval in the vector)
        //
        //count ++;
        //cout << count << endl;
        //cout << inte->get_lp() << " " << inte->get_rp() << endl;
        //cout << inte->get_error() << endl;
        //cout << g->get_error() << endl;
        //cout << g->get_error_sum() << endl;
        g->remove_globals(*inte);


        if(inte->get_points() == 5) {
            // Upgrade to 9 point rule
            inte->upgrade(f);
            g->add_globals(*inte);// update global variables
        } else if (inte->get_points() == 9 && (inte->get_points_old() == 0 || inte->get_points_old() == 5)){
            //  Split the interval to two 5 point rule intervals
            tie(inteL,inteR) = inte->split();
            Lerr = inteL.get_error();
            Rerr = inteR.get_error();
            // Upgrade original to 17 if rmax is small but split increases the error
            if(inte->get_rmax() < 1.0/4 && Lerr+ Rerr >= inte->get_error()){
                inte->upgrade(f);
                g->add_globals(*inte); // update only the globals
            }
            else{ // Continue with the split
                *inte = inteL; // store left at the old interval's place
                g->add_globals(inteL); // -> update only globals 
                g->add_interval(inteR); // store right interval
            }
        } else if(inte->get_points() == 9 && inte->get_points_old() == 17){
            // Upgrade to 17 point rule
            inte->upgrade(f);
            g->add_globals(*inte); // update the global variables
        } else if ( inte->get_points() == 17 && inte->get_points_old() == 9){
            // Split the interval to two 9 point rule intervals
            tie(inteL,inteR) = inte->split();
            Lerr = inteL.get_error();
            Rerr = inteR.get_error();
            // Upgrade original to 33 if rmax is small but split increases the error
            if(inte->get_rmax() < 1.0/8 && Lerr + Rerr >= inte->get_error()){
                inte->upgrade(f); // Compute f at 16 new points
                g->add_globals(*inte); // Update the global variables
            } else {
                // Store the parts to data structure and update global variables
                *inte = inteL;
                g->add_globals(inteL); // Only update the globals
                g->add_interval(inteR); // Add new interval 
            }
        } else if (inte->get_points() == 17 && inte->get_points_old() == 33 ) {
            // Upgrade
            inte->upgrade(f); // Compute f in 16 new points
            g->add_globals(*inte); // update global variables
        } else if (inte->get_points() == 33) {
            // Split the interval to two 17 point rule intervals
            tie(inteL,inteR) = inte->split();
            *inte = inteL; // Store left interval in the data structure
            g->add_globals(inteL); // Update global variables for left
            g->add_interval(inteR); // add the right interval
        }
        // Redefine global noise level
        noise = 50*eps*g->get_integral_abs();
        if(step_info){
            cout << "Iteration no. " << count << endl;
            if(nint_old < g->get_n_intervals()){
                cout << "Split to 2x " << inteL.get_points() << " from " << inteL.get_points_old() << endl;
            }
            if(inte->get_points() > inte->get_points_old())
                cout << "Upgrade to " << inte->get_points() << " from " << inte->get_points_old() << endl;

            g->print_globals();
            //g->print_intervals();
            //cout << "Integral estimate: " << g->get_integral() << endl;
            //cout << "Error estimate: " << g->get_error() << endl;
            //cout << "---" << endl;
        }
    }
    if ( g->get_points() >= nmax - 3){
        cout << "Stopping: maximum number of f evaluations, required tolerance maybe not met" << endl;
        for(int j=0; j < g->get_n_intervals(); j++){
            Interval inte = g->get_interval(j);
            cout << inte.get_lp() << " " << inte.get_rp()  << " " <<inte.get_integral() << " " << inte.get_error() << " " << inte.get_points() << " " << inte.get_points_old() << endl;
        }
    }
    if( g->get_n_intervals() >= int_max)
        cout << "Stopping: maximum number of intervals reached, required tolerance may no be met" << endl;
    if ( g->get_inter_too_small())
        cout << "Some interval too small, required tolerance may not be met" << endl;
    if( max(abs(g->get_integral())*tolrel,tolabs) < noise)
        cout << "Stopping:the tolerance below the noise level of the problem, required tolerance may not be met." << endl;
    if(g->get_singular())
        cout << "Singularity probably detected, required tolerance may not be met" << endl;

    if (trace){
        for(int j=0; j < g->get_n_intervals(); j++){
            Interval inte = g->get_interval(j);
            cout << inte.get_lp() << " " << inte.get_rp()  << " " <<inte.get_integral() << " " << inte.get_error() << " " << inte.get_points() << " " << inte.get_points_old() << endl;
        }
    }
    Q = g->get_integral_sum();
    nfun = g->get_points();
    return make_tuple(Q,nfun);
}




