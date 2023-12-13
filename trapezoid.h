#ifndef TRAPEZOID_H
#define TRAPEZOID_H

#include <complex>
#include <functional>
#include <vector>

using namespace std;

typedef complex<double> dcomp;

template <class inScalar, class outScalar>
outScalar trapezoid(function<outScalar(inScalar)> f,vector<inScalar> points)
{
    if(points.size() == 0 || points.size() == 1){
        cout << "Too few points given. Returning" << endl;
        return 0.0;
    }
        
    outScalar quad = 0.0;
    int n_points = points.size();

    inScalar l_int1 = points[1]-points[0];
    quad += 0.5*l_int1*f(points[0]);
    inScalar l_int2 = l_int1;
    for(int n = 1; n < n_points-1;n++){ 
        l_int2 = points[n+1]-points[n];
        quad += 0.5*(l_int1+l_int2)*f(points[n]);
        l_int1 = l_int2;
    }
    quad += 0.5*l_int2*f(points[n_points-1]);

    return quad;
}


template <class inScalar, class outScalar>
outScalar trapezoid(function<outScalar(inScalar)> f,inScalar a, inScalar b, int n_intervals)
{
    vector<inScalar> points;
    for(int n = 0; n < n_intervals+1;n++)
        points.push_back(a+(a-b)*n/n_intervals);

    return trapezoid<inScalar,outScalar>(f,points);
}



// Trapezoid assuming that the points are already ordered
template <class inScalar, class outScalar>
outScalar trapezoid(vector<outScalar> fvals, vector<inScalar> points)
{
    if(fvals.size() != points.size()){
        cout << "Number of points and function values do not match." << endl;
        return 0.0;
    }
    if(points.size() == 0 || points.size() == 1){
        cout << "Too few given. Returning" << endl;
        return 0.0;
    }
        
    outScalar quad = 0.0;
    int n_points = points.size();



    inScalar l_int1 = points[1]-points[0];
    quad += 0.5*l_int1*fvals[0];
    inScalar l_int2 = l_int1;
    for(int n = 1; n < n_points-1;n++){ 
        l_int2 = points[n+1]-points[n];
        quad += 0.5*(l_int1+l_int2)*fvals[n];
        l_int1 = l_int2;
    }
    quad += 0.5*l_int2*fvals[n_points-1];

    return quad;
}
#endif
