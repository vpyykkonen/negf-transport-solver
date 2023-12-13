#ifndef OBSERVABLE_H
#define OBSERVABLE_H

#include <iostream>
#include <complex>
#include <vector>
#include <tuple>
#include <functional>
#include <algorithm>

#include <Eigen/SparseCore>

using namespace Eigen;
using namespace std;

typedef complex<double> dcomp;
typedef Eigen::SparseMatrix<dcomp> SpMatrixXcd;
typedef tuple<int,int,int> sp_state; // single particle index tuple
typedef tuple<int,sp_state,sp_state,dcomp> m_elem; // matrix element

class Observable{
    private:
        int n_terms;
        // single particle state indies
        // tuple<int,int,int>, system part index, site index, spin index
        // Number of possibilities in each index labeling single-particle states in the observable
        int n_harmonics;
        int n_syst_parts; // System part index
        vector<int> n_sites; // Site indices for each system part
        int n_spin; // Particle/Hole + Spin index, always fixed in each parts of system

        vector<int> nz_harmonics;
        int n_nz_harmonics;
        int n_system; // Number of different single site indices


        // List of the non-zero matrix elements
        vector<m_elem> matrix_elements;

    public:
        Observable();

        Observable(int n_sites_syst, int n_harmonics);

        Observable(int n_harmonics,int n_syst_parts, vector<int> n_sites, int n_spin);

        // Add matrix element for single time dependent operator
        void add_matrix_element(int harmonic, sp_state state1,sp_state state2, dcomp element);


        // Add matrix element for time-independent operator
        void add_matrix_element(sp_state state1,sp_state state2, dcomp element);

        int get_n_terms(){return n_terms;}
        void set_n_harmonics(int n_harm){this->n_harmonics = n_harm;}

        void set_n_sites_part(int part, int n_sites_part);

        // Get expectation value of time-independent operator/observable
        dcomp get_expectation_value(function<dcomp(const sp_state, const sp_state)> Gl_time);

        // Get n=harmonic coeffiecient of the observable in <A(t)> = \sum_n <A>_n exp(in\omega_0 t)
        dcomp get_expectation_value(function<dcomp(const int, const sp_state, const sp_state)> Gl_time, int harmonic);


        // return the index of a single particle state in the default indexing scheme
        int get_index(sp_state state);

        // return the matrix related to a harmonic
        SpMatrixXcd get_matrix(int harmonic);

        // return the cross-harmonic matrix
        SpMatrixXcd get_cross_harmonic_matrix();
};


#endif
