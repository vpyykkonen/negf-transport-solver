#include <iostream>
#include <complex>
#include <vector>
#include <tuple>
#include <functional>
#include <algorithm>

#include <Eigen/SparseCore>
#include "Observable.h"

using namespace Eigen;
using namespace std;

Observable::Observable()
{
    n_syst_parts = 3;
    n_spin = 2;
    n_sites.push_back(1);
    n_sites.push_back(1);
    n_sites.push_back(1);
    n_terms = 0;
    n_harmonics = 0;
    n_system = 0;
    n_nz_harmonics = 0;
    for(int n = 0; n < n_syst_parts; n++)
        n_system += n_spin*n_sites[n];
}

Observable::Observable(int n_sites_syst, int n_harmonics)
    :n_harmonics{n_harmonics}
{
    n_syst_parts = 3;
    n_spin = 2;
    n_sites.push_back(1);
    n_sites.push_back(n_sites_syst);
    n_sites.push_back(1);
    n_terms = 0;
    n_system = 0;
    for(int n = 0; n < n_syst_parts; n++)
        n_system += n_spin*n_sites[n];
    n_nz_harmonics = 0;
}

Observable::Observable(int n_harmonics,int n_syst_parts, vector<int> n_sites, int n_spin)
    :n_harmonics{n_harmonics}, n_syst_parts{n_syst_parts}, n_sites{n_sites}, n_spin{n_spin}
{ 
    n_terms = 0;
    n_nz_harmonics = 0;
    n_system = 0;
    for(int n = 0; n < n_syst_parts; n++)
        n_system += n_spin*n_sites[n];
}

// Add matrix element for single time dependent operator
void Observable::add_matrix_element(int harmonic, sp_state sp_state1,sp_state sp_state2, dcomp element)
{
    bool sp_state1_ok = get<0>(sp_state1) < n_syst_parts && sp_state1 < make_tuple(n_syst_parts,n_sites[get<0>(sp_state1)],n_spin) && sp_state1 > make_tuple(-1,-1,-1);
    bool sp_state2_ok = get<0>(sp_state2) < n_syst_parts && sp_state2 < make_tuple(n_syst_parts,n_sites[get<0>(sp_state2)],n_spin) && sp_state2 > make_tuple(-1,-1,-1);
    bool harmonics_ok = harmonic <= n_harmonics && harmonic >= -n_harmonics;

    if(sp_state1_ok && sp_state2_ok && harmonics_ok){
        m_elem matrix_element = make_tuple(harmonic,sp_state1,sp_state2,element);
        matrix_elements.push_back(matrix_element);
        n_terms++;
        auto it = find(nz_harmonics.begin(),nz_harmonics.end(),harmonic);
        if(it == nz_harmonics.end()){
            nz_harmonics.push_back(harmonic);
            n_nz_harmonics++;
        }
    } else {
        cout << "Matrix element indices out of bounds" << endl;
    }

}


// Add matrix element for time-independent operator
void Observable::add_matrix_element(sp_state sp_state1,sp_state sp_state2, dcomp element)
{
    bool sp_state1_ok = get<0>(sp_state1) < n_syst_parts && sp_state1 < make_tuple(n_syst_parts,n_sites[get<0>(sp_state1)],n_spin) && sp_state1 > make_tuple(-1,-1,-1);
    bool sp_state2_ok = get<0>(sp_state2) < n_syst_parts && sp_state2 < make_tuple(n_syst_parts,n_sites[get<0>(sp_state2)],n_spin) && sp_state2 > make_tuple(-1,-1,-1);

    if(sp_state1_ok && sp_state2_ok){
        m_elem matrix_element = make_tuple(0,sp_state1,sp_state2,element);
        matrix_elements.push_back(matrix_element);
        n_terms++;
        auto it = find(nz_harmonics.begin(),nz_harmonics.end(),0);
        if(it == nz_harmonics.end()){
            nz_harmonics.push_back(0);
            n_nz_harmonics++;
        }
    } else {
        cout << "Matrix element indices out of bounds" << endl;
    }
}

void Observable::set_n_sites_part(int part, int n_sites_part)
{
    this->n_sites[part] = n_sites_part;
    n_system= 0;
    for(int n = 0; n < n_syst_parts; n++)
        n_system += n_spin*n_sites[n];
}

// Get expectation value of time-independent operator/observable
dcomp Observable::get_expectation_value(function<dcomp(const sp_state, const sp_state)> Gl_time){
    dcomp output = 0.0 +0.0i;
    dcomp Gl;
    for(int i = 0; i < n_terms; i++){
        m_elem m_element = matrix_elements[i];
        // inverse order due to  definition of Gl
        Gl = Gl_time(get<2>(m_element),get<1>(m_element));
        output += -1.0i*get<3>(m_element)*Gl;
    }
    return output;
}

// Get n=harmonic coeffiecient of the observable in <A(t)> = \sum_n <A>_n exp(in\omega_0 t)
dcomp Observable::get_expectation_value(function<dcomp(const int, const sp_state, const sp_state)> Gl_time, int harmonic){
    dcomp output = 0.0 +0.0i;
    dcomp Gl;
    for(int i = 0; i < n_terms; i++){
        m_elem m_element = matrix_elements[i];
        int obs_harmonics = get<0>(m_element);
        int Gl_harmonic = harmonic-obs_harmonics;
        Gl = Gl_time(Gl_harmonic,get<2>(m_element),get<1>(m_element));
        output += -1.0i*get<3>(m_element)*Gl;
    }
    return output;
}


// return the index of a single particle state in the default indexing scheme
int Observable::get_index(sp_state state){
    int block_first = 0;
    int idx = 0;
    //cout << "sp_state:" << endl;
    //cout << get<0>(state) <<  " " << get<1>(state) << " " << get<2>(state) << endl;
    //cout << n_spin << endl;
    for(int i = 0; i < get<0>(state); i++)
        block_first += n_spin*n_sites[i];
    //cout << block_first << endl;
    idx = block_first + n_spin*get<1>(state) + get<2>(state);
    //cout << "idx: " << idx << endl;
    return idx;
}

// Get the matrix of the operator related to a harmonic
SpMatrixXcd Observable::get_matrix(int harmonic){
    SpMatrixXcd obs_n(n_system,n_system);
    for(int n = 0; n < n_terms; n++){
        m_elem m_element = matrix_elements[n];
        if(get<0>(m_element) == harmonic){
            int row = get_index(get<1>(m_element));
            int col = get_index(get<2>(m_element));
            obs_n.insert(row,col) = get<3>(m_element);
        }
    }
    obs_n.makeCompressed();
    return obs_n;
}

// Get the cross-harmonic matrix
SpMatrixXcd Observable::get_cross_harmonic_matrix(){
    int n_freqs = 2*n_harmonics + 1;
    SpMatrixXcd obs_nm(n_system*n_freqs,n_system*n_freqs);
    for(int n = 0; n < n_nz_harmonics; n++){
        int harmonic = nz_harmonics[n];
        int first_row = max(-harmonic,0);
        int last_row = min(n_freqs-harmonic,n_freqs);
        for(int m = first_row; m < last_row; m++){
            for(int n = 0; n < n_terms; n++){
                m_elem m_element = matrix_elements[n];
                if(get<0>(m_element) == harmonic){
                    int row = m*n_system + get_index(get<1>(m_element));
                    int col = (m+harmonic)*n_system +get_index(get<2>(m_element));
                    obs_nm.insert(row,col) = get<3>(m_element);
                }
            }
        }
    }
    return obs_nm;
}


