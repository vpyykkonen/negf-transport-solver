#ifndef SCATTERINGSYSTEM_H
#define SCATTERINGSYSTEM_H

#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <tuple>
#include <vector>

#include "da2glob.h"

using namespace Eigen;
using namespace std;

typedef complex<double> dcomp;

class ScatteringSystem{
    private:
    // Essential
    MatrixXcd H0; // Single particle Hamiltonian without potentials
    VectorXcd Delta; // Superconducting order parameter
    VectorXd Hartree; // Hartree potential
    VectorXd local_potential; // Local potentials
    VectorXd disorder;
    double U; // Interaction strength
    double T; // Temperature
    double gate; // Gate potential

    // lattice properties
    int dim; // dimension
    int quasi_dim; // embedding dimension of the lattice
    vector<VectorXd> lattice; // Lattice vectors
    int unitcell_sites; // Number of sites in unit cell
    vector<VectorXd> basis; // Basis vectors of unit cell
    vector<int> n_unitcells; // number of unit cells in each dimension


    // Storage tight binding parameters for convenience
    vector<dcomp> on_site;
    vector<VectorXi> target_cells;
    vector<int> source_orbitals;
    vector<int> target_orbitals;
    vector<dcomp> hoppings;
    string edge_removal_str;

    // Auxiliary
    vector<double> Es;
    vector<MatrixXcd> gRs;
    vector<MatrixXcd> gls;
    int n_sites;
    bool parameters_changed;
    MatrixXcd HBdG; 
    MatrixXcd W; // eigenvectors of HBdG
    VectorXd eigE; // eigenvalues of HBdG

    public:
    ScatteringSystem(){n_sites = 1;  H0 = MatrixXcd::Identity(2,2); Delta = VectorXcd::Zero(2); Hartree = VectorXd::Zero(2); HBdG = MatrixXcd::Identity(4,4); HBdG(1,1) = -1; HBdG(3,3) = -1; W = MatrixXcd::Identity(4,4); eigE = VectorXd::Ones(4); eigE(0)=-1.0; eigE(1) = -1.0;  U=0; T= 0;}
    ScatteringSystem(MatrixXcd h0, VectorXcd delta , VectorXd hartree, double U, double Tsys);

ScatteringSystem(int dim, int unitcell_sites, string n_unitcells_str, string on_site_str, int n_hoppings, vector<string> hopping_strs, string edge_removal_str , double U, double T);
    ScatteringSystem(string geometry_path,  double U, double T);
    ScatteringSystem(string Hamiltonian_path, VectorXd hartree, VectorXcd Delta, double U, double T);


    ScatteringSystem(const ScatteringSystem& S);

    ~ScatteringSystem(){}

    void geom_strs_to_params(const string& on_site_str, const string& n_unitcells_str, const int& n_hoppings, const vector<string>& hopping_strs);
    void compile_Hamiltonian();
    void calculate_HBdG();

    int get_n_sites(){return this->n_sites;}
    double get_T(){return this->T;}
    MatrixXcd get_H0(){return this->H0;}
    VectorXcd get_Delta(){return this->Delta;}
    VectorXd get_Hartree(){return this->Hartree;}
    MatrixXcd get_HBdG()
    {
        if(parameters_changed)
           calculate_HBdG();
        return this->HBdG;
    }
    MatrixXcd get_W(){return this->W;}
    MatrixXcd get_eigE(){return this->eigE;}
    double get_U(){return this->U;}
    double get_gate(){return this->gate;}



    void clear_Greens(){Es.clear();gRs.clear();gls.clear();}

    void set_T(const double T){this->T = T; clear_Greens();}
    void set_U(const double U){this->U = U;}
    void set_H0(const MatrixXcd H0);
    void set_Delta(const VectorXcd Delta);
    void set_Hartree(const VectorXd Hartree);
    void set_Hartree_and_Delta(const VectorXd& Hartree, const VectorXcd& Delta);
    void set_eps(int site,const double eps);
    void set_local_V(const double& V, const int& site);
    void add_disorder(double dev, unsigned int seed);
    void set_gate(const double gate);

    VectorXcd update_scf(const VectorXcd& X);
    VectorXcd get_Delta_from_HBdG();
    VectorXd get_Hartree_from_HBdG();
    tuple<VectorXd,VectorXcd> get_Hartree_and_Delta_from_HBdG();
    bool self_consistent_loop(const VectorXd& Hartree0, const VectorXcd& Delta0, const string scf_cfg_path, const string save_path = "");

    tuple<MatrixXcd,MatrixXcd> get_gR_and_gl(double E, dcomp ieta);
    MatrixXcd get_Gl();
    MatrixXcd get_g(dcomp omega);
    tuple<MatrixXcd,MatrixXcd> get_non_nambu_gR_and_gl(double E, dcomp ieta);

    double ParticleNumberSpectral(double E, dcomp ieta);
    double ParticleNumberGreen(dcomp ieta);
    double ParticleNumber(int site);
    double ParticleNumber();
    dcomp PairCorrelator(int site);
    
};

#endif
