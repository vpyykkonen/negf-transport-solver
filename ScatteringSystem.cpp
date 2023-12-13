#include "ScatteringSystem.h"

#include <complex>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
//#include "kron.h"
#include <Eigen/LU>
#include "fd_dist.h"
#include "ScfMethod.h"
#include "ScfSolver.h"
#include "pade_frequencies.h"
#include "file_io.h"
#include "config_parser.h"


#include <iostream>


using namespace std;
using namespace Eigen;

typedef complex<double> dcomp;


ScatteringSystem::ScatteringSystem(MatrixXcd H0, VectorXcd Delta , VectorXd Hartree, double U, double T)
    : H0{H0}, Delta{Delta}, Hartree{Hartree}, U{U}, T{T}
{ 
    n_sites = H0.cols();

    gate = 0.0;
    local_potential = VectorXd::Zero(n_sites);
    disorder = VectorXd::Zero(n_sites);
    HBdG = MatrixXcd::Zero(2*n_sites,2*n_sites);
    parameters_changed = true;
}

ScatteringSystem::ScatteringSystem(int dim, int unitcell_sites, string n_unitcells_str, string on_site_str, int n_hoppings, vector<string> hopping_strs, string edge_removal_str , double U, double T)
    : U{U},T{T},dim{dim},unitcell_sites{unitcell_sites}, edge_removal_str{edge_removal_str} 
{

    geom_strs_to_params(on_site_str,n_unitcells_str,n_hoppings,hopping_strs);
    compile_Hamiltonian();
    n_sites = H0.cols();

    gate = 0.0;
    Delta = VectorXcd::Zero(n_sites);
    Hartree = VectorXd::Zero(n_sites);
    local_potential = VectorXd::Zero(n_sites);
    disorder = VectorXd::Zero(n_sites);
    parameters_changed = true;
}

ScatteringSystem::ScatteringSystem(string geometry_path, double U, double T)
    : U{U},T{T}
{
    
    map<string,string> geometry_config = load_config_file(geometry_path, "=");
    dim = stoi(geometry_config["dim"]);
    //quasi_dim = stoi(geometry_config["quasi_dim"]);

    unitcell_sites = stoi(geometry_config["unitcell_sites"]);
    int n_hoppings = stoi(geometry_config["n_hoppings"]);
    
    string on_site_str = geometry_config["on_site"];
    string n_unitcells_str = geometry_config["n_unitcells"];

    vector<string> hopping_strs;
    for(int i = 0; i < n_hoppings; i++){
        string hopping_str = geometry_config["hopping"+to_string(i+1)];
        hopping_strs.push_back(hopping_str);
    }

    edge_removal_str = geometry_config["edge"];

    geom_strs_to_params(on_site_str,n_unitcells_str,n_hoppings, hopping_strs);
    compile_Hamiltonian();
    n_sites = H0.cols();

    gate = 0.0;
    Delta = VectorXcd::Zero(n_sites);
    Hartree = VectorXd::Zero(n_sites);
    local_potential = VectorXd::Zero(n_sites);
    disorder = VectorXd::Zero(n_sites);
    parameters_changed = true;

}

void ScatteringSystem::geom_strs_to_params(const string& on_site_str, const string& n_unitcells_str, const int& n_hoppings, const vector<string>& hopping_strs)
{
    n_unitcells.clear();
    string tmp = n_unitcells_str;
    // number of unit cells in each dimension
    for(int i = 0; i < dim-1; i++){
        size_t comma_pos = tmp.find(",");
        int n_unitcell = stoi(tmp.substr(0,comma_pos));
        n_unitcells.push_back(n_unitcell);
        tmp.erase(tmp.begin(),tmp.begin()+comma_pos+1);
    }
    n_unitcells.push_back(stoi(tmp));
    

    // on-site energies
    // number equals to unitcell_sites 
    //vector<dcomp> on_site;
    on_site.clear();
    tmp = on_site_str;
    for(int i = 0; i < unitcell_sites-1; i++){
        size_t comma_pos = tmp.find(",");
        istringstream is(tmp.substr(0,comma_pos));
        dcomp eps; is >> eps;
        on_site.push_back(eps);
        tmp.erase(tmp.begin(),tmp.begin()+comma_pos+1);
    }
    istringstream is(tmp);
    dcomp eps; is >> eps;
    on_site.push_back(eps);
    
    // hopping amplitude configuration
    target_cells.clear();
    target_orbitals.clear();
    source_orbitals.clear();
    hoppings.clear();
    for(int i = 0; i < n_hoppings; i++){
        string hopping_str = hopping_strs[i];
        size_t sep_pos = hopping_str.find("=");
        string target_source_str = hopping_str.substr(0,sep_pos);

        string hopping_amp_str = hopping_str.substr(sep_pos+1);

        // Find the target unit cell
        size_t semicolon_pos = target_source_str.find(";");
        string target_cell_str = target_source_str.substr(0,semicolon_pos);
        VectorXi target_cell = VectorXi::Zero(dim);
        for(int j = 0; j < dim-1;j++){
            size_t comma_pos = target_cell_str.find(","); 
            int target_cell_j = stoi(target_cell_str.substr(0,comma_pos));
            target_cell(j) = target_cell_j;
            target_cell_str.erase(target_cell_str.begin(),target_cell_str.begin()+comma_pos+1);
        }
        target_cell(dim-1) = stoi(target_cell_str);

        target_source_str.erase(target_source_str.begin(),target_source_str.begin()+semicolon_pos+1);

        // find the oribital indices of hopping
        istringstream is = istringstream(target_source_str);
        char character;
        int source_orbital;
        int target_orbital;

        is >> source_orbital >> character >> target_orbital;

        // find the hopping amplitude
        is = istringstream(hopping_amp_str);
        dcomp hopping_amp;
        is >> hopping_amp;


        target_cells.push_back(target_cell);
        source_orbitals.push_back(source_orbital);
        target_orbitals.push_back(target_orbital);
        hoppings.push_back(hopping_amp);
    }
    parameters_changed = true;
}

void ScatteringSystem::compile_Hamiltonian()
{
    int n_hoppings = hoppings.size();
    // Compile the Hamiltonian matrix
    int total_unitcells = 1;
    VectorXi n_unitcell_prods = VectorXi::Ones(dim);
    for(int i = 0; i < dim; i++){
        for(int j = i+1; j < dim; j++)
            n_unitcell_prods(j) *= n_unitcells[i];
        total_unitcells *= n_unitcells[i];
    }
    n_sites = total_unitcells*unitcell_sites;

    // Calculate the relative positions of target cells of hoppings
    // in terms of the linear indicing of the unit cells
    vector<int> target_cells_rel;
    for(int i = 0; i < n_hoppings; i++){
        VectorXi target_cell = target_cells[i];
        int target_cell_rel = 0;
        for(int j = 0; j < dim; j++)
            target_cell_rel += target_cell(j)*n_unitcell_prods(j);
        target_cells_rel.push_back(target_cell_rel);
    }

    H0 = MatrixXcd::Zero(n_sites,n_sites);

    for(int i = 0; i < total_unitcells; i++){
        for(int j = 0; j < unitcell_sites; j++){
            H0(unitcell_sites*i+j,unitcell_sites*i+j) = on_site[j];
        }
        for(int j = 0; j < n_hoppings; j++){
            int target_cell_idx_p = i + target_cells_rel[j];
            int target_cell_idx_m = i - target_cells_rel[j];
            // if the target cell is the same unit cell, add both directions
            // otherwise, add only the elements where the source is the current unit cell
            if( target_cells_rel[j] == 0){
                H0(i*unitcell_sites+target_orbitals[j],i*unitcell_sites+source_orbitals[j]) =hoppings[j];
                if(target_orbitals[j] != source_orbitals[j])
                    H0(i*unitcell_sites+source_orbitals[j],i*unitcell_sites+target_orbitals[j]) =conj(hoppings[j]);
            }else{
                if( target_cell_idx_p >= 0 && target_cell_idx_p < total_unitcells)
                    H0(target_cell_idx_p*unitcell_sites+target_orbitals[j],i*unitcell_sites+source_orbitals[j]) = hoppings[j];
                if( target_cell_idx_m >= 0 && target_cell_idx_m < total_unitcells)
                    H0(target_cell_idx_m*unitcell_sites+source_orbitals[j],i*unitcell_sites+target_orbitals[j]) = conj(hoppings[j]);
            }
        }

    }

    stringstream stream1(edge_removal_str);
    string elem1;
    string elem2;
    vector<int> removed_sites;
    int i = 0;
    while(stream1.good()){
        getline(stream1,elem1,';');
        if(elem1.empty()){
            i++;
            continue;
        }
        stringstream stream2(elem1);
        int edge_dim = i/2;
        int edge_sign = i%2;

        int edge_size = 1; // Variable to contain edge size
        vector<int> edge_dims;
        vector<int> edge_side_prods;
        int prod = 1;
        edge_side_prods.push_back(prod);
        for(int j = 0; j < dim; j++){
            if( j != i/2){
                prod *= n_unitcells[j];
                edge_side_prods.push_back(prod);
                edge_size *= n_unitcells[j];
                edge_dims.push_back(j);
            }
        }

        while(stream2.good()){
            getline(stream2,elem2,',');
            int removed_item = stoi(elem2);
            for(int j = 0; j < edge_size; j++){
                int site_idx = 0;
                if(edge_sign == 1)
                    site_idx += (n_unitcells[edge_dim]-1)*n_unitcell_prods[edge_dim];
                for(int k = 0; k < dim-1; k++){
                    site_idx += (j%edge_side_prods[k+1])/edge_side_prods[k]*n_unitcell_prods[edge_dims[k]];
                }
                site_idx *= unitcell_sites;
                site_idx += removed_item;
                removed_sites.push_back(site_idx);
                //int n_rows = H0.rows();
                //int n_cols = H0.cols();
                //
                //H0.block(site_idx,0,n_rows-1-site_idx,n_cols) = H0.block(site_idx+1,0,n_rows-1-site_idx,n_cols);
                //H0.conservativeResize(n_rows-1,n_cols);

                //n_rows = H0.rows();
                //n_cols = H0.cols();

                //H0.block(0,site_idx,n_rows,n_cols-1-site_idx) = H0.block(0,site_idx+1,n_rows,n_cols-1-site_idx);
                //H0.conservativeResize(n_rows,n_cols-1);
                //n_sites -= 1;

            }
        }
        i++;
    }
    sort(removed_sites.begin(),removed_sites.end());
    auto ip = unique(removed_sites.begin(),removed_sites.end());
    removed_sites.resize(distance(removed_sites.begin(),ip));

    int n_removed = removed_sites.size();

    for(auto it = removed_sites.begin(); it != removed_sites.end(); ++it)
        cout << *it << " " ;
    cout << endl;

    MatrixXcd H0_resized = MatrixXcd::Zero(n_sites-n_removed,n_sites-n_removed);

    int k = 0;
    for(int i = 0; i < n_sites;i++){
        int l = 0;
        if(find(removed_sites.begin(),removed_sites.end(), i) == removed_sites.end()){
            for(int j = 0; j < n_sites;j++){
                if(find(removed_sites.begin(),removed_sites.end(), j) == removed_sites.end()){
                    H0_resized(k,l) = H0(i,j);
                    l++;
                }
            }
            k++;
        }
    }
    n_sites -= n_removed;

    H0 = H0_resized;

    cout << H0 << endl;

    HBdG = MatrixXcd::Zero(2*n_sites,2*n_sites);
    parameters_changed = true;
}

ScatteringSystem::ScatteringSystem(string Hamiltonian_path, VectorXd hartree, VectorXcd Delta, double U, double T)
    : U{U}, T{T}
{
    H0 = readComplexMatrix(Hamiltonian_path.c_str());
    n_sites = H0.rows();

    // Not a lattice in general. 
    // Only assumption that interaction is local
    // Assume that the Hamiltonian describes only one
    // unit cell
    dim = 0;
    quasi_dim = 0;
    unitcell_sites = n_sites;
    gate = 0.0;

    Delta = VectorXcd::Zero(n_sites);
    Hartree = VectorXd::Zero(n_sites);
    local_potential = VectorXd::Zero(n_sites);
    disorder = VectorXd::Zero(n_sites);
    
    HBdG = MatrixXcd::Zero(2*n_sites,2*n_sites);
    parameters_changed = true;
}


ScatteringSystem::ScatteringSystem(const ScatteringSystem& S)
{
    n_sites = S.n_sites;
    H0 = S.H0;
    Delta = S.Delta;
    Hartree = S.Hartree;
    HBdG = S.HBdG;
    W = S.W;
    eigE = S.eigE;
    U=S.U;
    T= S.T;
    Es=S.Es;
    gRs=S.gRs;
    gls=S.gls;
    gate = S.gate;
    local_potential = S.local_potential;
    disorder = S.disorder;

    dim = S.dim;
    unitcell_sites = S.unitcell_sites;
    n_unitcells = S.n_unitcells;

    on_site = S.on_site;
    target_orbitals = S.target_orbitals;
    source_orbitals = S.source_orbitals;
    hoppings = S.hoppings;
    edge_removal_str = S.edge_removal_str;

    parameters_changed =true;
}

void ScatteringSystem::calculate_HBdG()
{
    this->HBdG = MatrixXcd::Zero(2*n_sites,2*n_sites);
    for(int site = 0; site < n_sites;++site){
        HBdG(2*site,2*site) = H0(site,site) + local_potential(site);
        HBdG(2*site,2*site) += Hartree(site) + disorder(site) - gate;
        HBdG(2*site+1,2*site+1) = -HBdG(2*site,2*site);
        HBdG(2*site,2*site+1) = Delta(site);
        HBdG(2*site+1,2*site) = conj(Delta(site));
        for(int targ = 0; targ < n_sites; ++targ){
            if(targ != site){
                HBdG(2*site,2*targ) = H0(site,targ);
                HBdG(2*site+1,2*targ+1) = -conj(H0(site,targ));
            }
        }
    }
}

void ScatteringSystem::set_H0(const MatrixXcd H0) 
{
    this->H0 = H0;
    this->n_sites = H0.cols();
    parameters_changed = true;
}

void ScatteringSystem::set_Delta(const VectorXcd Delta)
{
    this->Delta = Delta;
    parameters_changed = true;
}

void ScatteringSystem::set_Hartree(const VectorXd Hartree)
{
    this->Hartree = Hartree;
    parameters_changed = true;
}

void ScatteringSystem::set_Hartree_and_Delta(const VectorXd& Hartree,const VectorXcd& Delta)
{
    this->set_Hartree(Hartree);
    this->set_Delta(Delta);
    parameters_changed = true;
}


void ScatteringSystem::set_eps(int site, const double eps)
{
    this->H0(site,site) = eps; // A site
    parameters_changed = true;
}


void ScatteringSystem::set_local_V(const double& V, const int& site)
{
    if(site < 0 || site >= n_sites){
        cerr << "site index out of bounds" << endl;
        return;
    }
    local_potential(site) = V;
    parameters_changed = true;
}

void ScatteringSystem::add_disorder(double dev, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> d{0.0,dev};
    disorder = VectorXd::Zero(n_sites);
    for(int site = 0; site < n_sites; ++site)
        disorder(site) = d(gen);
    parameters_changed = true;
}


void ScatteringSystem::set_gate(const double gate)
{
    this->gate = gate;
    parameters_changed = true;

}

MatrixXcd ScatteringSystem::get_g(dcomp omega)
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }

    MatrixXcd WD(W.rows(),W.cols());
    for(int i = 0; i < W.cols(); i++)
        WD.col(i) = 1.0/(omega - eigE(i))*W.col(i);

    return WD*W.adjoint();
}

tuple<MatrixXcd,MatrixXcd> ScatteringSystem::get_gR_and_gl(double E, dcomp ieta)
{
    //vector<double>::iterator it;
    //it = find_if(Es.begin(),Es.end(),[E](const double& x){ return abs(E-x) < 1.0e-10;});
    //if(it != Es.end())
    //    return make_tuple(gRs[it-Es.begin()],gls[it-Es.begin()]);
    //static MatrixXcd gR = MatrixXcd::Identity(2*n_sites,2*n_sites);
    //static MatrixXcd gl = MatrixXcd::Identity(2*n_sites,2*n_sites);
    //MatrixXcd I = MatrixXcd::Identity(2*n_sites,2*n_sites);

    MatrixXcd gR = get_g(E+ieta);
    
    //MatrixXcd inv_g = ((E+ieta)*I-HBdG);
    //gR = inv_g.colPivHouseholderQr().solve(I);
    double fd = fd_dist(E,0, this->T);
    MatrixXcd gl = fd*(gR.adjoint()-gR);

    //Es.push_back(E);
    //gRs.push_back(gR);
    //gls.push_back(gl);

    return make_tuple(gR,gl);
}

MatrixXcd ScatteringSystem::get_Gl()
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    VectorXcd fd_eigE = VectorXcd::Zero(2*n_sites);
    for(int n = 0; n < 2*n_sites; n++)
        fd_eigE(n) = fd_dist(eigE(n),0.0,T);
    return 1.0i*W*fd_eigE.asDiagonal()*W.adjoint();
}

tuple<MatrixXcd,MatrixXcd> ScatteringSystem::get_non_nambu_gR_and_gl(double E, dcomp ieta)
{
    MatrixXcd EI = (E+ieta)*MatrixXcd::Identity(this->n_sites,this->n_sites);
    MatrixXcd ginv = EI-this->get_H0();
    MatrixXcd gR = ginv.inverse();

    double fd = fd_dist(E,0,this->T);
    MatrixXcd gl = fd*(gR.adjoint()-gR);

    return make_tuple(gR,gl);
}

VectorXcd ScatteringSystem::get_Delta_from_HBdG()
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    int n_sites = W.cols()/2;
    dcomp uij, vij;
    VectorXcd delta_HBdG = VectorXcd::Zero(n_sites);
    for(int i = 0; i < n_sites; i++){
        for(int j = 0; j < n_sites; j++){
            uij = W(2*i,n_sites+j);
            vij = W(2*i + 1,n_sites+j);
            delta_HBdG(i) += uij*conj(vij)*tanh(eigE(j)/(2*T));
        }
    }
    delta_HBdG *= -U;
    return delta_HBdG;
}
VectorXd ScatteringSystem::get_Hartree_from_HBdG()
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    int n_sites = W.cols()/2;
    dcomp uij,vij;
    VectorXd hart_HBdG = VectorXd::Zero(n_sites);
    for(int i = 0; i < n_sites; i++){
        for(int j = 0; j < n_sites; j++){
            uij = W(2*i,n_sites+j);
            vij = W(2*i + 1,n_sites+j);
            hart_HBdG(i) += real(uij*conj(uij)*fd_dist(eigE(j),0.0,T));
            hart_HBdG(i) += real(vij*conj(vij)*fd_dist(-eigE(j),0.0,T));
        }
    }
    hart_HBdG *= U;
    return hart_HBdG;
}

tuple<VectorXd,VectorXcd> ScatteringSystem::get_Hartree_and_Delta_from_HBdG()
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    int n_sites = W.cols()/2;
    dcomp uij,vij;
    VectorXd hart_HBdG = VectorXd::Zero(n_sites);
    VectorXcd delta_HBdG = VectorXcd::Zero(n_sites);
    double fd_mE,fd_pE;//,tanhE;
    for(int j = 0; j < n_sites; j++){
        fd_pE = fd_dist(eigE(n_sites+j),0.0,T);
        fd_mE = fd_dist(-eigE(n_sites+j),0.0,T);
        //tanhE = tanh(eigE(n_sites+j)/(2*T));
        for(int i = 0; i < n_sites; i++){
            uij = W(2*i,n_sites+j);
            vij = W(2*i + 1,n_sites+j);
            hart_HBdG(i) += real(uij*conj(uij)*fd_pE);
            hart_HBdG(i) += real(vij*conj(vij)*fd_mE);
            //delta_HBdG(i) += -uij*conj(vij)*tanhE;
            delta_HBdG(i) += uij*conj(vij)*(2*fd_pE-1);
        }
    }
    delta_HBdG *= U;
    hart_HBdG *= U;
    return make_tuple(hart_HBdG,delta_HBdG);
}


VectorXcd ScatteringSystem::update_scf(const VectorXcd& X)
{
    int n_sites = X.size()/2;
    VectorXcd FX(X.size());
    this->set_Hartree_and_Delta(X.head(n_sites).real(),X.tail(n_sites));
    VectorXcd Hartree, Delta;
    tie(Hartree,Delta) = this->get_Hartree_and_Delta_from_HBdG();
    FX << Hartree,Delta;
    return FX;
}


bool ScatteringSystem::self_consistent_loop(const VectorXd& Hartree0,
        const VectorXcd& Delta0,const string scf_cfg_path, const string save_path)
{
    using namespace placeholders;
    if(abs(this->get_U()) < 1e-6){
        cout << "Interaction strength small, self-energy zero" << endl;
        return true;
    }
    int n_sites = this->get_n_sites();

    VectorXcd X = VectorXcd::Zero(2*n_sites);
    X << Hartree0,Delta0;
    // Save initial guess
    VectorXcd X0 = X;

    ScfSolver solver(&X,scf_cfg_path,save_path);
    solver.iterate(std::bind(
                    static_cast<VectorXcd(ScatteringSystem::*)
                    (const VectorXcd&)>
                    (&ScatteringSystem::update_scf),
                    this,_1));

    int total_iter = solver.get_iterations();
    bool converged = solver.get_converged();
        
    cout << "Scf loop converged: " << converged << "\n";
    cout << "Total iterations: " << total_iter << "\n";

    this->set_Hartree_and_Delta(X.head(n_sites).real(),X.tail(n_sites));

    return converged;
}

double ScatteringSystem::ParticleNumberSpectral(double E, dcomp ieta)
{
    int n_sites_syst = this->n_sites;

    MatrixXcd gR, gl;
    tie(gR,gl) = this->get_gR_and_gl(E, ieta);

    double particles = 0;
    for(int i = 0; i < n_sites_syst; i++){
        particles += imag(gl(2*i,2*i));
    }

    return particles;
}

double ScatteringSystem::ParticleNumberGreen(dcomp ieta)
{
    using namespace placeholders;
    double tol = 1.0e-6;
    double lim1 = eigE.minCoeff()-abs(0.5*eigE.minCoeff());
    double lim2 = eigE.maxCoeff()+abs(0.15*eigE.minCoeff());
    VectorXd ints(2);

    //ints(0) = min(lim1,lim2)-3.0*max(TL,TR);
    //ints(1) = max_mu+3.0*max(TL,TR);
    //
    ints(0) = lim1;
    ints(1) = lim2;
    //ints(0) = -30;
    //ints(1) = 30;
    
    static double pi = 3.141592653589793238462643383279502884197169;
    dcomp number;
    int nint;
    tie(number,nint) = da2glob(std::bind(static_cast<double (ScatteringSystem::*)(double,dcomp)>(&ScatteringSystem::ParticleNumberSpectral),this,_1,ieta),ints,tol,tol);
    number /= 2*pi; // remember 2 pi :)
    cout << nint << endl;

    return real(number);
}

double ScatteringSystem::ParticleNumber(int site)
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    int n_sites = W.cols()/2;
    try{
        if(site >= n_sites || site < 0)
            throw(site);
    } 
    catch (int site){
        cout << "ScatteringSystem::ParticleNumber: site index " << site << "off bounds." << "\n";
        return 0.0;
    }

    dcomp uij,vij;
    double pnum = 0.0, fd_mE = 0.0,fd_pE = 0.0;
    for(int j = 0; j < n_sites; j++){ // loop over quasiparticles
        fd_pE = fd_dist(eigE(n_sites+j),0.0,T);
        fd_mE = fd_dist(-eigE(n_sites+j),0.0,T);
        uij = W(2*site,n_sites+j);
        vij = W(2*site+1,n_sites+j);
        pnum += real(uij*conj(uij)*fd_pE);
        pnum += real(vij*conj(vij)*fd_mE);
    }
    return pnum;
}

double ScatteringSystem::ParticleNumber()
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    dcomp uij,vij;
    VectorXd hart_HBdG = VectorXd::Zero(n_sites);
    double fd_mE,fd_pE;
    for(int j = 0; j < n_sites; j++){ // loop over quasiparticles
        fd_pE = fd_dist(eigE(n_sites+j),0.0,T);
        fd_mE = fd_dist(-eigE(n_sites+j),0.0,T);
        for(int i = 0; i < n_sites; i++){ // loop over particles
            uij = W(2*i,n_sites+j);
            vij = W(2*i + 1,n_sites+j);
            hart_HBdG(i) += real(uij*conj(uij)*fd_pE);
            hart_HBdG(i) += real(vij*conj(vij)*fd_mE);
        }
    }
    return hart_HBdG.sum();
}

dcomp ScatteringSystem::PairCorrelator(int site)
{
    if(parameters_changed){
        this->calculate_HBdG();
        SelfAdjointEigenSolver<MatrixXcd> es;
        es.compute(HBdG,ComputeEigenvectors);
        W = es.eigenvectors();
        eigE = es.eigenvalues();
        parameters_changed = false;
    }
    try{
        if(site >= n_sites || site < 0)
            throw(site);

    } 
    catch (int site){
        cout << "ScatteringSystem::PairCorrelator: site index " << site << "off bounds." << "\n";
        return 0.0;
    }

    dcomp pair = 0.0+0.0i, uij,vij;
    double fd_pE = 0.0;
    for(int j = 0; j < n_sites; j++){ // loop over quasiparticles
        fd_pE = fd_dist(eigE(n_sites+j),0.0,T);
        uij = W(2*site,n_sites+j);
        vij = W(2*site+1,n_sites+j);
        pair += uij*conj(vij)*(2*fd_pE-1);
    }
    return pair;
}


