#ifndef SS_JUNCTION
#define SS_JUNCTION

#include <Eigen/Dense>
#include <Eigen/LU>
#include <complex>
#include <vector>
#include <tuple>
#include "Lead.h"
#include "ScatteringSystem.h"
#include "TwoTerminalSetup.h"
#include <Eigen/Sparse>



using namespace Eigen;
using namespace std;

typedef tuple<int,int,int> sp_state;
typedef complex<double> dcomp;
typedef tuple<int,sp_state,sp_state,dcomp> m_elem;
typedef SparseMatrix<dcomp> MatrixXcd_sp;
typedef Triplet<dcomp> T;
//ToDo: Think about how the energy is restricted from above
//Implement a different cutoff for above and below
//ToDo: make use of the independent branches of interconnected energies

// Class for handling the time-dependent superconducting-superconducting
// junction case.
// The difference is that the physics is time-dependent, in the harmonics of 
// the voltage.
// However, one can consider the physics via thinking of multiple copies of the same system, each representing different harmonics. The observables can be calculated by tracing over the copies in Green's functions. E.g. current is then a sum of independent currents of all the copies of the system.
// The copies are linked to each other at the connection points of the leads by the anomalous parts of the lead self-energies. Note that even and odd harmonics of the system are independent.
class SSJunction{
    private: 
    // Core parameters
    ScatteringSystem ssyst; // Scattering system
    double gate; // Gate potential
    double bias; // Chemical potential difference muL-muR
    Lead leadL; // Left lead
    Lead leadR; // Right lead
    double tSL; // Hopping amplitude between ssyst and leadL
    double tSR; // Hopping amplitude between ssyst and leadR
    int cpoint_L; // Contact point of left lead
    int cpoint_R; // Contact point of right lead
    double VCL; // Contact potential of left lead
    double VCR; // Contact potential of right lead
    double cutoff_below; // Below cutoff for energy
    double cutoff_above; // Above cutoff for energy
    int n_harmonics; // Number of harmonics of the base frequenc with observables


    // Auxiliary parameters:
    double base_frequency;
    int n_coupled_freqs;// Number of harmonics of base frequency considered 
    int n_sites_system;
    int n_sites_setup;

    // Bogoliubov de Gennes Hamiltonian in frequency basis
    // Blocks of size 2*n_sites x 2*n_sites correspond to different discrete frequency components of the Hamiltonian
    // each block is represented in the Nambu basis
    MatrixXcd_sp HBdG;

    // Lead-system contact perturbation
    MatrixXcd_sp Hpert; // Perturbation Hamiltonian in the Nambu-harmonics basis
    // Interaction self energy
    MatrixXcd_sp Sigma;

    // ----- Containers for calculated Green's  -----
    // Used to store spectral values for integration
    // To be emptied after update of parameters
    vector<double> Es; // Es corresponding to Gls_freq
    vector<MatrixXcd> Gls_freq; // Lesser Green's function in cross harmonic form for different frequencies
    vector<vector<MatrixXcd>> Glns_freq; // Lesser Green's functions at different harmonics for different frequencies
    vector<MatrixXcd> Glns_time; // Lesser Green's functions at equal time for different harmonics
    //vector<tuple<int,sp_state,sp_state>> calculated_elements;
    //

    // ----- Observables -----
    // Pair expectation and particle number in frequency basis for different sites and harmonics
    // row = site index,
    // column = harmonic index from negative to positive in increasing order
    MatrixXcd ParticleNumber;
    MatrixXcd PairExpectation;
    // Harmonics of the current
    VectorXcd Current;


    // Index vectors for get submatrices and certain blocks
    vector<VectorXi> freq_type_idx;
    vector<VectorXi> system_part_idx;


    public:
    //SSJunction():ssyst{},leadL{},leadR{},tSL{0.0},tSR{0.0},cpoint_L{0},cpoint_R{0},cutoff_energy{0}, n_harmonics{0}{n_coupled_freqs = 0; n_freqs=0;bias=leadL.get_muL()-leadR.get_muL();}
    SSJunction(ScatteringSystem ssyst, double gate, double bias,
            Lead leadL, Lead leadR, double tSL, double tSR,
            int cpoint_L, int cpoint_R, double VCL, double VCR,
            double below_cutoff,double above_cutoff, int n_harmonics);
    //SSJunction(ScatteringSystem ssys, Lead l1, Lead l2, double tsl, double tsr, int cpoint_l, int cpoint_r, double cutoff_energy, int n_harmonics, MatrixXcd ParticleNumber, MatrixXcd PairExpectation);
    ~SSJunction() {}

    void calculate_Hpert();
    void calculate_Sigma();
    void initialize();

    bool change_parameters(string param_name, dcomp value);

    void clear_Green_containers();
    void clear_Observable_containers();

    // ToDo: fix these ssyst/lead setter functions so that they 
    // won't cause paramete clashes on dependent variables
    //void set_ssyst(const ScatteringSystem& ssyst){this->ssyst = ssyst;clear_Green_containers();}
    //void set_leadL(const Lead& leadL){this->leadL = leadL; Es.clear();clear_Green_containers();}
    //void set_leadR(const Lead& leadR){this->leadR = leadR;clear_Green_containers();}
    void set_tSL(const double& tSL){this->tSL = tSL;clear_Green_containers();}
    void set_tSR(const double& tSR){this->tSR = tSR;clear_Green_containers();}
    void set_cpoint_L(const int& cpoint_L){this->cpoint_L = cpoint_L;clear_Green_containers();}
    void set_cpoint_R(const int& cpoint_R){this->cpoint_R = cpoint_R;clear_Green_containers();}
    void set_n_harmonics(const int& n_harmonics){this->n_harmonics = n_harmonics; }
    void set_bias(const double& bias);
    void set_gate(const double& gate){
        this->gate = gate;
        this->ssyst.set_gate(gate);
    }
    void set_VCL(const double& VCL){
        this->VCL = VCL;
        this->ssyst.set_local_V(this->VCL,this->cpoint_L);
    }
    void set_VCR(const double& VCR){
        this->VCR = VCR;
        this->ssyst.set_local_V(this->VCR,this->cpoint_R);
    }

    ScatteringSystem get_ssyst(){return this->ssyst;}
    Lead get_leadL(){return this->leadL;}
    Lead get_leadR(){return this->leadR;}
    double get_tSL(){return this-> tSL;}
    double get_tSR(){return this-> tSR;}
    int get_cpoint_L(){return this-> cpoint_L;}
    int get_cpoint_R(){return this-> cpoint_R;}
    double get_VCL(){return this-> VCL;}
    double get_VCR(){return this-> VCR;}
    int get_n_coupled_freqs(){return n_coupled_freqs;}
    int get_n_sites_setup(){return n_sites_setup;}
    int get_n_sites_system(){return n_sites_system;}
    int get_n_harmonics(){return n_harmonics;}
    double get_bias(){return this-> bias;}
    double get_gate(){return this->gate;}

    void set_PairExpectation_and_ParticleNumber(const MatrixXcd& Pair, const MatrixXcd& Number,bool reset_Greens);


    tuple<MatrixXcd_sp,MatrixXcd_sp> get_gR_and_gl(double E, dcomp ieta);
    //tuple<MatrixXcd_sp,MatrixXcd_sp>       split_ch_matrix_to_branches(const MatrixXcd_sp& ch_mat);
    //MatrixXcd merge_branches(const MatrixXcd& b1, const MatrixXcd& b2);
    
    vector<MatrixXcd_sp> split_ch_matrix_to_branches(const MatrixXcd_sp& ch_mat,const vector<vector<int>>& con);
    MatrixXcd merge_branches(const vector<MatrixXcd>& b1, const vector<vector<int>>& con);

    tuple<MatrixXcd,MatrixXcd> get_GR_and_Gl(double E, dcomp ieta, const MatrixXcd_sp& Sigma);
    MatrixXcd get_Gl_freq(double E, dcomp ieta, const MatrixXcd_sp& Sigma);
    MatrixXcd get_harmonic_trace(const MatrixXcd& m, int harmonic);
    MatrixXcd get_Gln_freq(double E, int harmonic,dcomp ieta,const MatrixXcd_sp& Sigma_int);
    double get_Gln_freq(double E, int harmonic, sp_state state1, sp_state state2,dcomp ieta, const MatrixXcd_sp& Sigma,bool real);
    int sp_state_to_idx(sp_state state);
    sp_state idx_to_sp_state(int idx);
    int harmonic_and_sp_state_to_idx(int harmonic, sp_state state);
    dcomp get_Gln_time(int harmonic, sp_state sp_state1, sp_state sp_state2, dcomp ieta,const MatrixXcd_sp& Sigma, double tol);
    //MatrixXcd& get_Gln_time(int harmonic){ return Glns_time[harmonic];}
    vector<MatrixXcd>& get_Glns_time(dcomp ieta, double tol);

    // ---- Self-consistent functions ----
    VectorXcd update(const VectorXcd& X, dcomp ieta, double tol_quad);
    bool self_consistent_loop(function<VectorXcd(const VectorXcd&)> FX,const MatrixXcd& Hartree0,const MatrixXcd& Delta0, string scf_path = "scf_params.cfg",string save_path = "");


    // Real time, store in the Object
    void CalculateParticleNumber(dcomp ieta, double tol);
    dcomp CalculateParticleNumber(int site, int harmonic, dcomp ieta, double tol);

    void CalculatePairExpectation(dcomp ieta, double tol);
    dcomp CalculatePairExpectation(int site, int harmonic, dcomp ieta, double tol);

    void CalculateCurrent(dcomp ieta, double tol,int lead_idx, int direction);

    dcomp CalculateCurrent(dcomp ieta, int harmonic, double tol, int lead_idx, int direction);

    // get functions for observables and HBdG
    MatrixXcd get_HBdG(){return HBdG;}
    MatrixXcd_sp get_Hpert(){return Hpert;}
    MatrixXcd_sp get_Sigma(){return Sigma;}
    dcomp get_PairExpectation(int site, int harmonic){return PairExpectation(site,2*abs(harmonic)+max(0,harmonic/abs(harmonic)));}
    MatrixXcd get_PairExpectation(){return PairExpectation;}
    dcomp get_ParticleNumber(int site, int harmonic)
    {
        if(harmonic >= 0)
            return ParticleNumber(site,harmonic);
        else
            return conj(ParticleNumber(site,-harmonic));
    }
    MatrixXcd get_ParticleNumber(){return ParticleNumber;}
    dcomp get_Current(int harmonic){
        if(harmonic >= 0)
            return Current(harmonic);
        else
            return conj(Current(-harmonic));
    }
    VectorXcd get_Current(){return Current;}
};


#endif
