#ifndef TWOTERMINALSETUP_H
#define TWOTERMINALSETUP_H

#include <Eigen/Dense>
#include <Eigen/LU>
#include <complex>
#include <vector>
#include <tuple>
#include "Lead.h"
#include "ScatteringSystem.h"


using namespace Eigen;
using namespace std;

typedef complex<double> dcomp;

// ToDo: Add label indicating the lead configuration to the
// TwoTerminalSetup class to avoid bugs (not an urgent issue)
// ToDo: Somehow make the separation of SS,SN,NS,NN 
// in and out of equilibrium more clear. Presently confusing.
// Ideas: have a virtual/abstract class TwoTerminalSetup
// and specialize diffent types of setups. 
// Equilibrium is always handled the same way so it could be
// its own.  Then SN,NS,NN are handled the same way out of equilibrium and SS out of equilibrium is it's own thing.
// Actually, time-independent and periodic time-dependcy are 
// the differences. Perhaps this could be used somehow to
// advantage to actually be able to understand also 
// periodically biased junctions (not urgent right now).
class TwoTerminalSetup{
    private:
    ScatteringSystem ssyst; // Scattering system
    double gate;
    double bias; // Chemical potential difference muL-muR
    Lead leadL; // Left lead
    Lead leadR; // Right lead
    double tSL; // Hopping amplitude between ssyst and leadL
    double tSR; // Hopping amplitude between ssyst and leadR
    int cpoint_L; // Contact point for leadL
    int cpoint_R; // Contact point for leadR
    double VCL;  // Contact potential for leadL
    double VCR; // Contact potential for leadR
                
    // Auxiliary variables
    int n_sites;
    int n_sites_setup;

    // Containers for calculated Green's 
    // Used to store spectral values for integration
    // To be emptied after update of parameters
    vector<MatrixXcd> Gls_freq; // Lesser Green's functions 
    //vector<MatrixXcd> GRs; // Retarded Green's functions
    vector<double> Es; // Energy values where calculated
    MatrixXcd Gl_time; // Container for calculated elements of Gl_time

    MatrixXcd Sigma_int;
    MatrixXcd Sigma_leads;



    public:
    //TwoTerminalSetup(){ssyst = ScatteringSystem(); leadL = Lead(); leadR = Lead(); tSL = 1; tSR =1;}
    TwoTerminalSetup(ScatteringSystem ssyst, double gate, double bias,
            Lead leadL, Lead leadR,
            double tSL, double tSR, 
            int cpoint_L, int cpoint_R,
            double VCL, double VCR
    ):
        ssyst{ssyst},gate{gate}, bias{bias},leadL{leadL},leadR{leadR},tSL{tSL},tSR{tSR},
        cpoint_L{cpoint_L},cpoint_R{cpoint_R},VCL{VCL},VCR{VCR}
    {   
        this->leadL.set_muL(this->bias);
        this->leadR.set_muL(0.0);
        this->ssyst.set_gate(this->gate);
        this->ssyst.set_local_V(this->VCL,this->cpoint_L);
        this->ssyst.set_local_V(this->VCR,this->cpoint_R);

        n_sites = this->ssyst.get_n_sites();
        n_sites_setup = n_sites + 2;

        Gl_time = MatrixXcd::Constant(2*n_sites_setup,
                2*n_sites_setup,numeric_limits<double>::infinity());

        Sigma_int = MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup);

        Sigma_leads = MatrixXcd::Zero(2*n_sites_setup,
                2*n_sites_setup);

        Sigma_leads(0,2*(cpoint_L+1)) = conj(tSL);
        Sigma_leads(1,2*(cpoint_L+1)+1) = -tSL;
        Sigma_leads(2*(cpoint_L+1),0) = tSL;
        Sigma_leads(2*(cpoint_L+1)+1,1) = -conj(tSL);

        Sigma_leads(2*n_sites_setup-2,2*(cpoint_R+1)) = conj(tSR);
        Sigma_leads(2*n_sites_setup-1,2*(cpoint_R+1)+1) = -tSR;
        Sigma_leads(2*(cpoint_R+1),2*n_sites_setup-2) = tSR;
        Sigma_leads(2*(cpoint_R+1)+1,2*n_sites_setup-1) = -conj(tSR);
    }

    ~TwoTerminalSetup(){}

    bool change_parameters(string param_name, dcomp value);

    void clear_Greens()
    {
        Es.clear();
        Gls_freq.clear();
        Gl_time = MatrixXcd::Constant(2*n_sites_setup,
                2*n_sites_setup,numeric_limits<double>::infinity());
    }

    // Don't set ssyst or leads by reference 
    // to avoid clashes of different dependent parameters
    //void set_ssyst(const ScatteringSystem ssyst){this->ssyst = ssyst;clear_Greens();}
    //void set_leadL(const Lead leadL){this->leadL = leadL; Es.clear();clear_Greens();}
    //void set_leadR(const Lead leadR){this->leadR = leadR;clear_Greens();}
    void set_tSL(const double& tSL){
        this->tSL = tSL;
        Sigma_leads(0,2*(cpoint_L+1)) = conj(tSL);
        Sigma_leads(1,2*(cpoint_L+1)+1) = -tSL;
        Sigma_leads(2*(cpoint_L+1),0) = tSL;
        Sigma_leads(2*(cpoint_L+1)+1,1) = -conj(tSL);
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }
    void set_tSR(const double& tSR){
        this->tSR = tSR;
        Sigma_leads(2*n_sites_setup-2,2*(cpoint_R+1)) = conj(tSR);
        Sigma_leads(2*n_sites_setup-1,2*(cpoint_R+1)+1) = -tSR;
        Sigma_leads(2*(cpoint_R+1),2*n_sites_setup-2) = tSR;
        Sigma_leads(2*(cpoint_R+1)+1,2*n_sites_setup-1) = -conj(tSR);
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }
    void set_cpoint_L(const int& cpoint_L){
        Sigma_leads(0,2*(cpoint_L+1)) = 0.0;
        Sigma_leads(1,2*(cpoint_L+1)+1) = 0.0;
        Sigma_leads(2*(cpoint_L+1),0) = 0.0;
        Sigma_leads(2*(cpoint_L+1)+1,1) = 0.0;

        this->cpoint_L = cpoint_L;
        Sigma_leads(0,2*(cpoint_L+1)) = conj(tSL);
        Sigma_leads(1,2*(cpoint_L+1)+1) = -tSL;
        Sigma_leads(2*(cpoint_L+1),0) = tSL;
        Sigma_leads(2*(cpoint_L+1)+1,1) = -conj(tSL);
        if(Es.size() != 0)
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }
    void set_cpoint_R(const int& cpoint_R){
        Sigma_leads(2*n_sites_setup-2,2*(cpoint_R+1)) = 0.0;
        Sigma_leads(2*n_sites_setup-1,2*(cpoint_R+1)+1) = 0.0;
        Sigma_leads(2*(cpoint_R+1),2*n_sites_setup-2) = 0.0;
        Sigma_leads(2*(cpoint_R+1)+1,2*n_sites_setup-1) = 0.0;

        this->cpoint_R = cpoint_R;
        Sigma_leads(2*n_sites_setup-2,2*(cpoint_R+1)) = conj(tSR);
        Sigma_leads(2*n_sites_setup-1,2*(cpoint_R+1)+1) = -tSR;
        Sigma_leads(2*(cpoint_R+1),2*n_sites_setup-2) = tSR;
        Sigma_leads(2*(cpoint_R+1)+1,2*n_sites_setup-1) = -conj(tSR);

        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }
    void set_gate(const double& gate){
        this->gate = gate;
        this->ssyst.set_gate(gate);
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }

    // enforce convention that the right lead is kept at zero chemical potential
    void set_bias(const double& bias)
    {
        leadL.set_muL(bias);
        leadR.set_muL(0.0);
        this-> bias = bias;
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }
    void set_VCL(const double& VCL){
        this->VCL = VCL;
        ssyst.set_local_V(VCL,this->cpoint_L);
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    };
    void set_VCR(const double& VCL){
        this->VCR = VCR;
        ssyst.set_local_V(VCR,this->cpoint_R);
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    };

    void set_Sigma_int(const MatrixXcd& Sigma_int){
        this->Sigma_int = Sigma_int;
        if(Es.size() != 0 && Gl_time(0,0) == numeric_limits<double>::infinity())
            clear_Greens();
    }

    // Don't return references from
    // get functions to avoid clashes
    // since many parameters are interdependent
    ScatteringSystem get_ssyst(){return this->ssyst;}
    Lead get_leadL(){return this->leadL;}
    Lead get_leadR(){return this->leadR;}
    double get_tSL(){return this-> tSL;}
    double get_tSR(){return this-> tSR;}
    double get_gate(){return this->gate;}
    double get_bias(){return this-> bias;}
    int get_cpoint_L(){return this-> cpoint_L;}
    int get_cpoint_R(){return this-> cpoint_R;}
    double get_VCL(){return this->VCL;}
    double get_VCR(){return this->VCR;}

    // Green's functions:
    //
    // ---- Green's functions in complex frequency plane ----
    // It gives the following Green's with specific values
    // E + ieta -> retarded
    // E - ieta -> advanced
    // (imaginary) Pade frequencies -> Pade 
    // (imaginary) Matsubara frequenices -> Matsubara
    MatrixXcd get_G(dcomp omega);
    tuple<MatrixXcd,MatrixXcd> get_GR_and_Gl(double E, dcomp ieta);
    MatrixXcd get_Gl(double E, dcomp ieta);

    // Real time
    double get_Gl_freq_elem(double E, int row, int col,
        dcomp ieta, bool get_real);
    dcomp get_Gl_time(int m, int n, dcomp ieta, 
            double cutoff_below,double cutoff_above, double tol);
    MatrixXcd get_Gl_time(dcomp ieta,  
            double cutoff_below,double cutoff_above, double tol);

    // Real time -- Equilibrium, finite temperature only
    MatrixXcd get_Gl_time_Pade(int n_approx);
    dcomp get_Gl_time_Pade(int row, int col, int n_approx);

    // ---- Self-consistent functions ----
    VectorXcd update(const VectorXcd& X, dcomp ieta, 
            double cutoff_below, double cutoff_above, double tol_quad);
    VectorXcd update_equilibrium(const VectorXcd& X, int n_approx);
    bool self_consistent_loop(function<VectorXcd(const VectorXcd&)> FX,
            const VectorXd& Hart0,const VectorXcd& Delta0,
            string scf_path = "scf_params.cfg", string save_path = "");



    VectorXd ParticleNumber(dcomp ieta,
            double cutoff_below, double cutoff_above, double tol);
    double TotalParticleNumber(dcomp ieta, 
            double cutoff_below, double cutoff_above, double tol);
    VectorXcd PairExpectation(dcomp ieta,
            double cutoff_below, double cutoff_above, double tol);
    double Current(dcomp ieta, 
            double cutoff_below, double cutoff_above,
            double tol, int lead_idx, int direction);

    VectorXd ParticleNumberEquilibrium(int n_approx);
    double TotalParticleNumberEquilibrium(int n_approx);
    VectorXcd PairExpectationEquilibrium(int n_approx);
    double CurrentEquilibrium(int n_approx);

};

#endif 
