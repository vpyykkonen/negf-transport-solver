#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <Eigen/LU>
#include <complex>
#include <cmath>
#include <vector>
#include <iomanip>
#include <functional>
#include "TwoTerminalSetup.h"
#include "Lead.h"
#include "ScatteringSystem.h"
#include "fd_dist.h"
#include "ScfMethod.h"
#include "ScfSolver.h"
#include "pade_frequencies.h"
#include "da2glob.h"
#include "config_parser.h"
#include "file_io.h"


using namespace Eigen;
using namespace std;

typedef complex<double> dcomp;

// Interface function
bool TwoTerminalSetup::change_parameters(string param_name, dcomp param_value)
{
    if(param_name == "U"){
        double U = real(param_value);
        this->ssyst.set_U(U);
        return true;
        // Temporarily, change both pots with same value due to restrictions in handling them separately
        // ToDo: Fix this when then data point generation issues are corrected
    } else if(param_name == "VCL"){
        this->set_VCL(real(param_value));
        this->set_VCR(real(param_value));
        return true;
    } else if(param_name == "VCR"){
        this->set_VCL(real(param_value));
        this->set_VCR(real(param_value));
        return true;
    } else if(param_name == "gate"){
        double gate = real(param_value);
        //this->ssyst.set_eps(-gate,-gate,true);
        this->set_gate(gate);
        return true;
    } else if(param_name == "bias"){
        this->set_bias(real(param_value));
        return true;
    } else if(param_name == "Tbias"){
        double Tbias = real(param_value);
        double minT = min(leadR.get_T(),leadL.get_T());
        if(Tbias < 0){
            this->leadR.set_T(minT);
            this->leadL.set_T(minT);
        }
        this->leadR.set_T(minT);
        this->leadL.set_T(minT+Tbias);
        return true;
        // ToDo: Enforce lead_config here
    } else if(param_name == "DeltaL"){
        dcomp DeltaL = param_value;
        this->leadL.set_Delta(DeltaL);
        return true;
    } else if(param_name == "DeltaR"){
        dcomp DeltaR = param_value;
        this->leadR.set_Delta(DeltaR);
        return true;
        // ToDo: Enfore equilibrium condition with temperatures
    } else if(param_name == "TL"){
        double TL = real(param_value);
        this->leadL.set_T(TL);
        return true;
    } else if(param_name == "TR"){
        double TR = real(param_value);
        this->leadR.set_T(TR);
        return true;
    } else if(param_name == "phaseL"){
        double phaseL = real(param_value);
        double absDeltaL = abs(this->leadL.get_Delta());
        dcomp DeltaL = absDeltaL*exp(1.0i*phaseL);
        this->leadL.set_Delta(DeltaL);
        return true;
    } else if(param_name == "phaseR"){
        double phaseR = real(param_value);
        double absDeltaR = abs(this->leadR.get_Delta());
        dcomp DeltaR = absDeltaR*exp(1.0i*phaseR);
        this->leadR.set_Delta(DeltaR);
        return true;
    } else
        return false;
}

tuple<MatrixXcd,MatrixXcd> TwoTerminalSetup::get_GR_and_Gl(double E,
        dcomp ieta)
{

    MatrixXcd I = MatrixXcd::Identity(2*n_sites_setup,2*n_sites_setup);

    // get lead Green's
    MatrixXcd gRLL,glLL;
    MatrixXcd gRRR,glRR;
    tie(gRLL,glLL) = this->leadL.get_gR_and_gl(E,ieta);
    tie(gRRR,glRR) = this->leadR.get_gR_and_gl(E,ieta);

    // Calculate Green's functions
    MatrixXcd gRSS,glSS;
    tie(gRSS,glSS) = this->ssyst.get_gR_and_gl(E,ieta);

    MatrixXcd gR = MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup);
    MatrixXcd gl = MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup);
    gR.block(0,0,2,2) = gRLL;
    gR.block(2*n_sites_setup-2,2*n_sites_setup-2,2,2) = gRRR;
    gR.block(2,2,2*n_sites,2*n_sites) = gRSS;
    gl.block(0,0,2,2) = glLL;
    gl.block(2*n_sites_setup-2,2*n_sites_setup-2,2,2) = glRR;
    gl.block(2,2,2*n_sites,2*n_sites) = glSS;


    MatrixXcd SigmaR = Sigma_leads + Sigma_int;


    MatrixXcd GR = (I-gR*SigmaR).partialPivLu().solve(gR);
    //MatrixXcd GR = (I-gR*SigmaR).colPivHouseholderQr().solve(gR);
    
    //MatrixXcd GA = GR.adjoint();
    MatrixXcd IpGRSigmaR = I + GR*SigmaR;
    MatrixXcd Gl = IpGRSigmaR*gl*IpGRSigmaR.adjoint();
    
    //MatrixXcd IpGRSigmaR_setupL = IpGRSigmaR.block(0,0,2*n_sites_setup,2);
    //MatrixXcd IpGRSigmaR_setupR = IpGRSigmaR.block(0,2*n_sites_setup-2,2*n_sites_setup,2);
   
    //MatrixXcd Gl = IpGRSigmaR_setupL*glLL*IpGRSigmaR_setupL.adjoint();
    //Gl += IpGRSigmaR_setupR*glRR*IpGRSigmaR_setupR.adjoint();
    return make_tuple(GR,Gl);
}

MatrixXcd TwoTerminalSetup::get_Gl(double E, dcomp ieta)
{
    vector<double>::iterator it;
    it = find_if(Es.begin(),Es.end(),
            [E](const double& x){return fabs(x-E)<1e-14*fabs(E);});
    if(it!=Es.end()){
        return Gls_freq[it-Es.begin()];
    }

    MatrixXcd GR,Gl;
    tie(GR,Gl) = get_GR_and_Gl(E,ieta);

    Es.push_back(E);
    Gls_freq.push_back(Gl);

    return Gl;
}

double TwoTerminalSetup::get_Gl_freq_elem(double E,
        int row, int col,
        dcomp ieta, bool get_real)
{
    if(get_real)
        return real(this->get_Gl(E,ieta)(row,col));
    else
        return imag(this->get_Gl(E,ieta)(row,col));

}

dcomp TwoTerminalSetup::get_Gl_time(int row,int col,
        dcomp ieta, double cutoff_below, double cutoff_above, double tol_quad)
{
    using namespace placeholders;
    static double pi = 3.141592653589793238462643383279502884197169;

    if(Gl_time(row,col) != numeric_limits<double>::infinity())
        return Gl_time(row,col);

    // divide interval into a couple of pieces
    int n_division_below = 7;
    int n_division_above = 3;
    int n_division = 0;
    double muR = leadR.get_muL();
    double muL = leadL.get_muL();
    bool finite_bias = abs(bias) > 1e-14;
    bool finite_DeltaL = abs(leadL.get_Delta()) > 1e-14;
    bool finite_DeltaR = abs(leadR.get_Delta()) > 1e-14;
    VectorXd concat_ints;
    if(finite_bias){
        VectorXd ints1;
        VectorXd ints2;
        VectorXd ints3;
        if(finite_DeltaR){
            ints1 = VectorXd::LinSpaced(n_division_below+1,cutoff_below,-bias);
            ints2 = VectorXd::LinSpaced(2,-bias,bias);
            ints3 = VectorXd::LinSpaced(n_division_above+1,bias,cutoff_above);
        } else {
            ints1 = VectorXd::LinSpaced(n_division_below+1,cutoff_below,min(muL,muR));
            ints2 = VectorXd::LinSpaced(2,min(muL,muR),max(muL,muR));
            ints3 = VectorXd::LinSpaced(n_division_above+1,max(muL,muR),cutoff_above);
        }
        n_division = n_division_below+n_division_above+1;
        concat_ints = VectorXd::Zero(n_division+1);
        concat_ints << ints1, ints3;
    } else if( finite_DeltaL || finite_DeltaR){
        double Delta_max = max(abs(leadL.get_Delta()),abs(leadR.get_Delta()));
        VectorXd ints1 = VectorXd::LinSpaced(n_division_below+1,cutoff_below,-Delta_max);
        VectorXd ints2 = VectorXd::LinSpaced(2,-Delta_max,Delta_max);
        VectorXd ints3 = VectorXd::LinSpaced(n_division_above+1,Delta_max,cutoff_above);
        n_division = n_division_below+n_division_above+1;
        concat_ints = VectorXd::Zero(n_division+1);
        concat_ints << ints1, ints3;
    } else {
        n_division = n_division_below + n_division_above;
        concat_ints = VectorXd::LinSpaced(n_division+1,cutoff_below,cutoff_above);
    }

    int nint = 0;
    double Gl_real = 0.0, Gl_imag = 0.0;
    Gl_time(row,col) = 0.0 + 0.0i;
    for(int i = 0; i < n_division; i++){
        VectorXd interval(2);
        interval << concat_ints(i), concat_ints(i+1);
        tie(Gl_real,nint) = da2glob(std::bind(
                    static_cast<double (TwoTerminalSetup::*)
                    (double,int,int,dcomp,bool)>
                    (&TwoTerminalSetup::get_Gl_freq_elem),
                    this,_1,row,col,ieta,true),
                    interval,tol_quad,tol_quad);
        tie(Gl_imag,nint) = da2glob(std::bind(
                    static_cast<double (TwoTerminalSetup::*)
                    (double,int,int,dcomp,bool)>
                    (&TwoTerminalSetup::get_Gl_freq_elem),
                    this,_1,row,col,ieta,false),
                    interval,tol_quad,tol_quad);
        Gl_time(row,col) += (Gl_real + 1.0i*Gl_imag)/(2*pi);
    }
    
    return Gl_time(row,col);
}

MatrixXcd TwoTerminalSetup::get_Gl_time(dcomp ieta,
       double cutoff_below, double cutoff_above, double tol_quad)
{
    // Make sure all the elements are calculated
    for(int n = 0; n < 2*n_sites_setup; n++)
        for(int m = 0; m < 2*n_sites_setup; m++)
            this->get_Gl_time(n,m,ieta,cutoff_below,cutoff_above,tol_quad);
    return Gl_time;
}

// Green's function on complex plane
// Retarded/Advanced function on (real E)+- ieta
// Matsubara/Pade on imaginary i omega_n
MatrixXcd TwoTerminalSetup::get_G(dcomp omega)
{

    //MatrixXcd HBdG = this->ssyst.get_HBdG();
    MatrixXcd gLL = this->leadL.get_g(omega);
    MatrixXcd gRR = this->leadR.get_g(omega);
    MatrixXcd gSS = this->ssyst.get_g(omega);
    MatrixXcd g = MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup);
    g.block(0,0,2,2) = gLL;
    g.block(2,2,2*n_sites,2*n_sites) = gSS;
    g.block(2*n_sites_setup-2,2*n_sites_setup-2,2,2) = gRR;

    MatrixXcd I = MatrixXcd::Identity(2*n_sites_setup,2*n_sites_setup);

    MatrixXcd Sigma = Sigma_leads + Sigma_int;

    MatrixXcd G = (I-g*Sigma).partialPivLu().solve(g);
    //MatrixXcd G = (I-gR*SigmaR).colPivHouseholderQr().solve(gR);

    return G;
}

MatrixXcd TwoTerminalSetup::get_Gl_time_Pade(int n_approx)
{
    double T = this->leadL.get_T();
    double freq = 0, residue = 0;

    if(Gl_time(0,0) != numeric_limits<double>::infinity())
        return Gl_time;

    MatrixXcd Gpp,Gpm;
    int np = 1;


    // Pade summation
    // Constant part 
    //double large = 1.0e10; // Large number after Ozaki PRB
    //Gpp = this->get_G(1i*large,Sigma); // positive imaginary
    ////dcomp number = 0.0;
    //Gl_time += -0.5*large*Gpp;
    // Constant part
    Gl_time = 0.5i*MatrixXcd::Identity(2*n_sites_setup,2*n_sites_setup);

    // Pade approximation pole sum 
    while(np <= n_approx){
        tie(freq,residue) = pade_frequency(np,n_approx);
        freq *= T;
        residue *= T;
        Gpp = this->get_G(1i*freq);
        //Gpm = this->get_G(-1i*freq,Sigma);
        Gpm = Gpp.adjoint(); // G(-i\omega) = G(i\omega)^\dagger
        // sum the positive and negative frequency parts
        Gl_time += -1.0i*residue*(Gpp+Gpm);
        np++; 
    }

    return Gl_time;
}

dcomp TwoTerminalSetup::get_Gl_time_Pade(int row, int col, int n_approx)
{
    if(Gl_time(0,0) != numeric_limits<double>::infinity())
        return Gl_time(row,col);
    get_Gl_time_Pade(n_approx);
    return Gl_time(row,col);
}



VectorXcd TwoTerminalSetup::update(const VectorXcd& X, dcomp ieta,
        double cutoff_below, double cutoff_above, double tol_quad)
{
    VectorXcd FX = VectorXcd::Zero(2*n_sites); // result vector
    
    // Self-energy for interaction
    Sigma_int = MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup); 
    for(int i = 0; i < n_sites; i++)
        Sigma_int.block(2*(i+1),2*(i+1),2,2) << real(X(i)) , X(n_sites+i) , conj(X(n_sites+i)) , -real(X(i));
    // Clear Green's functions due to change of self-energy
    clear_Greens();


    double hartree;
    dcomp pair;
    for(int i = 0; i < n_sites; i++){
        hartree = real(-1.0i*get_Gl_time(2*(i+1),2*(i+1),ieta,cutoff_below,cutoff_above,tol_quad));
        pair = -1.0i*get_Gl_time(2*(i+1),2*(i+1)+1,ieta,cutoff_below,cutoff_above,tol_quad);
        FX(i) = hartree;
        FX(n_sites+i) = pair;
    }
    //double hartree_avg = real(FX.head(n_sites).mean());
    //// subtract the average to keep positions of the peaks at the same
    //// as in the non-interacting case
    //for(int i = 0; i < n_sites; i++){
    //    FX(i) -= hartree_avg;
    //}
    FX *= this->ssyst.get_U();

    return FX;
}

VectorXcd TwoTerminalSetup::update_equilibrium(
        const VectorXcd& X, int n_approx)
{
    double U = this->ssyst.get_U();

    VectorXcd hartree(n_sites);
    VectorXcd delta(n_sites);
    VectorXcd FX = VectorXcd::Zero(2*n_sites);

    // Mean-field self-energy
    Sigma_int = MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup);
    for(int i=0; i < n_sites;i++){
        hartree(i) = real(X(i));
        delta(i) = X(i+n_sites);
        Sigma_int.block(2*(i+1),2*(i+1),2,2) << hartree(i) , delta(i) , conj(delta(i)) , -hartree(i);
    }

    clear_Greens(); // clear Green's due to change in self-energy
    get_Gl_time_Pade(n_approx);


    hartree = VectorXcd::Zero(n_sites);
    delta = VectorXcd::Zero(n_sites);

    for(int i = 0; i < n_sites; i++){
        hartree(i) = -1.0i*U*get_Gl_time_Pade(2*(i+1),2*(i+1),n_approx);
        delta(i) = -1.0i*U*get_Gl_time_Pade(2*(i+1),2*(i+1)+1,n_approx);
    }

    //dcomp hartree_mean = hartree.mean();
    for(int i = 0; i < n_sites; i++){
        FX(i) += hartree(i);//-hartree_mean;
        FX(n_sites+i) += delta(i);
    }

    return FX;
}
    
bool TwoTerminalSetup::self_consistent_loop(
        function<VectorXcd(const VectorXcd&)> FX,
        const VectorXd& Hartree0, const VectorXcd& Delta0,
        string scf_cfg_path, string save_path)
{
    if(abs(this->ssyst.get_U()) < 1e-6){
        cout << "Interaction strength small, self-energy zero" << endl;
        return true;
    }
    int n_sites = this->ssyst.get_n_sites();

    this->ssyst.set_Hartree_and_Delta(VectorXd::Zero(Hartree0.size()),VectorXcd::Zero(Delta0.size()));
    clear_Greens();

    VectorXcd X = VectorXcd::Zero(2*n_sites);
    X << Hartree0,Delta0;

    VectorXcd X0 = X; // Save initial guess
    ScfSolver solver(&X,scf_cfg_path,save_path);
    solver.iterate(FX);

    int total_iter = solver.get_iterations();
    bool converged = solver.get_converged();
        
    cout << "Scf loop converged: " << converged << "\n";
    cout << "Total iterations: " << total_iter << "\n";

    this->ssyst.set_Hartree_and_Delta(X.head(n_sites).real(),X.tail(n_sites));

    this->set_Sigma_int(MatrixXcd::Zero(2*n_sites_setup,2*n_sites_setup));

    return converged;
}


VectorXd TwoTerminalSetup::ParticleNumber(dcomp ieta,
       double cutoff_below, double cutoff_above, double tol)
{
    VectorXd nums = VectorXd::Zero(n_sites);
    for(int i = 0; i < n_sites; i++)
        nums(i) = real(-2.0i*get_Gl_time(2*(i+1),2*(i+1),ieta,cutoff_below,cutoff_above,tol));

    return nums;
}

double TwoTerminalSetup::TotalParticleNumber(dcomp ieta,
       double cutoff_below, double cutoff_above, double tol)
{
    return ParticleNumber(ieta,cutoff_below,cutoff_above,tol).sum();
}

VectorXcd TwoTerminalSetup::PairExpectation(dcomp ieta,
        double cutoff_below, double cutoff_above, double tol)
{
    VectorXcd pairs = VectorXcd::Zero(n_sites);
    for(int i = 0; i < n_sites; i++)
        pairs(i) = -1.0i*get_Gl_time(2*(i+1),2*(i+1)+1,ieta,cutoff_below,cutoff_above,tol);

    return pairs;
}

double TwoTerminalSetup::Current(dcomp ieta,
        double cutoff_below, double cutoff_above, double tol, int lead_idx, int direction)
{
    double cur = 0;
    if(lead_idx == 0)
        cur = -4.0*direction*real(tSL*get_Gl_time(0,2*(cpoint_L+1),ieta,cutoff_below,cutoff_above,tol));
    if(lead_idx == 1)
        cur = -4.0*direction*real(tSR*get_Gl_time(2*n_sites_setup-2,2*(cpoint_R+1),ieta,cutoff_below,cutoff_above,tol));
    return cur;
}

double TwoTerminalSetup::TotalParticleNumberEquilibrium(int n_approx)
{

    double number = 0.0;
    for(int i = 0; i < n_sites; ++i)
        number += real(-2.0i*get_Gl_time_Pade(2*(i+1),2*(i+1),n_approx));
    return real(number);
}

VectorXd TwoTerminalSetup::ParticleNumberEquilibrium(int n_approx)
{

    VectorXd pnum(n_sites);
    for(int i = 0; i < n_sites; ++i)
        pnum(i) = real(-2.0i*get_Gl_time_Pade(2*(i+1),2*(i+1),n_approx));
    return pnum;
}

VectorXcd TwoTerminalSetup::PairExpectationEquilibrium(int n_approx)
{

    VectorXcd pair(n_sites);
    for(int i = 0; i < n_sites; i++)
        pair(i) = -1.0i*get_Gl_time_Pade(2*(i+1),2*(i+1)+1,n_approx);
    return pair;
}



double TwoTerminalSetup::CurrentEquilibrium(int n_approx)
{
    return -4.0*real(tSL*get_Gl_time_Pade(0,2*(cpoint_L+1),n_approx));
}


