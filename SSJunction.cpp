#include <iostream>
#include <tuple>
#include <complex>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include "Observable.h"
#include "SSJunction.h"
#include "Lead.h"
#include "ScatteringSystem.h"
#include "fd_dist.h"
#include "ScfMethod.h"
#include "ScfSolver.h"
#include "pade_frequencies.h"
#include "da2glob.h"
#include "config_parser.h"
#include "trapezoid.h"
#include "file_io.h"
#include "connectivity.h"

SSJunction::SSJunction(ScatteringSystem ssyst,
        double gate, double bias,
        Lead leadL, Lead leadR, double tSL, double tSR,
        int cpoint_L, int cpoint_R, double VCL, double VCR,
        double cutoff_below, double cutoff_above, int n_harmonics):
    ssyst{ssyst},gate{gate},bias{bias},leadL{leadL},leadR{leadR},
    tSL{tSL},tSR{tSR},cpoint_L{cpoint_L},cpoint_R{cpoint_R},
    VCL{VCL},VCR{VCR}, cutoff_below{cutoff_below}, cutoff_above{cutoff_above},
    n_harmonics{n_harmonics}
{
    // Check if bias is too small
    if(bias < 0.001*abs(this->leadL.get_Delta())){
        bias = 1.0;
    }
    // Due to our approach, leads have zero chemical potential
    // The bias is conveyed in the time dependence of the hoppings
    this->leadL.set_muL(0.0);
    this->leadR.set_muL(0.0); // Enforce convention
    this->ssyst.set_gate(this->gate);
    this->ssyst.set_local_V(this->VCL,this->cpoint_L);
    this->ssyst.set_local_V(this->VCR,this->cpoint_R);

    base_frequency = abs(bias);
    n_coupled_freqs = (int)ceil(abs(cutoff_above-cutoff_below)/base_frequency);

    n_sites_system = ssyst.get_n_sites();
    n_sites_setup = ssyst.get_n_sites()+2; 

    n_harmonics = max(n_harmonics,
            4*(int)ceil(abs(leadL.get_Delta())/base_frequency));

    // 
    if(n_harmonics > n_coupled_freqs)
        n_harmonics = n_coupled_freqs;

    ParticleNumber = MatrixXcd::Zero(n_sites_system,n_harmonics+1);
    PairExpectation = MatrixXcd::Zero(n_sites_system,2*n_harmonics+1);
    Current = VectorXcd::Zero(n_harmonics+1);


    initialize();

}


// Thought: how to make this more general? 
// If one would consider one of the leads to be at same chemical potential
// as the scattering system, then here the change is simple: on that hopping
// the hopping is at constant frequency
//
// Note: the chemical potential differences are taken into account
// in the lead-system hoppings
// The hopping direction is opposite to the chemilal potential gradient:
// since if chemical potential decreases, the one with lower chemical potential
// has to be raised in energy in comparison
// Thus, e.g. tSL has components n+1,n as S has lower chemical potential than L
void SSJunction::calculate_Hpert()
{
    Hpert = MatrixXcd_sp(2*n_sites_setup*n_coupled_freqs,2*n_sites_setup*n_coupled_freqs);
    vector<T> Hpert_triplets;
    // Cross-harmonic matrix for time-dependent perturbation. Check notes for details
    for(int n = 0; n < n_coupled_freqs; n++){
        // Help indices
        int n_freq_block = 2*n_sites_setup*n;
        int np1_freq_block = 2*n_sites_setup*(n+1);

        // L,R,cpoint_L,cpoint_R indices for both n and n+1 harmonics
        int nL_idx = n_freq_block;
        int np1L_idx = np1_freq_block;
        int nR_idx = np1_freq_block-2;
        //int np1R_idx = nR_idx + 2*n_sites_setup;
        int nS_L_idx = n_freq_block + 2*(cpoint_L+1);
        int np1S_L_idx = np1_freq_block + 2*(cpoint_L+1);
        int nS_R_idx = n_freq_block + 2*(cpoint_R+1);
        //int np1S_R_idx = np1_freq_block + 2*(cpoint_R+1);


        // Left lead connection ( only non-zero terms shown)
        if(n < n_coupled_freqs - 1){
            Hpert_triplets.push_back(
                    T(np1S_L_idx,nL_idx, // HSL, up up, n+1,n
                        tSL)); 
            Hpert_triplets.push_back(
                    T(nS_L_idx+1,np1L_idx+1, // HSL down down n,n+1
                        -conj(tSL))); 
            Hpert_triplets.push_back(
                    T(nL_idx,np1S_L_idx, // HLS, up up, n,n+1
                        conj(tSL))); 
            Hpert_triplets.push_back(
                    T(np1L_idx+1,nS_L_idx+1, // HLS, down, down, n+1,n
                        -tSL)); 
        }

        // Right lead connections, at equilibrium with scattering system
        Hpert_triplets.push_back(
                T(nS_R_idx,nR_idx, // HSR, up up, n,n
                    tSR));
        Hpert_triplets.push_back(
                T(nS_R_idx+1,nR_idx+1, // HSR down, down n,n
                    -conj(tSR)));
        Hpert_triplets.push_back(
                T(nR_idx,nS_R_idx, // HRS,up,up n,n
                    conj(tSR))); 
        Hpert_triplets.push_back(
                T(nR_idx+1,nS_R_idx+1,// HRS,down,down, n,n
                    -tSR)); 
    }
    Hpert.setFromTriplets(Hpert_triplets.begin(),Hpert_triplets.end());
    Hpert.makeCompressed();

}

// Calculates Sigma based on the stored particle number 
// and pair expectation data
void SSJunction::calculate_Sigma()
{
    Sigma = MatrixXcd_sp(2*n_coupled_freqs*n_sites_setup,2*n_coupled_freqs*n_sites_setup);
    if( abs(ssyst.get_U()) < 1e-6 )
        return;
    double U = this->ssyst.get_U();
    vector<T> Sigma_triplets;

    for(int n = 0; n < n_harmonics +1; n++){
        int last_row = n_coupled_freqs-n;
        for(int row = 0; row < last_row-1; row++){
            int n_block = 2*n_sites_setup*row;
            int mp_block = 2*n_sites_setup*(row+n);
            //int mn_block = 2*n_sites_setup*(row-n);
            for(int i = 0; i < n_sites_system; i++){
                int ni_idx = n_block + 2*(i+1);
                int mpi_idx = mp_block + 2*(i+1);
                if( n == 0){
                    Sigma_triplets.push_back(
                            T(ni_idx,mpi_idx,
                                0.5*U*ParticleNumber(i,0)));
                    Sigma_triplets.push_back(
                            T(ni_idx,mpi_idx+1,
                                U*PairExpectation(i,0)));
                    Sigma_triplets.push_back(
                            T(ni_idx+1,mpi_idx,
                                U*conj(PairExpectation(i,0))));
                    Sigma_triplets.push_back(
                            T(ni_idx+1,mpi_idx+1,
                                -0.5*U*conj(ParticleNumber(i,0))));
                } else {
                    // Upper triangular part
                    Sigma_triplets.push_back(
                            T(ni_idx,mpi_idx,
                                0.5*U*ParticleNumber(i,n)));
                    Sigma_triplets.push_back(
                            T(ni_idx,mpi_idx+1,
                                U*PairExpectation(i,2*n-1)));
                    Sigma_triplets.push_back(
                            T(ni_idx+1,mpi_idx,
                                U*conj(PairExpectation(i,2*n))));
                    Sigma_triplets.push_back(
                            T(ni_idx+1,mpi_idx+1,
                                -0.5*U*conj(ParticleNumber(i,n))));

                    // Add hermitian conjugates, lower triangular part
                    Sigma_triplets.push_back(
                            T(mpi_idx,ni_idx,
                                0.5*U*conj(ParticleNumber(i,n))));
                    Sigma_triplets.push_back(
                            T(mpi_idx,ni_idx+1,
                                U*PairExpectation(i,2*n)));
                    Sigma_triplets.push_back(
                            T(mpi_idx+1,ni_idx,
                                U*conj(PairExpectation(i,2*n-1))));
                    Sigma_triplets.push_back(
                            T(mpi_idx+1,ni_idx+1,
                                -0.5*U*ParticleNumber(i,n)));
                }

            }
        }
    }
    Sigma.setFromTriplets(Sigma_triplets.begin(),Sigma_triplets.end());
    Sigma.makeCompressed();
}

void SSJunction::clear_Green_containers(){
    Es.clear(); 
    Glns_freq.clear(); 
    Gls_freq.clear(); 
    Glns_time.clear(); 

    MatrixXcd init_matrix = MatrixXcd::Constant(
            2*n_sites_setup,2*n_sites_setup,
            numeric_limits<double>::infinity());

    // Put zero matrices to Glns_time
    for(int n = 0; n < n_coupled_freqs + 1; n++)
        Glns_time.push_back(init_matrix);
}

void SSJunction::clear_Observable_containers()
{
    ParticleNumber = MatrixXcd::Zero(n_sites_system,n_harmonics+1);
    PairExpectation = MatrixXcd::Zero(n_sites_system,2*n_harmonics+1);
    Current = VectorXcd::Zero(n_harmonics+1);
}

// Initialize Green's functions and self-energies
void SSJunction::initialize()
{
    clear_Green_containers();
    calculate_Hpert();

    if(ParticleNumber.norm() < 1.0e-7 &&
            PairExpectation.norm() < 1.0e-7)
        Sigma = MatrixXcd_sp(2*n_sites_setup*n_coupled_freqs,
                2*n_sites_setup*n_coupled_freqs);
    else
        calculate_Sigma();


}



bool SSJunction::change_parameters(string param_name, dcomp param_value)
{
    if(param_name == "U"){
        double U = real(param_value);
        this->ssyst.set_U(U);
        return true;
        // ToDo: same comment as with TwoTerminalSetup
    } else if(param_name == "VCL"){
        this->set_VCL(real(param_value));
        this->set_VCR(real(param_value));
        return true;
    } else if(param_name == "VCR"){
        this->set_VCL(real(param_value));
        this->set_VCR(real(param_value));
        return true;
    } else if(param_name == "gate"){
        gate = real(param_value);
        this->ssyst.set_gate(gate);
        return true;
    } else if(param_name == "bias"){
        double bias = real(param_value);
        // Due to our approach, leads have zero chemical potential
        // The bias is conveyed in the time dependence of the hoppings
        this->set_bias(bias);
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
    } else if(param_name == "DeltaL"){
        dcomp DeltaL = param_value;
        this->leadL.set_Delta(DeltaL);
        n_harmonics = max(n_harmonics,
                4*(int)ceil(abs(leadL.get_Delta())/bias));
        ParticleNumber = MatrixXcd::Zero(n_sites_system,n_harmonics+1);
        PairExpectation = MatrixXcd::Zero(n_sites_system,
                2*n_harmonics+1);
        Current = VectorXcd::Zero(n_harmonics+1);
        return true;
    } else if(param_name == "DeltaR"){
        dcomp DeltaR = param_value;
        this->leadR.set_Delta(DeltaR);
        n_harmonics = max(n_harmonics,
                4*(int)ceil(abs(leadR.get_Delta())/bias));
        ParticleNumber = MatrixXcd::Zero(n_sites_system,
                n_harmonics+1);
        PairExpectation = MatrixXcd::Zero(n_sites_system,
                2*n_harmonics+1);
        Current = VectorXcd::Zero(n_harmonics+1);
        return true;
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


void SSJunction::set_bias(const double& bias)
{
    this->bias = bias;
    // Make sure that chemical potential of the leads is zero
    // (it is included in the time-dependency of the lead-system hoppings)
    this->leadR.set_muL(0); 
    this->leadL.set_muL(0);
    if( bias/abs(this->leadL.get_Delta()) < 1e-4){
        cout << "Too small bias. Using bias = abs(DeltaL) instead"
            << '\n';
        this->bias = abs(this->leadL.get_Delta());
    }
    base_frequency = bias;
    n_coupled_freqs = (int)ceil((cutoff_above-cutoff_below)/base_frequency);

    n_harmonics = max(n_harmonics,
            4*(int)ceil(abs(leadL.get_Delta())/this->bias));
    // Initialize containers since change in voltage changes
    // number of cinsidered harmonics
    //HBdG = MatrixXcd_sp(2*n_coupled_freqs*n_sites_system,
    //        2*n_coupled_freqs*n_sites_system);
    ParticleNumber = MatrixXcd::Zero(n_sites_system,n_harmonics+1);
    PairExpectation = MatrixXcd::Zero(n_sites_system,2*n_harmonics+1);
    Current = VectorXcd::Zero(n_harmonics+1);
    initialize();
}

void SSJunction::set_PairExpectation_and_ParticleNumber(
        const MatrixXcd& PairExpectation, const MatrixXcd& Number,
        bool reset_Greens)
{
    if(2*Number.cols()-1 != PairExpectation.cols()){
        cout << "Particle Number and Pair Expectation" 
            << "number of harmonics do not match" 
            << '\n';
        return;
    }
    n_harmonics = Number.cols()-1;
    Current = VectorXcd::Zero(n_harmonics);

    this->ParticleNumber = Number;
    this->PairExpectation = PairExpectation;

    initialize();

    if(reset_Greens)
        this->clear_Green_containers();

}

// Get non-iteracting Green's functions as sparse matrices
// Use sparse since most of the elements in the result matrix
// are zero
//
tuple<MatrixXcd_sp,MatrixXcd_sp> SSJunction::get_gR_and_gl(
        double E, dcomp ieta){
    int H_size = 2*n_sites_setup*n_coupled_freqs;

    MatrixXcd_sp I(H_size,H_size);
    I.setIdentity();

    Matrix2cd gRLL,glLL,gRRR,glRR;
    MatrixXcd gRSS, glSS;

    MatrixXcd_sp gR(H_size,H_size);
    MatrixXcd_sp gl(H_size,H_size);
    //SigmaR.makeCompressed();
    vector<T> gR_triplets;
    vector<T> gl_triplets;

    //cout << "Assemble sparse non-perturbed Green's" << '\n';
    // Assemble non-perturbed Green's functions as sparse matrices
    for(int n = 0; n < n_coupled_freqs; ++n){
        tie(gRLL,glLL) = leadL.get_gR_and_gl(
                E + cutoff_below + n*base_frequency,ieta);
        tie(gRRR,glRR) = leadR.get_gR_and_gl(
                E + cutoff_below + n*base_frequency,ieta);
        tie(gRSS,glSS) = ssyst.get_gR_and_gl(
                E + cutoff_below + n*base_frequency,ieta);
        int nL_idx = 2*n_sites_setup*n; // Left lead index 
        int nR_idx = 2*n_sites_setup*(n+1)-2; // Right lead index 

        // Retarded Green's function, left lead
        gR_triplets.push_back(T(nL_idx,nL_idx, gRLL(0,0)));
        gR_triplets.push_back(T(nL_idx+1,nL_idx, gRLL(1,0)));
        gR_triplets.push_back(T(nL_idx,nL_idx+1, gRLL(0,1)));
        gR_triplets.push_back(T(nL_idx+1,nL_idx+1, gRLL(1,1)));
        // Lesser Green's function, left lead
        gl_triplets.push_back(T(nL_idx,nL_idx, glLL(0,0)));
        gl_triplets.push_back(T(nL_idx+1,nL_idx, glLL(1,0)));
        gl_triplets.push_back(T(nL_idx,nL_idx+1, glLL(0,1)));
        gl_triplets.push_back(T(nL_idx+1,nL_idx+1, glLL(1,1)));
        // Retarded Green's function, right lead
        gR_triplets.push_back(T(nR_idx,nR_idx, gRRR(0,0)));
        gR_triplets.push_back(T(nR_idx+1,nR_idx, gRRR(1,0)));
        gR_triplets.push_back(T(nR_idx,nR_idx+1, gRRR(0,1)));
        gR_triplets.push_back(T(nR_idx+1,nR_idx+1, gRRR(1,1)));
        // Lesser Green's function, right lead
        gl_triplets.push_back(T(nR_idx,nR_idx, glRR(0,0)));
        gl_triplets.push_back(T(nR_idx+1,nR_idx, glRR(1,0)));
        gl_triplets.push_back(T(nR_idx,nR_idx+1, glRR(0,1)));
        gl_triplets.push_back(T(nR_idx+1,nR_idx+1, glRR(1,1)));

        for(int i = 0; i < n_sites_system; i++){
            int ni_idx = nL_idx+2*(i+1);
            for(int j = 0; j < n_sites_system; j++){
                int nj_idx = nL_idx+2*(j+1);
                // Retarded Green's function, system
                gR_triplets.push_back(T(ni_idx,nj_idx,
                            gRSS(2*i,2*j)));
                gR_triplets.push_back(T(ni_idx+1,nj_idx,
                            gRSS(2*i+1,2*j)));
                gR_triplets.push_back(T(ni_idx,nj_idx+1,
                            gRSS(2*i,2*j+1)));
                gR_triplets.push_back(T(ni_idx+1,nj_idx+1, 
                            gRSS(2*i+1,2*j+1)));
                // Lesser Green's function, system
                gl_triplets.push_back(T(ni_idx,nj_idx,
                            glSS(2*i,2*j)));
                gl_triplets.push_back(T(ni_idx+1,nj_idx,
                            glSS(2*i+1,2*j)));
                gl_triplets.push_back(T(ni_idx,nj_idx+1,
                            glSS(2*i,2*j+1)));
                gl_triplets.push_back(T(ni_idx+1,nj_idx+1,
                        glSS(2*i+1,2*j+1)));
            }
        }
    }
    gR.setFromTriplets(gR_triplets.begin(),gR_triplets.end());
    gl.setFromTriplets(gl_triplets.begin(),gl_triplets.end());
    gR.makeCompressed();
    gl.makeCompressed();

    return make_tuple(gR,gl);
}

vector<MatrixXcd_sp> SSJunction::split_ch_matrix_to_branches(
        const MatrixXcd_sp& ch_mat,const vector<vector<int>>& con)
{

    vector<MatrixXcd_sp> result;
    vector<vector<T>> ch_mat_b_triplets;

    //MatrixXi index_table = class_division_table(con);
    static bool con_determined = false;
    static MatrixXi index_table;
    if(!con_determined ){
        con_determined = true;
        index_table = class_division_table(con);
    }

    // Initialize result and triplets
    for(int i = 0; i < (int)con.size(); ++i){
        result.push_back(MatrixXcd_sp(con[i].size(),con[i].size()));
        ch_mat_b_triplets.push_back(vector<T>());
    }

    for(int k = 0; k < ch_mat.outerSize(); ++k){
        for(MatrixXcd_sp::InnerIterator it(ch_mat,k); it; ++it){
            int elem_class = index_table(it.row(),0);
            int b_row_idx = index_table(it.row(),1);
            int b_col_idx = index_table(it.col(),1);
            ch_mat_b_triplets[elem_class].push_back(T(b_row_idx, b_col_idx,it.value()));
        }
    }
    for(int i = 0; i < (int)con.size(); ++i)
        result[i].setFromTriplets(ch_mat_b_triplets[i].begin(),ch_mat_b_triplets[i].end());
    return result;
}

MatrixXcd SSJunction::merge_branches(
        const vector<MatrixXcd>& bs,const vector<vector<int>>& con)
{
    static bool con_determined = false;
    static MatrixXi index_table;
    if(!con_determined ){
        con_determined = true;
        index_table = class_division_table(con);
    }
    //MatrixXi index_table = class_division_table(con);
    //
    int size = index_table.rows();
    MatrixXcd ch_mat = MatrixXcd::Zero(size,size);
    for(int i = 0; i < size; ++i){
        for(int j = 0; j < size; ++j){
            if(index_table(i,0) == index_table(j,0)){ // Add only if index in same branch, others zero
                ch_mat(i,j) = bs[index_table(i,0)](index_table(i,1),index_table(j,1));
            }
        }
    }
    return ch_mat;
}

tuple<MatrixXcd,MatrixXcd> SSJunction::get_GR_and_Gl(
        double E, dcomp ieta, const MatrixXcd_sp& SigmaR_int)
{
    MatrixXcd_sp gR,gl;
    tie(gR,gl) = get_gR_and_gl(E,ieta);

    MatrixXcd_sp SigmaR = Hpert+SigmaR_int;

    static bool con_determined = false;
    static vector<vector<int>> disconnected; 
    if(!con_determined){
        con_determined = true;
        cout << "Total number of d.o.f.: " << SigmaR.cols() << "\n" << flush;
        disconnected = unconnected_branches_DFS(MatrixXcd(gR+gl+SigmaR),1.0e-8);
        cout << "Number of disconnected branches: " << disconnected.size() << "\n" << flush;
        for(auto it = disconnected.begin(); it != disconnected.end(); ++it){
            cout << "Branch " << it - disconnected.begin() << "\n";
            for(auto jt = it->begin(); jt != it->end(); ++jt)
                cout << *jt << " ";
            cout << "\n" << flush;
        }
    }
    
    vector<MatrixXcd_sp> gR_bs = split_ch_matrix_to_branches(gR,disconnected);
    vector<MatrixXcd_sp> gl_bs = split_ch_matrix_to_branches(gl,disconnected);
    vector<MatrixXcd_sp> SigmaR_bs = split_ch_matrix_to_branches(SigmaR,disconnected);
    vector<MatrixXcd> GRs;
    vector<MatrixXcd> Gls;

    for(int i = 0; i < (int)gR_bs.size();i++){
        MatrixXcd_sp I_b(gR_bs[i].rows(),gR_bs[i].cols());
        I_b.setIdentity();
        //MatrixXcd GR_b = MatrixXcd(I_b-gR_bs[i]*SigmaR_bs[i]).
        //    partialPivLu().solve(MatrixXcd(gR_bs[i]));
        MatrixXcd GR_b = MatrixXcd(I_b-gR_bs[i]*SigmaR_bs[i]).
            colPivHouseholderQr().solve(MatrixXcd(gR_bs[i]));
        MatrixXcd IpGRSigmaR_b = I_b+GR_b*SigmaR_bs[i];
        MatrixXcd Gl_b = IpGRSigmaR_b*(gl_bs[i]*IpGRSigmaR_b.adjoint());
        GRs.push_back(GR_b);
        Gls.push_back(Gl_b);
    }

    MatrixXcd GR = merge_branches(GRs,disconnected);
    MatrixXcd Gl = merge_branches(Gls,disconnected);

    return make_tuple(GR,Gl);
}

// Function to manage the storage of Gl and calculate it if needed
MatrixXcd SSJunction::get_Gl_freq(double E,
        dcomp ieta, const MatrixXcd_sp& Sigma_int)
{
    vector<double>::iterator it;
    //it = find_if(Es.begin(),Es.end(),
    //[E](const double& x){return fabs(x-E)<1e-14*fabs(E);});
    it = find_if(Es.begin(),Es.end(),[E](const double& x)
            {return fabs(x-E)<1e-10;});
    if(it!=Es.end())
        return Gls_freq[it-Es.begin()];

    MatrixXcd GR,Gl;
    tie(GR,Gl) = get_GR_and_Gl(E,ieta,Sigma_int);

    Es.push_back(E);
    Gls_freq.push_back(Gl);

    return Gl;
}

MatrixXcd SSJunction::get_harmonic_trace(const MatrixXcd& m, int harm)
{
    MatrixXcd m_traced = MatrixXcd::Zero(2*n_sites_setup,
                                        2*n_sites_setup);
    for(int orb_i = 0; orb_i < 2 * n_sites_setup; ++orb_i){
        for(int orb_j = 0; orb_j < 2 * n_sites_setup; ++orb_j){
            // Trace for the element
            for(int h = 0; h < n_coupled_freqs - harm; h++){
                int hi_idx = 2 * n_sites_setup * h + orb_i;
                int hj_idx = 2 * n_sites_setup * (h + harm) + orb_j;
                m_traced(orb_i,orb_j) += m(hi_idx,hj_idx);
            }
        }
    }

    return m_traced;

}


MatrixXcd SSJunction::get_Gln_freq(double E,int harmonic,dcomp ieta,
        const MatrixXcd_sp& Sigma_int)
{
    //static int new_counter = 0;
    // Find if the set is already calculated
    vector<double>::iterator it;
    it = find_if(Es.begin(),Es.end(),
            [E](const double& x){return fabs(x-E)<1e-10;});
    // Calculate if not
    if(it == Es.end()){
        MatrixXcd GR,Gl;
        vector<MatrixXcd> Glns;
        tie(GR,Gl) = get_GR_and_Gl(E,ieta,Sigma_int);
        // Calculated the appropriate traces for positive harmonics
        for(int n = 0; n < n_coupled_freqs+1; n++){
            MatrixXcd Gln = this->get_harmonic_trace(Gl,n);
            Glns.push_back(Gln);
        }
        Es.push_back(E);
        Glns_freq.push_back(Glns);
        it = Es.end()-1;
    }
    if(harmonic >= 0)
        return Glns_freq[it-Es.begin()][harmonic];
    else
        return -Glns_freq[it-Es.begin()][-harmonic].adjoint();
}


// Wrapper for da2glob
double SSJunction::get_Gln_freq(double E, int harmonic,
        sp_state state1, sp_state state2,
        dcomp ieta, const MatrixXcd_sp& Sigma_int, bool is_real)
{
    dcomp Gln_freq = get_Gln_freq(E,harmonic,ieta,Sigma_int)
        (sp_state_to_idx(state1),sp_state_to_idx(state2));

    if(is_real)
        return real(Gln_freq);
    else
        return imag(Gln_freq);
}

int SSJunction::sp_state_to_idx(sp_state state)
{
    int idx = 0;
    if(get<0>(state) == 0){
        idx = get<2>(state);
    }else if(get<0>(state) == 1){
        idx = 2+2*get<1>(state) + get<2>(state);
    }else {
        idx = 2+2*n_sites_system+get<2>(state);
    }
    return idx;
}

sp_state SSJunction::idx_to_sp_state(int idx)
{
    int spin = idx%2;
    int location = idx/2;
    int syst_part = 0;
    int site = 0;
    if(location == 0){
        syst_part = 0;
        site = 0;
    }
    else if(location <= this->n_sites_system){
        syst_part = 1;
        site = location-1;
    }
    else{
        syst_part = 2;
        site = 0;
    }
    return make_tuple(syst_part,site,spin);
}

int SSJunction::harmonic_and_sp_state_to_idx(int harmonic, sp_state state)
{
    int idx = sp_state_to_idx(state);
    idx += 2*n_sites_setup*harmonic;
    return idx;
}


dcomp SSJunction::get_Gln_time(int harmonic,
        sp_state sp_state1, sp_state sp_state2,
        dcomp ieta, const MatrixXcd_sp& Sigma_int, double tol)
{
    using namespace placeholders;
    static double pi = 3.141592653589793238462643383279502884197169;

    if(fabs(harmonic) > n_coupled_freqs){
        cout << "Harmonic too large" << '\n';
        return 0.0 + 0.0i;
    }

    double Emin = -0.5*abs(base_frequency);
    double Emax = 0.5*abs(base_frequency);

    VectorXd ints = VectorXd::LinSpaced(2,Emin,Emax);

    bool is_calculated = false;

    if(harmonic >= 0)
        is_calculated = abs(Glns_time[harmonic]
                (sp_state_to_idx(sp_state1),sp_state_to_idx(sp_state2)))
            != numeric_limits<double>::infinity();
    if(harmonic < 0)
        is_calculated = abs(Glns_time[-harmonic]
                (sp_state_to_idx(sp_state2),sp_state_to_idx(sp_state1))) 
            != numeric_limits<double>::infinity();


    // Return already calculated if exists
    if(is_calculated && harmonic >= 0){
        return Glns_time[harmonic]
            (sp_state_to_idx(sp_state1),sp_state_to_idx(sp_state2));
    }
    else if(is_calculated && harmonic < 0){
        return -conj(Glns_time[-harmonic]
                (sp_state_to_idx(sp_state2),sp_state_to_idx(sp_state1)));
    }
    else{ // else calculate the value and add it to the list of known values
        double Glij_real = 9000, Glij_imag = 9000;
        int nint_r,nint_i;
        tie(Glij_real,nint_r) = da2glob(
                std::bind(static_cast<double (SSJunction::*)
                    (double,int,sp_state, sp_state,
                     dcomp, const MatrixXcd_sp&,bool)>
                    (&SSJunction::get_Gln_freq),
                    this,_1,harmonic,sp_state1,sp_state2,
                    ieta,Sigma_int,true),ints,tol,tol);
        tie(Glij_imag,nint_i) = da2glob(
                std::bind(static_cast<double (SSJunction::*)
                    (double,int,sp_state,sp_state,
                     dcomp, const MatrixXcd_sp&,bool)>
                    (&SSJunction::get_Gln_freq),
                    this,_1,harmonic,sp_state1,sp_state2,
                    ieta,Sigma_int,false),ints,tol,tol);
        //if(harmonic == 0){
        //    TwoTerminalSetup tts(this->ssyst,this->gate,0.0,this->leadL,
        //            this->leadR,this->tSL,this->tSR,
        //            this->cpoint_L,this->cpoint_R,this->VCL,this->VCR);

        //}

        //Glij_real = trapezoid<double,double>(std::bind(
        //static_cast<double (SSJunction::*)(
        //double,int,sp_state,sp_state, dcomp, MatrixXcd_sp,bool)>(
        //&SSJunction::get_Gln_freq),
        //this,_1,harmonic,sp_state1,sp_state2,ieta,Sigma_int,false),
        //0.0,abs(bias/2),100);
        //Glij_imag = trapezoid<double,double>(
        //std::bind(static_cast<double (SSJunction::*)
        //(double,int,sp_state,sp_state, dcomp, MatrixXcd_sp,bool)>
        //(&SSJunction::get_Gln_freq),
        //this,_1,harmonic,sp_state1,sp_state2,ieta,Sigma_int,true),
        //0.0,abs(bias/2),100);
        if(harmonic >= 0){// Save the harmonic
            Glns_time[harmonic]
                (sp_state_to_idx(sp_state1),sp_state_to_idx(sp_state2)) 
                = (Glij_real + 1.0i*Glij_imag)/(2*pi); 
        } else {// Save the corresponding positive harmonic
            Glns_time[-harmonic]
                (sp_state_to_idx(sp_state2),sp_state_to_idx(sp_state1))
                = -(Glij_real - 1.0i*Glij_imag)/(2*pi);
        }
        return (Glij_real+1.0i*Glij_imag)/(2*pi);
    } 

    return 0.0i;
}

vector<MatrixXcd>& SSJunction::get_Glns_time(dcomp ieta,
        double tol)
{
    // Make sure all elements are calculated
    for(int n = 0; n < 2*n_harmonics+1; n++){
        for(int i = 0; i < 2*n_sites_setup; i++){
            sp_state statei = idx_to_sp_state(i);
            for(int j = 0; j < 2*n_sites_setup; j++){
                sp_state statej = idx_to_sp_state(j);
                get_Gln_time(n,statei,statej,
                        ieta, this->Sigma,tol);
            }

        }
    }
    return Glns_time;
}



VectorXcd SSJunction::update(const VectorXcd& X, dcomp ieta,double tol_quad)
{
    using namespace placeholders;

    VectorXcd FX = VectorXcd::Zero(n_sites_system*3*(n_harmonics + 1)); // result vector, note that both positive and negative harmonics are needed for Delta but only non-negative for the particle number
    // Ordering: for each non-negative frequency, for each site particle number harmonic and the positive and negative harmonic for Delta

    // Self-energy for interaction
    MatrixXcd_sp Sigma_int(2*n_sites_setup*n_coupled_freqs,2*n_sites_setup*n_coupled_freqs); 

    // Convert from X to Sigma_int
    int first_row = 0;
    int last_row = 0;
    vector<T> Sigma_int_triplets;
    for(int n = 0; n < n_harmonics+1; n++){
        if(n == 0){
            for(int i = 0; i < n_sites_system;i++){
                for(int m = 0; m < n_coupled_freqs; m++){
                    int mi_idx = 2*n_sites_setup*m + 2*(i+1);
                    Sigma_int_triplets.push_back(T(mi_idx,mi_idx,X(3*i)));
                    Sigma_int_triplets.push_back(T(mi_idx,mi_idx+1,X(3*i+1)));
                    Sigma_int_triplets.push_back(T(mi_idx+1,mi_idx,conj(X(3*i+1))));
                    Sigma_int_triplets.push_back(T(mi_idx+1,mi_idx+1,-X(3*i)));
                }
            }
        } else if (n < n_coupled_freqs) {
            first_row = 0;
            last_row = n_coupled_freqs - n; 
            for(int i = 0; i < n_sites_system;i++){
                int Xni_idx = n*3*n_sites_system+3*i;
                dcomp Xni = X(Xni_idx);
                dcomp Xnp1i = X(Xni_idx+1);
                dcomp Xnp2i = X(Xni_idx+2);
                for(int m = first_row; m < last_row-1; m++){
                    int mi_idx = 2*n_sites_setup*m + 2*(i+1);
                    int mpni_idx = 2*n_sites_setup*(m+n) + 2*(i+1);
                    // Upper triangular terms
                    Sigma_int_triplets.push_back(T(mi_idx,mpni_idx,Xni));
                    Sigma_int_triplets.push_back(T(mi_idx,mpni_idx+1,Xnp1i));
                    Sigma_int_triplets.push_back(T(mi_idx+1,mpni_idx,conj(Xnp2i)));
                    Sigma_int_triplets.push_back(T(mi_idx+1,mpni_idx+1,-conj(Xni)));

                    // Lower triangular terms
                    Sigma_int_triplets.push_back(T(mpni_idx,mi_idx,conj(Xni)));
                    Sigma_int_triplets.push_back(T(mpni_idx+1,mi_idx,conj(Xnp1i)));
                    Sigma_int_triplets.push_back(T(mpni_idx,mi_idx+1,Xnp2i));
                    Sigma_int_triplets.push_back(T(mpni_idx+1,mi_idx+1,-Xni));
                }
            }
        }
    }
    Sigma_int.setFromTriplets(Sigma_int_triplets.begin(),Sigma_int_triplets.end());

    clear_Green_containers();
    // From Sigma_int to FX
    for(int i = 0; i < n_sites_system; i++){
        sp_state hartree_state = make_tuple(1,i,0);
        sp_state delta_state1 = make_tuple(1,i,0);
        sp_state delta_state2 = make_tuple(1,i,1);
        Observable hartree(n_sites_system,n_coupled_freqs);
        hartree.add_matrix_element(0,
                hartree_state,hartree_state,
                ssyst.get_U());
        Observable delta(n_sites_system,n_coupled_freqs);
        delta.add_matrix_element(0,
                delta_state1, delta_state2,
                ssyst.get_U());
        for(int n = 0; n < n_harmonics+1; n++){
            auto Gln_time_func = std::bind(
                    static_cast<dcomp (SSJunction::*)
                    (int,sp_state, sp_state,dcomp, const MatrixXcd_sp&, double)>
                    (&SSJunction::get_Gln_time),
                    this,_1,_2,_3, ieta, Sigma_int, tol_quad);
            FX(n*3*n_sites_system + 3*i) =
                hartree.get_expectation_value(Gln_time_func,n);
            FX(n*3*n_sites_system + 3*i+1) =
                delta.get_expectation_value(Gln_time_func,n);
            if( n == 0)
                FX(3*i+2) = FX(3*i+1);
            else
                FX(n*3*n_sites_system + 3*i+2) =
                    delta.get_expectation_value(Gln_time_func,-n);
        }
    }

    return FX;
}

bool SSJunction::self_consistent_loop(
        function<VectorXcd(const VectorXcd&)> FX,
        const MatrixXcd& Hartree0,const MatrixXcd& Delta0,
        string scf_cfg_path, string save_path)
{

    using namespace placeholders;
    //int X_size = Hartree0.size()+Delta0.size()+1;
    int X_size = Hartree0.rows()*Hartree0.cols() 
        + Delta0.rows()*Delta0.cols() + n_sites_system;

    if(abs(this->ssyst.get_U()) < 1e-6){
        cout << "Interaction strength small, self-energy zero" << '\n';
        return true;
    }

    this->set_PairExpectation_and_ParticleNumber(
            MatrixXcd::Zero(n_sites_system,2*n_harmonics+1),
            MatrixXcd::Zero(n_sites_system,n_harmonics+1),
            true);
    clear_Green_containers();

    VectorXcd X = VectorXcd::Zero(X_size);
    VectorXcd X0 = X;
    for(int n = 0; n < n_harmonics+1;n++){
        int n_idx = 3*n_sites_system*n;
        for(int i = 0; i < n_sites_system; i++){
            int ni_idx = n_idx + 3*i;
            X(ni_idx) = Hartree0(i,n);
            if(n == 0){
                X(ni_idx+1) = Delta0(i,0);
                X(ni_idx+2) = Delta0(i,0);
            }else{
                X(ni_idx+1) = Delta0(i,2*n-1);
                X(ni_idx+2) = Delta0(i,2*n);
            }
        }
    }

    ScfSolver solver(&X,scf_cfg_path,save_path);
    solver.iterate(FX);

    int total_iter = solver.get_iterations();
    bool converged = solver.get_converged();

    cout << "Scf loop converged: " << converged << "\n";
    cout << "Total iterations: " << total_iter << "\n";

    MatrixXcd Hart(n_sites_system,n_harmonics+1);
    MatrixXcd Delta(n_sites_system,2*n_harmonics+1);
    for(int n = 0; n < n_harmonics+1; n++){
        for(int i = 0; i < n_sites_system; i++){
            Hart(i,n) = X(3*n*n_sites_system+3*i);
            if(n == 0)
                Delta(i,0) = X(3*i+1);
            else{
                Delta(i,2*n-1) = X(3*n*n_sites_system+3*i+1);
                Delta(i,2*n) = X(3*n*n_sites_system+3*i+2);
            }
        }
    }
    if(abs(ssyst.get_U()) > 1.0e-6){
        this->set_PairExpectation_and_ParticleNumber(Delta/ssyst.get_U(),2*Hart/ssyst.get_U(),true);
    }
    calculate_Sigma();
    clear_Green_containers();
    return converged;
}


void SSJunction::CalculateParticleNumber(dcomp ieta, double tol)
{
    using namespace placeholders;
    ParticleNumber = MatrixXcd::Zero(n_sites_system,n_harmonics+1);

    for(int i = 0; i < n_sites_system; i++){
        Observable pnum(n_sites_system,n_coupled_freqs);
        sp_state pnum_state = make_tuple(1,i,0);
        pnum.add_matrix_element(0,pnum_state,pnum_state,2.0);
        for(int n = 0; n < n_harmonics+1; n++){
            ParticleNumber(i,n) = pnum.get_expectation_value(
                    std::bind(static_cast<dcomp (SSJunction::*)
                        (int,sp_state, sp_state,dcomp, const MatrixXcd_sp&, double)>
                        (&SSJunction::get_Gln_time),
                        this,_1,_2,_3,ieta, Sigma, tol),n);
        }
    }
    cout << "Particle Number calculated" << '\n';
}

dcomp SSJunction::CalculateParticleNumber(int site, int harmonic, dcomp ieta, double tol)
{
    using namespace placeholders;
    Observable pnum(n_sites_system,n_coupled_freqs);
    sp_state pnum_state = make_tuple(1,site,0);
    pnum.add_matrix_element(0,pnum_state,pnum_state,2.0);

    return pnum.get_expectation_value(
            std::bind(static_cast<dcomp (SSJunction::*)
                (int,sp_state, sp_state,dcomp,const MatrixXcd_sp&, double)>
                (&SSJunction::get_Gln_time),this,_1,_2,_3,ieta, Sigma, tol),
            harmonic);
}

void SSJunction::CalculatePairExpectation(dcomp ieta,double tol)
{
    using namespace placeholders;
    PairExpectation = MatrixXcd::Zero(n_sites_system,2*n_harmonics+1);

    for(int i = 0; i < n_sites_system; i++){
        Observable pair(n_sites_system,n_coupled_freqs);
        sp_state pair_state1 = make_tuple(1,i,1);
        sp_state pair_state2 = make_tuple(1,i,0);
        pair.add_matrix_element(0,pair_state1,pair_state2,1.0);
        for(int n = 0; n < n_harmonics+1; n++){
            if(n == 0)
                PairExpectation(i,0) = pair.get_expectation_value(
                        std::bind(static_cast<dcomp (SSJunction::*)
                        (int,sp_state, sp_state,dcomp, const MatrixXcd_sp&, double)>
                        (&SSJunction::get_Gln_time),
                        this,_1,_2,_3,ieta, Sigma, tol),0);
            else{
                PairExpectation(i,2*n-1) = pair.get_expectation_value(
                        std::bind(static_cast<dcomp (SSJunction::*)
                        (int,sp_state, sp_state,dcomp, const MatrixXcd_sp&, double)>
                        (&SSJunction::get_Gln_time),
                        this,_1,_2,_3,ieta, Sigma, tol),n);
                PairExpectation(i,2*n) = pair.get_expectation_value(
                        std::bind(static_cast<dcomp (SSJunction::*)
                        (int,sp_state, sp_state,dcomp, const MatrixXcd_sp&, double)>
                        (&SSJunction::get_Gln_time),this,_1,_2,_3,ieta, Sigma, tol),-n);
            }
        }
    }
    cout << "Pair Expectation calculated " << '\n';
}


dcomp SSJunction::CalculatePairExpectation(int site,int harmonic,
        dcomp ieta,double tol)
{
    using namespace placeholders;

    Observable pair(n_sites_system,n_coupled_freqs);
    sp_state pair_state1 = make_tuple(1,site,0);
    sp_state pair_state2 = make_tuple(1,site,1);
    pair.add_matrix_element(0,pair_state1,pair_state2,1.0);
    return pair.get_expectation_value(
            std::bind(static_cast<dcomp (SSJunction::*)
            (int,sp_state, sp_state,dcomp, const MatrixXcd_sp&, double)>
            (&SSJunction::get_Gln_time),
            this,_1,_2,_3,ieta, Sigma, tol),harmonic);
}

void SSJunction::CalculateCurrent(dcomp ieta,double tol,
        int lead_idx, int direction)
{
    using namespace placeholders;

    Current = VectorXcd::Zero(n_harmonics+1);

    Observable cur(n_sites_system,n_coupled_freqs);
    sp_state cur_state1;
    sp_state cur_state2;
    if(lead_idx == 0){
        cur_state1 = make_tuple(1,cpoint_L,0);
        cur_state2 = make_tuple(0,0,0);
        cur.add_matrix_element(-1,cur_state1,cur_state2,-1.0i*tSL);
        cur.add_matrix_element(1,cur_state2,cur_state1,1.0i*conj(tSL));
    } else {
        cur_state1 = make_tuple(1,cpoint_R,0);
        cur_state2 = make_tuple(2,0,0);
        cur.add_matrix_element(0,cur_state1,cur_state2,1.0i*tSR);
        cur.add_matrix_element(0,cur_state2,cur_state1,-1.0i*conj(tSR));
    }

    for(int n = 0; n < n_harmonics+1; n++){
        Current(n) = direction*2.0*cur.get_expectation_value(
                std::bind(static_cast<dcomp (SSJunction::*)
                (int,sp_state, sp_state,dcomp,const MatrixXcd_sp&, double)>
                (&SSJunction::get_Gln_time),
                this,_1,_2,_3,ieta, Sigma, tol),n);
    }

    cout << "Current calculated! " << '\n';
}

dcomp SSJunction::CalculateCurrent(dcomp ieta,int harmonic,
        double tol, int lead_idx, int direction)
{
    using namespace placeholders;

    Observable cur(n_sites_system,n_coupled_freqs);
    sp_state cur_state1;
    sp_state cur_state2;
    if(lead_idx == 0){
        cur_state1 = make_tuple(1,cpoint_L,0);
        cur_state2 = make_tuple(0,0,0);
        cur.add_matrix_element(-1,cur_state1,cur_state2,-1.0i*tSL);
        cur.add_matrix_element(1,cur_state2,cur_state1,1.0i*conj(tSL));
    } else {
        cur_state1 = make_tuple(1,cpoint_R,0);
        cur_state2 = make_tuple(2,0,0);
        cur.add_matrix_element(0,cur_state1,cur_state2,1.0i*tSR);
        cur.add_matrix_element(0,cur_state2,cur_state1,-1.0i*conj(tSR));
    }


    dcomp current  = direction*2.0*cur.get_expectation_value(
        std::bind(static_cast<dcomp (SSJunction::*)
        (int,sp_state, sp_state,dcomp,const MatrixXcd_sp&, double)>
        (&SSJunction::get_Gln_time),
        this,_1,_2,_3,ieta, Sigma, tol),harmonic);

    return current;
}





