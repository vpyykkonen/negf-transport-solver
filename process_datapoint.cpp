#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <chrono>
#include <algorithm>
#include <map>
#include <Eigen/Dense>
#include <functional>

#include "Lead.h"
#include "ScatteringSystem.h"
#include "TwoTerminalSetup.h"
#include "SSJunction.h"
#include "file_io.h"

#include "H5Cpp.h"
using namespace H5;

using namespace std;
using namespace Eigen;

typedef complex<double> dcomp;

#include <filesystem> // C++17 feature

int main(int argc, char* argv[])
{
    using namespace placeholders;
    // Data point base path from argument
    if(argc < 2)
    {
        cout << "Data point folder not given. Exiting." << endl;
        return 0;
    }
    string data_point_path = argv[1];

    string format;
    if(data_point_path.substr(data_point_path.size() - 3) == ".h5")
        format = "hdf5";
    else
        format = "csv";

    string scf_save_path = ""; // Path to save scf_iter
    if(format == "csv")
        scf_save_path = data_point_path + "/Scf_iterations";
    else
        scf_save_path = data_point_path;

    string method_config_path = "./parameters.cfg";
    string scf_config_path = "./scf_params.cfg";
    string scf_initial_path = "";

    cout << "Processing " << data_point_path << "\n";


    if(argc < 3)
        cout << "Method configuration file not given. Using standard." << endl;
    else 
        method_config_path = argv[2];

    if(argc < 4)
        cout << "Scf config file not given. Using standard." << endl;
    else
        scf_config_path = argv[3];
    if(argc < 5)
        cout << "Scf initial guess not provided. Generating based on configs. " << endl;
    else
        scf_initial_path = argv[4];

    map<string,string> method_config = load_file_to_string_map(method_config_path, "=");
    map<string,string> scf_config = load_file_to_string_map(scf_config_path, "=");

    string method = method_config["method"];
    try {
        if(method == "default" || method == "Pade" 
                || method == "real_freq_H_time_periodic"
         || method == "real_freq_H_time_independent" || method == "closed")
            cout << "Method: " << method << endl;
        else throw(method);
    } catch (string method){
        cout << "Invalid method string: " << method << endl;
    }


    VectorXcd x0;
    if(scf_initial_path != ""){
        MatrixXcd x0_temp = readComplexMatrix(scf_initial_path.c_str());
        x0 = Map<VectorXcd>(x0_temp.data(),x0_temp.rows()*x0_temp.cols());
    }
    
    // parameters;
    double tL = 0.0, tR = 0.0, TL = 0.0, TR = 0.0, tLS = 0.0;
    double tRS = 0.0, VCL = 0.0, VCR = 0.0, gate = 0.0, U = 0.0, bias = 0.0, disorder = 0.0;
    double cutoff_below = 0.0, cutoff_above = 0.0, tol_quad =.0;
    dcomp ieta = 0.0i, init_delta;
    string lead_config("XX");
    string cpoint_L_str(""), cpoint_R_str("");
    int cpoint_L = 0,cpoint_R = 0, n_approx = 0, n_harmonics = 0, seed = 0;
    int n_sites_system = 0;
    bool equilibrium = false, finite_temp = false;
    double pi = 3.14159265;
    dcomp DeltaL = 0.0 + 0.0i;
    dcomp DeltaR = 0.0 + 0.0i;

    if(format == "csv"){
        map<string,string> data_point_info = load_file_to_string_map(
                data_point_path+"/data_point_info.txt","=");

        lead_config = data_point_info["lead_config"];
        equilibrium = stoi(data_point_info["equilibrium"]);

        map<string,string> setup_params = load_file_to_string_map(
                data_point_path+"/parameters.cfg","=");
        // Common
        // Lead parameters
        tL = stod(setup_params["tL"]);
        tR = stod(setup_params["tR"]);
        TL = stod(setup_params["TL"]);

        // System-lead parameters
        tLS = stod(setup_params["tLS"]);
        tRS = stod(setup_params["tRS"]);
        VCL = stod(setup_params["VCL"]);
        VCR = stod(setup_params["VCR"]);

        // Scattering system
        gate = stod(setup_params["gate"]);
        if(setup_params.find("disorder") != setup_params.end())
            disorder = stod(setup_params["disorder"]);
        if(setup_params.find("seed") != setup_params.end())
            seed = stoi(setup_params["seed"]);
        U = stod(setup_params["U"]);

        TR = TL;
        if(!equilibrium)
            TR = stod(setup_params["TR"]);

        bias = 0;
        if(!equilibrium)
            bias = stod(setup_params["bias"]);

        finite_temp = (TL > 1e-4);

        cpoint_L_str = setup_params["cpoint_L"];
        cpoint_R_str = setup_params["cpoint_R"];

        // Handle the lead order parameters based on the configuration
        if(lead_config.compare("SS") == 0){
            DeltaL = stod(setup_params["DeltaL"])*exp(1.0i*pi*stod(setup_params["phaseL"]));
            DeltaR = stod(setup_params["DeltaR"])*exp(1.0i*pi*stod(setup_params["phaseR"]));
        } else if (lead_config.compare("SN")==0){
            DeltaL = stod(setup_params["DeltaL"])*exp(1.0i*pi*stod(setup_params["phaseL"]));
        }else if (lead_config.compare("NS")==0){
            DeltaR = stod(setup_params["DeltaR"])*exp(1.0i*pi*stod(setup_params["phaseR"]));
        }
    } else if (format == "hdf5"){
        H5File* dpoint_file = new H5File(data_point_path, H5F_ACC_RDONLY);
        get_data_from_h5attr(dpoint_file,"lead_config","string",&lead_config);
        get_data_from_h5attr(dpoint_file,"tL","double",&tL);
        get_data_from_h5attr(dpoint_file,"tR","double",&tR);
        get_data_from_h5attr(dpoint_file,"TL","double",&TL);
        get_data_from_h5attr(dpoint_file,"tLS","double",&tLS);
        get_data_from_h5attr(dpoint_file,"tRS","double",&tRS);
        get_data_from_h5attr(dpoint_file,"VCL","double",&VCL);
        get_data_from_h5attr(dpoint_file,"VCR","double",&VCR);
        get_data_from_h5attr(dpoint_file,"gate","double",&gate);
        if(!get_data_from_h5attr(dpoint_file,"disorder","double",&disorder))
            disorder = 0.0;
        if(!get_data_from_h5attr(dpoint_file,"seed","int",&seed))
            seed = 0;
        get_data_from_h5attr(dpoint_file,"U","double",&U);
        
        string equil_str;
        get_data_from_h5attr(dpoint_file,"equilibrium","string",&equil_str);
        if(equil_str == "1")
            equilibrium = true;
        else
            equilibrium = false;
        finite_temp = (TL > 1.0e-4);

        if(equilibrium){
            bias = 0.0;
            TR = TL; 
        } else {
            get_data_from_h5attr(dpoint_file,"TR","double",&TR);
            get_data_from_h5attr(dpoint_file,"bias","double",&bias);
        }

        get_data_from_h5attr(dpoint_file,"cpoint_L","string",&cpoint_L_str);
        get_data_from_h5attr(dpoint_file,"cpoint_R","string",&cpoint_R_str);

        double DeltaL_abs, DeltaL_phase, DeltaR_abs, DeltaR_phase;
        if(lead_config.compare("SS") == 0){
            get_data_from_h5attr(dpoint_file,"DeltaL","double",&DeltaL_abs);
            get_data_from_h5attr(dpoint_file,"phaseL","double",&DeltaL_phase);
            get_data_from_h5attr(dpoint_file,"DeltaR","double",&DeltaR_abs);
            get_data_from_h5attr(dpoint_file,"phaseR","double",&DeltaR_phase);
            DeltaL = DeltaL_abs*exp(1.0i*pi*DeltaL_phase);
            DeltaR = DeltaR_abs*exp(1.0i*pi*DeltaR_phase);
        } else if (lead_config.compare("SN")==0){
            get_data_from_h5attr(dpoint_file,"DeltaL","double",&DeltaL_abs);
            get_data_from_h5attr(dpoint_file,"phaseL","double",&DeltaL_phase);
            DeltaL = DeltaL_abs*exp(1.0i*pi*DeltaL_phase);
        }else if (lead_config.compare("NS")==0){
            get_data_from_h5attr(dpoint_file,"DeltaR","double",&DeltaR_abs);
            get_data_from_h5attr(dpoint_file,"phaseR","double",&DeltaR_phase);
            DeltaR = DeltaR_abs*exp(1.0i*pi*DeltaR_phase);
        }

        delete dpoint_file;
    }

    cout << "bias: " << bias << "\n";
    cout << "gate: " << gate << "\n";
    cout << "U: " << U << "\n";
    cout << "TL: " << TL << "\n";
    cout << "TR: " << TR << "\n";

    Lead leadL(tL,DeltaL,bias,TL);
    Lead leadR(tR,DeltaR,0.0,TR);

    ScatteringSystem ssyst;
    if(format == "csv")
        ssyst = ScatteringSystem(data_point_path + "/geometry.cfg",U,TL);
    if(format == "hdf5"){
        H5File* dpoint_file = new H5File(data_point_path, H5F_ACC_RDONLY);
        int unitcell_sites, dim, n_hoppings;
        string n_unitcells_str, on_site_str, edge_removal_str;
        get_data_from_h5attr(dpoint_file,"unitcell_sites","int",&unitcell_sites);
        get_data_from_h5attr(dpoint_file,"dim","int",&dim);
        get_data_from_h5attr(dpoint_file,"n_hoppings","int",&n_hoppings);
        get_data_from_h5attr(dpoint_file,"n_unitcells","string",&n_unitcells_str);
        get_data_from_h5attr(dpoint_file,"on_site","string",&on_site_str);
        get_data_from_h5attr(dpoint_file,"edge","string",&edge_removal_str);
        vector<string> hopping_strs;
        for(int j = 0; j < n_hoppings; j++){
            string hopping_str;
            get_data_from_h5attr(dpoint_file,"hopping"+to_string(j+1),"string",&hopping_str);
            hopping_strs.push_back(hopping_str);
        }
        //cout << "Building scattering system" << endl;
        ssyst = ScatteringSystem(dim, unitcell_sites, n_unitcells_str, on_site_str, n_hoppings, hopping_strs, edge_removal_str, U,TL);
        //cout << "Done" << endl;
        delete dpoint_file;
    }
    ssyst.set_gate(gate);
    istringstream(method_config["seed"]) >> seed;
    ssyst.add_disorder(disorder,seed);

    n_sites_system = ssyst.get_n_sites();
    if(cpoint_L_str.compare("end") == 0)
        cpoint_L = n_sites_system-1;
    else 
        cpoint_L = stoi(cpoint_L_str);

    cpoint_R = 0;
    if(cpoint_R_str.compare("end") == 0)
        cpoint_R = n_sites_system-1;
    else 
        cpoint_R = stoi(cpoint_R_str);



    // data output directory path
    string data_output_path = data_point_path + "/Data";
    if(!fs::exists(data_output_path) && format == "csv")
        fs::create_directory(data_output_path);
    IOFormat full(FullPrecision, 0, "; ", "\n", "", "","","");

    VectorXcd current;
    MatrixXcd pair_expectations;
    MatrixXcd particle_numbers;
    vector<MatrixXcd> Glns_time;
    bool converged = false;

    if(method == "closed" || lead_config == "closed"){
        cout << "closed" << endl;
        istringstream(method_config["init_delta"]) >> init_delta;

        VectorXd Hartree0; 
        VectorXcd Delta0; 
        if(scf_initial_path != "" && x0.size() == 2*n_sites_system){
            Hartree0 = x0.head(n_sites_system).real();
            Delta0 = x0.tail(n_sites_system);
        } else{
            Hartree0 = VectorXd::Zero(n_sites_system);
            Delta0 = init_delta*VectorXcd::Ones(n_sites_system);
        }

        converged = ssyst.self_consistent_loop(
                Hartree0,Delta0,
                scf_config_path,scf_save_path);

        VectorXcd pnum(n_sites_system);
        VectorXcd pair(n_sites_system);
        for(int n = 0; n < n_sites_system;n++){
            pnum(n) = ssyst.ParticleNumber(n);
            pair(n) = ssyst.PairCorrelator(n);
        }
        particle_numbers = Map<MatrixXcd>(pnum.data(),n_sites_system,1);
        pair_expectations = Map<MatrixXcd>(pair.data(),n_sites_system,1);
        current = VectorXcd::Zero(1);

        Glns_time.push_back(ssyst.get_Gl());

        cout << "Particle numbers: " << '\n' << particle_numbers << endl;
        cout << "Total particle number: " << '\n'
            << particle_numbers.real().sum() << endl;
        cout << "Pair expectations: " << '\n' << pair_expectations << endl;

    }
    if((method == "default" && equilibrium && finite_temp)
            || method == "Pade" ){
        n_approx = stoi(method_config["n_approx"]);
        TwoTerminalSetup setup(ssyst,gate,0.0,leadL,leadR,tLS,tRS,
                cpoint_L,cpoint_R,VCL,VCR);

        istringstream(method_config["init_delta"]) >> init_delta;

        VectorXd Hartree0; 
        VectorXcd Delta0; 
        if(scf_initial_path != "" && x0.size() == 2*n_sites_system){
            Hartree0 = x0.head(n_sites_system).real();
            Delta0 = x0.tail(n_sites_system);
        } else{
            Hartree0 = VectorXd::Zero(n_sites_system);
            Delta0 = init_delta*VectorXcd::Ones(n_sites_system);
        }

        // Run self-consistent loop
        converged = setup.self_consistent_loop(
                std::bind(
                    static_cast<VectorXcd(TwoTerminalSetup::*)
                    (const VectorXcd&,int)>
                    (&TwoTerminalSetup::update_equilibrium),
                    &setup,_1,n_approx),
                Hartree0,Delta0,
                scf_config_path, scf_save_path);
        current = VectorXcd::Zero(1);
        current[0] = setup.CurrentEquilibrium(n_approx);
        VectorXcd pnum = setup.ParticleNumberEquilibrium(n_approx);
        particle_numbers = Map<MatrixXcd>(pnum.data(),n_sites_system,1);
        pair_expectations = Map<MatrixXcd>(setup.PairExpectationEquilibrium(n_approx).data(),n_sites_system,1);
        MatrixXcd Gl_time = setup.get_Gl_time_Pade(n_approx);
        Glns_time.push_back(Gl_time);

        cout << "Current: " << '\n' << current.real() << endl;
        cout << "Particle numbers: " << '\n' << particle_numbers << endl;
        cout << "Total particle number: " << '\n'
            << particle_numbers.real().sum() << endl;
        cout << "Pair expectations: " << '\n' << pair_expectations << endl;
    }
    if ((method == "default" && !equilibrium && lead_config == "SS") 
            || method == "real_freq_H_time_periodic"){
        cutoff_below = stod(method_config["cutoff_below"]);
        cutoff_above = stod(method_config["cutoff_above"]);
        istringstream(method_config["ieta"]) >> ieta;
        istringstream(method_config["init_delta"]) >> init_delta;
        tol_quad = stod(method_config["tol_quad"]);
        n_harmonics = stoi(method_config["n_harmonics"]);

        SSJunction setup(ssyst,gate,bias,leadL,leadR,tLS,tRS,
                cpoint_L,cpoint_R, VCL, VCR,
                cutoff_below, cutoff_above, n_harmonics);

        // zero and positive harmonics of Hartree, Delta and Delta*
        // negative harmonics are obtained from the positive
        // In total 3*n_sites_system for each harmonic
        int x0_size = 3*n_sites_system*(n_harmonics+1);
        int Delta_rows = n_sites_system;
        int Delta_cols = 2*n_harmonics+1;
        int Hartree_rows = n_sites_system;
        int Hartree_cols = n_harmonics+1;

        MatrixXcd Hartree0 = MatrixXcd::Zero(
                Hartree_rows, Hartree_cols);
        MatrixXcd Delta0 = MatrixXcd::Zero(
                Delta_rows,Delta_cols);

        if(scf_initial_path != "" && x0_size == x0.size() ){
            for(int i = 0; i < n_sites_system; i++){
                for(int j = 0; j < n_harmonics+1; j++){
                    Hartree0(i,j) = x0(3*n_sites_system*j+3*i);
                    Delta0(i,2*j) = x0(3*n_sites_system*j+3*i+1);
                    if(j > 0)
                        Delta0(i,2*j-1) = x0(3*n_sites_system*j+3*i+2);
                }
            }
        } else 
            for(int i = 0; i < n_sites_system; ++i)
                Delta0(i,0) = init_delta;

        converged = setup.self_consistent_loop(std::bind(
                    static_cast<VectorXcd(SSJunction::*)
                    (const VectorXcd&,dcomp,double)>
                    (&SSJunction::update),&setup,_1,ieta,tol_quad), 
                Hartree0,Delta0,
                scf_config_path,scf_save_path);
            //data_point_path+"/scf_iterations/scf_params.cfg");
        n_harmonics = setup.get_n_harmonics();
        Glns_time = setup.get_Glns_time(ieta,tol_quad);
        setup.CalculateParticleNumber(ieta,tol_quad);
        setup.CalculatePairExpectation(ieta,tol_quad);
        setup.CalculateCurrent(ieta,tol_quad,0,1);
        particle_numbers = setup.get_ParticleNumber();
        pair_expectations = setup.get_PairExpectation();
        current = setup.get_Current();
        cout << "Current: " << '\n' << current << endl;
        cout << "Particle numbers: "<< '\n' << particle_numbers << endl;
        cout << "Total particle number: " << '\n'<< particle_numbers.col(0).real().sum() << endl;
        cout << "Pair expectations: "<< '\n' << pair_expectations << endl;
    }
    if ((method == "default" && ((!equilibrium && lead_config != "SS") 
                    ||(equilibrium && !finite_temp))) 
            || method == "real_freq_H_time_independent"){
        cutoff_below = stod(method_config["cutoff_below"]);
        cutoff_above = stod(method_config["cutoff_above"]);
        istringstream(method_config["ieta"]) >> ieta;
        istringstream(method_config["init_delta"]) >> init_delta;
        tol_quad = stod(method_config["tol_quad"]);
        TwoTerminalSetup setup(ssyst,gate,bias,leadL,leadR,tLS,tRS,
                cpoint_L,cpoint_R,VCL,VCR);
        VectorXd Hartree0; 
        VectorXcd Delta0; 
        if(scf_initial_path != "" && x0.size() == 2*n_sites_system){
            Hartree0 = x0.head(n_sites_system).real();
            Delta0 = x0.tail(n_sites_system);
        } else{
            Hartree0 = VectorXd::Zero(n_sites_system);
            Delta0 = init_delta*VectorXcd::Ones(n_sites_system);
        }

        converged = setup.self_consistent_loop(std::bind(
                    static_cast<VectorXcd(TwoTerminalSetup::*)
                    (const VectorXcd&,dcomp,double, double, double)>
                    (&TwoTerminalSetup::update),&setup,_1,ieta,
                    cutoff_below, cutoff_above, tol_quad), 
                Hartree0,Delta0,
                scf_config_path,scf_save_path);
        current = VectorXcd::Zero(1);
        current[0] = setup.Current(ieta,cutoff_below, cutoff_above,tol_quad,0,1);
        VectorXcd pnums = setup.ParticleNumber(ieta,cutoff_below,cutoff_above,
                tol_quad);
        particle_numbers = Map<MatrixXcd>(pnums.data(),n_sites_system,1);
        pair_expectations = Map<MatrixXcd>(setup.PairExpectation(ieta,
                    cutoff_below, cutoff_above, tol_quad).data(),n_sites_system,1);
        Glns_time.push_back(setup.get_Gl_time(ieta,
                    cutoff_below, cutoff_above, tol_quad));

        cout << "Current: " <<'\n' << current << endl;
        cout << "Particle numbers: " <<'\n' << particle_numbers << endl;
        cout << "Total particle number: " <<'\n' << particle_numbers.sum() << endl;
        cout << "Pair expectations: " <<'\n' << pair_expectations << endl;
    }

    // Save data
    if(format == "csv" && converged){
        ofstream current_file(data_output_path+"/current.csv");
        ofstream pair_file(data_output_path+"/pair_expectation.csv");
        ofstream particle_file(data_output_path+"/particle_number.csv");

        current_file << scientific << setprecision(10) << current.format(full);
        pair_file << pair_expectations.format(full);
        particle_file << particle_numbers.format(full);
        int count = 0;
        for(auto it = Glns_time.begin(); it != Glns_time.end();++it){
            ofstream Gl_time_file(data_output_path+"/Gl"+to_string(count)+"_time.csv");
            Gl_time_file << (*it).format(full);
            Gl_time_file.close();
        }

        current_file.close();
        pair_file.close();
        particle_file.close();
    }
    if( format == "hdf5" && converged ){
        H5File* dpoint_file = new H5File(data_point_path,H5F_ACC_RDWR);
        Group data_group(dpoint_file->openGroup("/data"));

        H5std_string status("1");
        save_data_to_h5attr(&data_group,"calculated","string", &status);
        save_data_to_h5attr(&data_group, "method", "string", &method);

        if(method == "Pade" || (method == "default" && equilibrium && finite_temp)){
            save_data_to_h5attr(&data_group,"n_approx","int",&n_approx);
        } else {
            save_data_to_h5attr(&data_group,"tol_quad","double",&tol_quad);
            save_data_to_h5attr(&data_group,"cutoff_below","double", &cutoff_below);
            save_data_to_h5attr(&data_group,"cutoff_above","double", &cutoff_above);
            double ieta_imag = ieta.imag();
            save_data_to_h5attr(&data_group,"ieta_imag","double",&ieta_imag);

            if(lead_config == "SS" && !equilibrium){
                save_data_to_h5attr(&data_group,"n_harmonics","int",&n_harmonics);
            }

        }

        for(int i = 0; i < n_harmonics+1;i++){
            cout << "Glns_time[i].rows()" << endl;
            cout << Glns_time[i].rows() << endl;
            write_MatrixXcd_to_group(&data_group,Glns_time[i], "Gl"+to_string(i)+"_time");
        }

        //int count = 0;
        //for(auto it = Glns_time.begin(); it != Glns_time.end(); ++it){
        //    write_MatrixXcd_to_group(&data_group,*it, "Gl"+to_string(count)+"_time");
        //    count++;
        //}

        write_MatrixXcd_to_group(&data_group,Map<MatrixXcd>(current.data(),current.size(),1),"current"); 
        write_MatrixXcd_to_group(&data_group,particle_numbers,"particle_number");
        write_MatrixXcd_to_group(&data_group,pair_expectations,"pair_expectation");
        data_group.close();
        delete dpoint_file;
    }
    return 0;
}
