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

#include "H5Cpp.h"
using namespace H5;

#include <array>
#include <memory>
#include <cstdio>
#include <stdexcept>

#include "prepare_datapoints.h"
#include "file_io.h"
#include "config_parser.h"
//#include "ScatteringSystem.h"

using namespace std;

typedef complex<double> dcomp;

#include <filesystem> // C++17 feature
namespace fs = std::filesystem;


// To Do: write sbatch script for data points automatically once
// the data point analysis program is ready
//
// Ideas: if two variables would be dependent on each other (e.g. VCL,VCR)
// then this could be implied in parameter file (e.g. by VCL+VCR)
// and they would then vary together in the data points
//
// Some scf loops might benefit of using other converged data points as 
// initial quesses (somehow scaled perhaps). This could be implemented e.g.
// as such dependcy list and this could be communicated to the calculation 
// itself as a list of such dependencies
int main(int argc, char* argv[])
{
    // Get parameter file path
    string parameters_path = "parameters.cfg"; // Default
    if(argc == 2)
        parameters_path = argv[1];

    // Load general parameters
    map<string,string> params = load_config_file(parameters_path,"=");

    // Load geometry parameters
    string geometry_path = params["geometry_path"];
    map<string,string> geom = load_config_file(geometry_path,"=");
    //ScatteringSystem ssyst(geometry_path,0.0,0.0); // auxiliary
    //int n_sites = ssyst.get_n_sites();

    // Load self-consistent loop parameters
    //string scf_cfg_path = params["scf_cfg_path"];
    //map<string,string> scf = load_config_file(scf_cfg_path,"=");

    string lead_config = params["lead_config"]; // SS,NN,NS or NN
    // Lead configuration, bools to indicate if lead is superconducting
    bool leadL_sc = false;
    bool leadR_sc = false;
    try {
        if(lead_config == "SS"){
            leadL_sc = true;
            leadR_sc = true;
        } else if(lead_config == "SN"){
            leadL_sc = true;
            leadR_sc = false;
        } else if(lead_config == "NS"){
            leadL_sc = false;
            leadR_sc = true;
        } else if(lead_config == "NN"){
            leadL_sc = false;
            leadR_sc = false;
        } else if(lead_config == "closed"){
            leadL_sc = false;
            leadR_sc = false;
        } else 
            throw(lead_config);
    } catch (string config) {
        cout << "Lead configuration string invalid.\n";
        cout << "The input was: " << config << "\n";
    }

    // Data point save format
    string format = params["format"]; 
    try {
        if(format != "hdf5" && format != "csv")
            throw(format);
    } catch(string form){
        cout << "Datapoint format string invalid.\n";
        cout << "The input was: " << form << "\n";
    }

    bool equilibrium = stoi(params["equilibrium"]);

    // Names of different parameters
    vector<string> lead_param_names{"tL","TL",
                                    "tR"};
    if(leadL_sc){
        lead_param_names.push_back("DeltaL");
        lead_param_names.push_back("phaseL");
    }
    if(leadR_sc){
        lead_param_names.push_back("DeltaR");
        lead_param_names.push_back("phaseR");
    }
    if(!equilibrium)
        lead_param_names.push_back("TR");

    vector<string> lead_ssyst_param_names{"tLS","tRS",
                                        "cpoint_L","cpoint_R",
                                        "VCL","VCR"};
    if(!equilibrium)
        lead_ssyst_param_names.push_back("bias");
    vector<string> ssyst_param_names{"U","gate","disorder","seed"};


    // Names of geometry parameters
    vector<string> geom_param_names{"dim", "unitcell_sites",
                                    "n_unitcells","on_site",
                                    "n_hoppings", "edge"};
    for(int n = 0; n < stoi(geom["n_hoppings"]); n++)
        geom_param_names.push_back("hopping"+to_string(n+1));



    // Get list of variables from the map
    string variables_string = params.at("variables");
    vector<string> variables = split_str(variables_string,",");
    int n_variables = variables.size();

    vector<string> possible_variable_names;
    if(lead_config != "closed"){
        possible_variable_names.insert(possible_variable_names.end(),
                                lead_ssyst_param_names.begin(),
                                lead_ssyst_param_names.end());
        possible_variable_names.insert(possible_variable_names.end(),
                                lead_param_names.begin(),
                                lead_param_names.end());
    }
    possible_variable_names.insert(possible_variable_names.end(),
                                ssyst_param_names.begin(),
                                ssyst_param_names.end());
    possible_variable_names.insert(possible_variable_names.end(),
                                geom_param_names.begin(),
                                geom_param_names.end());
    possible_variable_names.push_back("T");
                                
    // Make the list consistent with the allowed parameter name list
    // by deleting other variables
    for(auto it = variables.begin(); it != variables.end(); it++){
        if(find(possible_variable_names.begin(),
                    possible_variable_names.end(),*it) 
                == possible_variable_names.end()){
            it = variables.erase(it);
            n_variables--;
        }
    }

    // Get variable range strings (e.g. gates = 10:-4.0,4.0)
    vector<string> range_strs;
    for(auto it = variables.begin(); it != variables.end(); it++){
        string variable_name = *it + "s";
        string range_str = params.at(variable_name);
        range_strs.push_back(range_str);
    }

    // Get value vectors for each variables from range strs
    vector<vector<double>> variable_vectors;
    for(int n = 0; n < n_variables;n++){
        vector<double> variable_vector;
        if(variables[n] == "phaseL" || variables[n] == "phaseR"){
            double pi = 3.14159265;
            variable_vector = get_vector_from_range(range_strs[n],pi);
        }
        else
            variable_vector = get_vector_from_range(range_strs[n]);
        variable_vectors.push_back(variable_vector);
    }

    // Generate the data points as a grid of variable vectors
    // (all combinations of the variables,
    // e.g. all points in n_variables dimensional discrete space forming
    // a grid,
    // in Kronecker product basis order)
    vector<vector<double>> data_points;
    data_points = get_grid_from_vectors(variable_vectors);
    int n_data_points = data_points.size();

    // Make a main folder for data points 
    // and include list of the fixed parameters
    // and a list of varied parameters for each data point for reference
    // File format: header "number <var_name_1> <var_name_2> ..."
    string output_root = params["output_root"];
    string fol_name = create_main_folder(output_root,params,geom);
    string output_path = output_root+"/"+fol_name;

    vector<string> data_point_index_str;
    for(int n = 1;n <= n_data_points;n++)
        data_point_index_str.push_back(to_string(n));

    // Compile two terminal system parameters
    vector<string> ttsys_param_names = lead_param_names;
    ttsys_param_names.insert(ttsys_param_names.end(),
                             lead_ssyst_param_names.begin(),
                             lead_ssyst_param_names.end());
    ttsys_param_names.insert(ttsys_param_names.end(),
                            ssyst_param_names.begin(),
                            ssyst_param_names.end());

    // Two terminal system params
    map<string,string> ttsys_params;
    for(auto it = ttsys_param_names.begin();
            it != ttsys_param_names.end();
            it++)
        ttsys_params[*it] = params[*it];



    // Geometry params
    map<string,string> geom_params;
    for(auto it = geom_param_names.begin();
            it != geom_param_names.end();
            it++)
        geom_params[*it] = geom[*it];

    // Save the data points to a file
    save_matrix_as_csv(data_points,"data_points.csv",output_path,10,variables,data_point_index_str,true);
    save_string_map_to_file(ttsys_params, " = ", 
            output_path, "parameters_const.cfg");
    save_string_map_to_file(geom_params, " = ", 
            output_path, "geometry_const.cfg");


    vector<string> data_point_folders;
    if(format == "csv"){
        // csv format
        // Make folder for each data point and insert to folder: 
        // geometry configuration
        // scf configuration
        // list of parameters
        // initial quess for scf loop (size of the system needs to be read from the geometry file first): delta0, hartee0
        // List of things to save, all the scf loop iterations and finally the lesser Green's function in time

        // Naming convention for the folders:
        // Collective folder for an analysis: some descriptive name
        // Single folders with indexing following the data point numbers
        data_point_folders =
            create_directories(output_path,data_point_index_str);
    }

    for(int n = 0; n < n_data_points; n++){
        // Set the data point specific variables (overwrites the constant values)
        for(int m = 0; m < n_variables; m++){
            if(variables[m] == "T"){
                ttsys_params["TL"] = num_to_str(data_points[n][m]);
                ttsys_params["TR"] = num_to_str(data_points[n][m]);
            } else if(find(ttsys_param_names.begin(),
                        ttsys_param_names.end(),
                        variables[m]) !=
                    ttsys_param_names.end())
                ttsys_params[variables[m]] = num_to_str(data_points[n][m]);
            else if(find(geom_param_names.begin(),
                        geom_param_names.end(),
                        variables[m]) !=
                    geom_param_names.end())
                geom_params[variables[m]] = num_to_str(data_points[n][m]);
        }

        // Data point info to be saved
        map<string,string> data_point_info;
        string git_version_hash = exec("git rev-parse HEAD");
        data_point_info["git_version_hash"] = git_version_hash;
        data_point_info["calculated"] = "0";
        string variables_string;
        for(const auto &piece: variables) variables_string += piece+",";
        variables_string.pop_back();
        data_point_info["variables"] = variables_string;
        data_point_info["lead_config"] = lead_config;
        data_point_info["equilibrium"] = params["equilibrium"];

        if(format == "csv"){
            string data_point_folder = data_point_folders[n];
            // Save parameter files
            save_string_map_to_file(ttsys_params, " = ", 
                    data_point_folder, "parameters.cfg");
            save_string_map_to_file(geom_params, " = ", 
                    data_point_folder, "geometry.cfg");

            // Save info on the data point
            save_string_map_to_file(data_point_info, " = ",
                    data_point_folder, "data_point_info.txt");

            // Create directory for scf iterations
            string scf_folder = data_point_folder+"/scf_iterations";
            if(!fs::exists(scf_folder))
                fs::create_directory(scf_folder);

        }
        if(format == "hdf5"){
            H5File* dpoint_file = new H5File(output_path+"/"+to_string(n+1)+".h5", H5F_ACC_TRUNC);
            data_point_folders.push_back(output_path+"/"+to_string(n+1)+".h5");
            Group* data_group = new Group(dpoint_file->createGroup("/data"));
            Group* scf_group = new Group(dpoint_file->createGroup("/scf_iterations"));

            // Save datapoint properties as hdf5 attributes to the root folder
            // (file itself)
            DataSpace ds(H5S_SCALAR);
            // save system parameters as attributes of the datapoint
            for(auto it = ttsys_params.begin(); it != ttsys_params.end(); it++){
                if(it->first == "cpoint_L" || it->first == "cpoint_R"){
                    StrType dt(PredType::C_S1, H5T_VARIABLE);
                    Attribute* ttsys_attr = new Attribute(
                            dpoint_file->createAttribute(it->first,dt,ds));
                    ttsys_attr->write(dt,it->second);
                    delete ttsys_attr;
                } else {
                    FloatType dt(PredType::IEEE_F64LE);
                    Attribute* ttsys_attr = new Attribute(
                            dpoint_file->createAttribute(it->first,dt,ds));
                    double value = stod(it->second);
                    ttsys_attr->write(dt,&value);
                    delete ttsys_attr;
                }
            }
            // geometry
            for(auto it = geom_params.begin(); it != geom_params.end();it++){
                //vector<string> geom_param_names{"dim", "unitcell_sites",
                //                                "n_unitcells","on_site",
                //                                "n_hoppings", "edge"};
                //for(int n = 0; n < stoi(geom["n_hoppings"]); n++)
                //    geom_param_names.push_back("hopping"+to_string(n+1));
                if(it->first.substr(0,7) == "hopping" || it->first == "edge" 
                        || it->first == "n_unitcells" || it->first == "on_site"){
                    StrType dt(PredType::C_S1, H5T_VARIABLE);
                    Attribute* ttsys_attr = new Attribute(
                            dpoint_file->createAttribute(it->first,dt,ds));
                    ttsys_attr->write(dt,it->second);
                    delete ttsys_attr;
                } else {
                    IntType dt(PredType::STD_I32LE);
                    Attribute* ttsys_attr = new Attribute(
                            dpoint_file->createAttribute(it->first,dt,ds));
                    int value = stoi(it->second);
                    ttsys_attr->write(dt,&value);
                    delete ttsys_attr;
                }
            }
            // general info
            for(auto it = data_point_info.begin(); it != data_point_info.end();it++){
                StrType dt(PredType::C_S1, H5T_VARIABLE);
                Attribute* ttsys_attr = new Attribute(
                        dpoint_file->createAttribute(it->first,dt,ds));
                ttsys_attr->write(dt,it->second);
                delete ttsys_attr;
            }
            // sfc properties initialized
            //StrType dt(PredType::C_S1, H5T_VARIABLE);
            //Attribute* scf_attr1 = new Attribute(
            //        scf_group->createAttribute(string("converged"),dt,ds));
            //scf_attr1.write(dt,"false");
            //FloatType dt2(PredType::IEEE_F64LE);
            //Attribute* scf_attr2 = new Attribute(
            //        scf_group->createAttribute(string("rel_error",dt2,ds));
            //scf_attr2.write(dt2,1.0);
            //delete scf_attr1;
            //delete scf_attr2;

            // initial guesses for self-consistent parameters





            // Data and scf folders will gain their own attributes in the end

            delete data_group;
            delete scf_group;
            delete dpoint_file;

        }
    }

    // Write execution bash file
    int sh_count = 0;
    string run_script_path = "./run_scripts";
    if(!fs::exists(run_script_path)){
        fs::create_directory(run_script_path);
        cout << "Run scripts directory: " << run_script_path << " created." << endl;
    }
    string sh_file = run_script_path +"/"+"process_"+fol_name;
    string sh_name = sh_file;
    while(fs::exists(sh_name+".sh")){
        sh_count++;
        sh_name = sh_file + to_string(sh_count);
    }
    write_bash_execution_file(sh_name+".sh",data_point_folders);

    cout << "Data points succesfully prepared at: " << endl;
    cout << output_path << endl;
    cout << "To process data points, run the following" << endl;
    cout << "bash " << sh_name << ".sh" << endl;
    cout << "or at Triton: " << endl;
    cout << "sbatch " << sh_name << ".sh" << endl;

}

// Snippet from 
// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}


// Get a list of varied parameter names
//vector<string> get_variables_from_map(const map<string,string>& params)
//{
//    // Prepare the analysis 
//    string variables_string = params.at("variables");
//    vector<string> variables = split_str(variables_string,",");
//    return variables;
//}


// Get the variable range strings from the map
//vector<string> get_variable_ranges_from_map(const map<string,string>& params, const vector<string>& variables)
//{
//    vector<string> ranges;
//    for(auto it = variables.begin(); it != variables.end(); it++){
//        string variable_name = *it + "s";
//        string range_str = params.at(variable_name);
//        ranges.push_back(range_str);
//    }
//    return ranges;
//}

// Generate sorted vectors of wanted variable values from range strs
vector<double> get_vector_from_range(const string& range, const double scale)
{
    string range_copy(range);
    vector<double> output;
    while(!range_copy.empty()){
        size_t comma_pos = range_copy.find(",");
        string range_str;
        if(comma_pos != string::npos)
            range_str = range_copy.substr(0,comma_pos);
        else
            range_str = range_copy;
        istringstream str(range_str);
        char colon;
        int n;
        double l_value, u_value;
        str >> n >> colon >> l_value >> colon  >> u_value;

        vector<double> new_points;
        // Generate values, in case n=1 only l_value is returned
        if(n == 1) 
            new_points.push_back(l_value);
        else
            for(int i = 0; i < n; i++){
                double value = l_value + (double)i/(n-1)*(u_value-l_value);
                new_points.push_back(value);
            }
        output.insert(output.end(),new_points.begin(),new_points.end());

        if(comma_pos != string::npos)
            range_copy.erase(range.begin(),range_copy.begin()+range_str.length()+1);
        else
            range_copy.erase(range_copy.begin(),range_copy.begin()+range_str.length());
    }

    std::sort(output.begin(),output.end());
    return output;
}

vector<vector<double>> get_grid_from_vectors(const vector<vector<double>>& vectors)
{
    // We use the Kronecker product indexing here for the combinations
    // e.g. the different dimensions are handled as a hierarchy of block structure:
    // the first variables range is represented by the basic block,
    // the second variable is represented by a collection of the basic blocks and so forth
    vector<int> n_values;

    int n_variables = vectors.size();

    for(int n = 0; n < n_variables; n++)
        n_values.push_back(vectors[n].size());

    // Vector for smallest periods of change of a variable in the indexing
    // e.g. first variable changes when incrementing by 1 index,
    // the second variable changes when incrementing by n_values[0],
    // the third variable changes when incremeting by n_values[1]*n_values[0] etc.
    // These are also the respective increments that change the particular value but not
    // any other.
    // Also, var_change_period[var+1] is the size of the smallest complete cycle for var
    vector<int> var_change_periods; 

    // Number of data points to be calculated (all the combinations)
    int n_datapoints = 1;
    for(int var = 0; var < n_variables; var++){
        var_change_periods.push_back(n_datapoints); // Index of first element of variable n
        n_datapoints *= n_values[var];
    }
    var_change_periods.push_back(n_datapoints); // Add also the size of the data at the end for convenience
    
    // Make a list of variable combinations
    // Loop always the inner variable first
    vector<vector<double>> data_points;
    for(int index = 0; index < n_datapoints; index++){
        vector<double> data_point;
        for(int var = 0; var < n_variables; var++){
            // Indexing is explained above but essentially subtract the numbers of
            // of smallest complete cycles by remainder operator and then get
            // the index of the particular variable by diving by the smallest incremental step
            int idx_var = (index%var_change_periods[var+1])/var_change_periods[var];
            data_point.push_back(vectors[var][idx_var]);
        }
        data_points.push_back(data_point);
    }
    return data_points;
}


string create_main_folder(const string& output_root,
        map<string,string>& params,
        map<string,string>& geom)
{
    // Handle the output folder system
    string output_format = params["output_format"];
    vector<string> output_sequence = split_str(output_format,"_");

    istringstream is(params["ieta"]);
    dcomp ieta; is >> ieta;
    is = istringstream(params["DeltaL"]);
    dcomp DeltaL; is >> DeltaL;
    is = istringstream(params["DeltaR"]);
    dcomp DeltaR; is >> DeltaR;

    string lead_config = params["lead_config"];

    ostringstream fol_name;
    fol_name << fixed << setprecision(2);
    for(auto it = output_sequence.begin(); it != output_sequence.end(); ++it){
        bool written_new = false;
        if(*it == "type"){
            fol_name << params["lead_config"];
            if(params["equilibrium"] == "1" && lead_config != "closed")
                fol_name << "_e";
            if(params["equilibrium"] == "0" && lead_config != "closed")
                fol_name << "_ne";
            written_new = true;
        }
        if(*it == "lattice"){
            fol_name << geom["lattice_name"] 
                << geom["n_unitcells"] ;
            written_new = true;
        }
        if(*it == "variables"){
            vector<string> vars = split_str(params["variables"],"_");
            for(auto it = vars.begin(); it != vars.end(); ++it){
                fol_name << *it << params[*it+"s"];
                if(vars.end()-it != 1)
                    fol_name << "_";
            }
            written_new = true;
        }
        if(*it == "note"){
            string output_note = params["output_note"];
            string output_note_in;
            //getline(cin,output_note_in);
            //if(!output_note_in.empty())
            //    output_note = output_note_in;
            fol_name << output_note;
            written_new = true;
        }
        if(*it == "tLS" && lead_config != "closed"){
            fol_name << "tLS" << params["tLS"];
            written_new = true;
        }
        if(*it == "tRS"&& lead_config != "closed"){
            fol_name << "tRS" << params["tRS"];
            written_new = true;
        }
        if(*it == "DeltaL" && (lead_config == "SN" || lead_config == "SS")){
            fol_name << "DeltaL" << abs(DeltaL);
            written_new = true;
        }
        if(*it == "DeltaR" && (lead_config == "NS" || lead_config == "SS")){
            fol_name << "DeltaR" << abs(DeltaR);
            written_new = true;
        }
        if(*it == "phase" &&  lead_config == "SS"){
            fol_name << "phase" << params["phase0"];
            written_new = true;
        }
        if(*it == "phaseL" && (lead_config == "SN" || lead_config == "SS")){
            fol_name << "phaseL" << params["phaseL"];
            written_new = true;
        }
        if(*it == "phaseR" && (lead_config == "NS" || lead_config == "SS")){
            fol_name << "phaseR" << params["phaseR"];
            written_new = true;
        }
        if(*it == "VCL"&& lead_config != "closed"){
            fol_name << "VCL" << params["VCL"];
            written_new = true;
        }
        if(*it == "ieta"){
            fol_name << scientific << setprecision(1);
            fol_name << "ieta" << abs(ieta);
            written_new = true;
        }
        fol_name << fixed << setprecision(2);
        if(*it == "bias"&& lead_config != "closed"){
            fol_name << "bias" << params["bias"];
            written_new = true;
        }
        if(*it == "gate"){
            fol_name << "gate" << params["gate"];
            written_new = true;
        }
        if(*it == "disorder"){
            fol_name << "disorder" << params["disorder"];
            written_new = true;
        }
        if(*it == "seed"){
            fol_name << "seed" << params["seed"];
            written_new = true;
        }
        if(*it == "U"){
            fol_name << "U" << params["U"];
            written_new = true;
        }
        if(*it == "TL" && lead_config != "closed"){
            fol_name << "TL" << params["TL"];
            written_new = true;
        }
        if(*it == "TR" && lead_config != "closed"){
            fol_name << "TR" << params["TR"];
            written_new = true;
        }
        if(*it == "tL" && lead_config != "closed"){
            fol_name << "tL" << params["tL"];
            written_new = true;
        }
        if(it+1 != output_sequence.end() && written_new)
            fol_name << "_";
    }


    // Initialize the folder
    if(!fs::exists(output_root)){
        fs::create_directory(output_root);
        cout << "Output root directory: " << output_root << " created." << endl;
    }
    string folder_name = fol_name.str();
    if(folder_name.substr(folder_name.length()-1,1) == "_")
        folder_name.pop_back();

    string output_path = output_root+"/"+folder_name;

    // Check how many alterations exists with same name and add the new folder
    int path_count = 0;
    string count_str = "";
    while(fs::exists(output_path+count_str)){
        path_count++;
        count_str = "_"+to_string(path_count);
    }
    if(path_count > 0){
        output_path += count_str;
        folder_name += count_str;
    }


    cout << "Output name: " << output_path << endl;
    fs::create_directory(output_path);
    cout << "Output path " << output_path << " created." << endl;

    return folder_name;
    
}

void save_data_point_list_as_csv(string output_path, vector<string> variables, vector<vector<double>> data_points){
    ofstream data_points_file(output_path+"/data_points.csv");

    string header_line = "Number;";
    for(auto it = variables.begin(); it != variables.end(); it++){
        header_line += *it; 
        if(it+1 != variables.end())
            header_line += ";";
    }
    data_points_file << header_line << endl;

    int count = 0;
    for(auto it = data_points.begin(); it != data_points.end(); it++){
        string line = to_string(count)+";";
        for(auto it2 = it->begin(); it2 != it->end(); it2++){
            line += to_string(*it2);
            if(it2+1 != it->end())
                line += ";";
        }
        data_points_file << line << endl;
        count ++;
    }

    data_points_file.close();
}

void write_bash_execution_file(const string& output_name, const vector<string>& data_point_folders)
{
    string format = "csv";
    if(data_point_folders[0].substr(data_point_folders[0].size()-3) == ".h5")
        format = "hdf5";

    string base_folder = data_point_folders[0].substr(0,data_point_folders[0].size()-5);
    if(format == "csv")
        base_folder = data_point_folders[0].substr(0,data_point_folders[0].size()-2);

    ofstream execution_bash(output_name);
    execution_bash << "#!/bin/bash" << endl;
    execution_bash << "#SBATCH --time=25:00:00" << endl;
    execution_bash << "#SBATCH --mem=800M" << endl;
    execution_bash << "#SBATCH --job-name=" << endl;
    execution_bash << "#SBATCH --cpus-per-task=2" << endl;
    execution_bash << "#SBATCH --output="+base_folder+"/output%a.out" << endl;
    execution_bash << "#SBATCH --array=1-" << data_point_folders.size() << endl;
    execution_bash << "n=$SLURM_ARRAY_TASK_ID" << endl;

    execution_bash << endl;

    if(format == "hdf5")
        execution_bash << "#srun ./process_datapoint " + base_folder + "/${n}.h5";
    else
        execution_bash << "#srun ./process_datapoint " + base_folder + "/${n}";

    execution_bash << endl;

    for(auto it = data_point_folders.begin(); it != data_point_folders.end(); it++){
        execution_bash << "./process_datapoint "+*it;
        execution_bash << endl;
    }

    execution_bash.close();

}












