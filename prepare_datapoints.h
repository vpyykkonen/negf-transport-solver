#ifndef PREPARE_ANALYSIS
#define PREPARE_ANALYSIS

#include <vector>
#include <string>
#include <array>
#include <map>
#include <memory>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace std;


//vector<string> split_str(string input, string sep);
//vector<string> get_variables_from_map(const map<string,string>& params);
//vector<string> get_variable_ranges_from_map(const map<string,string>& params, const vector<string>& variables);
//
vector<double> get_vector_from_range(const string& range, const double scale = 1.0);

string exec(const char* cmd);

vector<vector<double>> get_grid_from_vectors(const vector<vector<double>>& vectors);

string create_main_folder(const string& output_root,
        map<string,string>& params,
        map<string,string>& geom);
void save_data_point_list_as_csv(string output_path, vector<string> variables, vector<vector<double>> data_points);
//vector<string> save_data_point_folders(string base_path, vector<string> variables, vector<vector<double>> data_points);

void write_bash_execution_file(const string& output_name, const vector<string>& data_point_folders);


#endif
