#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <complex>
#include <vector>
#include <fstream>
#include <map>

using namespace std;

map<string,string> load_config_file(const string& file_path, const string& sep = "=");

#endif 
