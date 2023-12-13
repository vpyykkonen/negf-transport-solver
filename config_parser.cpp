#include "config_parser.h"

#include <string>
#include <complex>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>

using namespace std;

map<string,string> load_config_file(const string& file_path, const string& sep)
{
    ifstream file(file_path);
    map<string,string> output;

    // Enter the file and 
    if(file.is_open()){
        string line;
        while(getline(file,line)){
            line.erase(remove_if(line.begin(),line.end(), ::isspace),line.end());
            if(line[0] == '#' || line.empty())
                continue;
            size_t commentPos = line.find("#");
            if(commentPos != string::npos)
                line.erase(line.begin()+commentPos,line.end());


            size_t delimiterPos = line.find(sep);
            string name = line.substr(0,delimiterPos);
            string value = line.substr(delimiterPos+sep.length());
            output[name] = value;
        }
    } else {
        cerr << "Couldn't open the parameter file for reading.\n";
    }
    return output;
}

