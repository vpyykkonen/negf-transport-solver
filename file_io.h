#ifndef FILE_IO
#define FILE_IO

#include <string>
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "H5Cpp.h"
using namespace H5;
#include <filesystem> // C++17 
using namespace std;
namespace fs = std::filesystem;

using namespace std::complex_literals;

using namespace Eigen;

// Creates a set of directories to a base folder 
// It is assumed that the base folder exists, otherwise does not do anything
// In case the directories to be created exists, new can be created with additional _number, 
// where number is the number of already existed versions, if wanted
vector<string> create_directories(const string& base_path, const vector<string>& directory_names, bool create_new_dirs_if_exist = false);

// Saves a container to a csv files.
// Assumes the iterated calss supports
template <typename Iterator> bool save_container_as_csv(Iterator begin, Iterator end, string file_name, string path, unsigned int precision = 15, vector<string> header = vector<string>(), bool overwrite = false)
{
    unsigned int size = distance(begin,end);
    bool write_header = !header.empty();
    if(size == 0){
        cout << "Nothing to be written" << endl;
        return false;
    }
    if(write_header && header.size() != size){
        cout << "Container and header do not have compatible sizes!" << endl;
        return false;
    }
    if(!overwrite && fs::exists(path+"/"+file_name)){
        cout << "File : " << path+"/"+file_name << " already exists. Not overwriting." << endl;
    }

    ofstream file(path+"/"+file_name);
    if(!file.is_open()){
        cout << "File : " << path+"/"+file_name << " couldn't be opened." << endl;
        return false;
    }

    file << std::scientific;
    file << std::setprecision(precision);

    if(write_header){
        int count = 0;
        for(string str : header){
            count++;
            file << str;
            if(count != size)
                file << ";";
        }
        file << endl;
    }

    int count = 0;
    while(begin!=end){
        count++;
        file << *begin;
        if( count != size)
            file << ";";
        begin++;
    }
    file << endl;
    file.close();

    return true;
}

// Saves a matrix as csv
template<class T> 
bool save_matrix_as_csv(const vector<vector<T>>& mat,const string& file_name, const string& path, unsigned int precision = 15,const vector<string> h_header = vector<string>(0), const vector<string> v_header = vector<string>(0),bool overwrite = false)
{
    int rows = mat.size();
    int cols = mat[0].size();
    bool write_h_header = !h_header.empty();
    bool write_v_header = !v_header.empty();
    if(rows == 0){
        cout << "Nothing to be written" << endl;
        return false;
    }
    if(write_h_header && h_header.size() != cols){
        cout << "Container and horizontal header do not have compatible sizes!" << endl;
        return false;
    }
    if(write_v_header && v_header.size() != rows){
        cout << "Container and vertical header do not have compatible sizes!" << endl;
        return false;
    }
    if(!overwrite && fs::exists(path+"/"+file_name)){
        cout << "File : " << path+"/"+file_name << " already exists. Not overwriting." << endl;
        return false;
    }

    ofstream file(path+"/"+file_name);
    if(!file.is_open()){
        cout << "File : " << path+"/"+file_name << " couldn't be opened." << endl;
        return false;
    }

    file << std::scientific;
    file << std::setprecision(precision);

    if(write_h_header){
        int count = 0;
        for(string str : h_header){
            count++;
            file << str;
            if(count != cols)
                file << ";";
        }
        file << endl;
    }

    for(auto row_it = mat.begin(); row_it != mat.end(); row_it ++){
        int count = 0;
        int cols = row_it->size();
        if(write_v_header)
            file << v_header[row_it-mat.begin()] << ";";
        for(auto col_it = row_it->begin(); col_it != row_it->end();col_it++){
            count ++;
            file << *col_it;
            if(count != cols)
                file << ";";
        }
        file << endl;
    }

    file.close();
    return true;

}

// Load a matrix from csv
template<class T>
bool load_matrix_from_csv(vector<vector<T>>& mat, const string& path, bool is_h_header = false,vector<string>& h_header = vector<string>(0) , bool is_v_header = false, vector<string>& v_header = vector<string>(0))
{
    ifstream file(path);
    if(!file.is_open()){
        cout << "File : " << path << " couldn't be opened." << endl;
        return false;
    }

    if(is_h_header){
        string header_str;
        getline(file,header_str);
        //h_header = split_str(header_str,";");
        while(!header_str.empty()){
            size_t sep_pos = header_str.find(";");
            if(sep_pos == string::npos){
                h_header.push_back(header_str);
                header_str.erase(header_str.begin(),header_str.end());
            }
            else{
                h_header.push_back(header_str.substr(0,sep_pos));
                header_str.erase(header_str.begin(),header_str.begin()+sep_pos+1);
            }
        }
    }

    string line;

    int count = 0;
    while(getline(file,line)){
        vector<string> num_strs;
        while(!line.empty()){
            size_t sep_pos = line.find(";");
            if(sep_pos == string::npos){
                num_strs.push_back(line);
                line.erase(line.begin(),line.end());
            }
            else{
                num_strs.push_back(line.substr(0,sep_pos));
                line.erase(line.begin(),line.begin()+sep_pos+1);
            }
        }
        // Get the vertical header
        if(is_v_header && (count > 0 || !is_h_header)){
            v_header.push_back(num_strs[0]);
            num_strs.erase(num_strs.begin());
        }

        vector<T> row;
        for(auto it = num_strs.begin(); it != num_strs.end(); it++){
            istringstream is(*it);
            T num; is >> num;
            row.push_back(num);
        }
        mat.push_back(row);
    }

    file.close();

    count++;

    return true;

}

MatrixXd readRealMatrix(const char* path, const char sep = ',');
MatrixXcd readComplexMatrix(const char* path, const char sep = ';' );

template<typename M>
M readMatrix(const char* path,const char sep = ',')
{
    int rows = 0, cols = 0;
    vector<typename M::Scalar> vals;
    ifstream in_file;
    in_file.open(path);
    while(in_file.good()){
        string line;
        getline(in_file,line);
        if(line.empty())
            continue;

        int temp_cols = 0;
        stringstream stream(line);
        string elem;
        while(stream.good()){
            getline(stream,elem,sep);
            istringstream elem_stream(elem);
            typename M::Scalar val; 
            elem_stream >> val;
            vals.push_back(val);
            temp_cols++;
        }
        if (temp_cols == 0)
            continue;
        if (cols == 0)
            cols = temp_cols;
        rows++;
    }
    in_file.close();

    Map<Matrix<typename M::Scalar,
        M::RowsAtCompileTime,
        M::ColsAtCompileTime,
        RowMajor>> 
            result(vals.data(),rows,cols);
    return result;

}

// Load parameter file into a map<string,string>
map<string,string> load_file_to_string_map(const string& file_path, const string& sep);

// Save map<string,string> to a parameter file
bool save_string_map_to_file(const map<string,string>& params, const string& sep,const string& path, const string& file_name);


template<typename T>
string complex_to_string(complex<T> comp, unsigned int precision = 10)
{
    T real = comp.real();
    T imag = comp.imag();
    ostringstream oss;
    oss << std::scientific;
    oss << std::setprecision(precision);
    oss << "(" << real << "," << imag <<")";
    return oss.str();
}

template<typename T>
string num_to_str(T num, unsigned int precision = 10)
{
    ostringstream oss;
    oss << std::scientific;
    oss << std::setprecision(precision);
    oss << num;
    return oss.str();
}


// Function to separate a string to parts based on a separator string
vector<string> split_str(string input, string sep);

// Hdf5 functions
void write_MatrixXcd_to_file(H5File* file, const MatrixXcd& mat, const H5std_string dset_path);
MatrixXcd get_MatrixXcd_from_file(H5File* file, const H5std_string dset_path);
void write_MatrixXd_to_file(H5File* file, const MatrixXd& mat, const H5std_string dset_path);
MatrixXd get_MatrixXd_from_file(H5File* file, const H5std_string dset_path);

void write_MatrixXcd_to_group(Group* group, const MatrixXcd& mat, const H5std_string dset_path);
MatrixXcd get_MatrixXcd_from_group(Group* group, const H5std_string dset_path);
void write_MatrixXd_to_group(Group* group, const MatrixXd& mat, const H5std_string dset_path);
MatrixXd get_MatrixXd_from_group(Group* group, const H5std_string dset_path);

bool get_data_from_h5attr(Group* group, string name, string type, void* mem);

void save_data_to_h5attr(Group* group, string name, string type, const void* mem);
#endif
