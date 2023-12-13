#include <string>
#include <complex>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <map>

#include <filesystem> // C++17 
#include "file_io.h"

#include "H5Cpp.h"
using namespace H5;

using namespace std;
namespace fs = std::filesystem;

using namespace std::complex_literals;
using namespace Eigen;

typedef complex<double> dcomp;

typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixXd_row;

#define MAXBUFSIZE  ((int) 1e6)


vector<string> create_directories(const string& base_path, const vector<string>& directory_names, bool create_new_dirs_if_exists)
{
    // Created subdirectory paths as output
    vector<string> directory_paths;

    // If base directory does not exist, do nothing and return empty list
    if(!fs::exists(base_path))
        return directory_paths;

    // Create the needed directories 
    for(auto it = directory_names.begin(); it != directory_names.end(); it++){
        string directory_path = base_path + "/" + *it;

        //Count the existing directories with the same base name
        unsigned int path_count = 0;
        string count_str = "";
        while(fs::exists(directory_path+count_str)){
            path_count++;
            count_str = to_string(path_count);
        }

        // If the wanted directory does not exist, create new.
        // Else if a new directory is wanted in any case, 
        // create new with a number
        if(!fs::exists(directory_path)){
            fs::create_directory(directory_path);
            directory_paths.push_back(directory_path);
        }
        else if(create_new_dirs_if_exists){
            fs::create_directory(directory_path+"_"+
                    to_string(path_count));
            directory_paths.push_back(directory_path+"_"+
                    to_string(path_count));
        } else 
            directory_paths.push_back(directory_path);
    }
    return directory_paths;
}


MatrixXd readRealMatrix(const char* filename, const char sep)
{
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    if(infile.fail())
        cout << "File could not be opened" << endl;

    while (infile.good())
    {
        string line;
        getline(infile, line);
        if(line.empty())
            break;

        int temp_cols = 0;
        stringstream stream(line);
        string elem;
        while(stream.good()){
            getline(stream,elem,sep);
        // Store in column major fashion
            buff[cols*rows+temp_cols++] = stod(elem);
        }
        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;
        rows++;
    }

    infile.close();

    //rows--;

    //Map<Matrix<double,Dynamic,Dynamic, RowMajor>>
    //result(buff,rows,cols);
    //return result;


    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result(i,j) = buff[ cols*i+j ];
        }
    }

    return result;
};

MatrixXcd readComplexMatrix(const char* filename, const char sep)
{
    int cols = 0, rows = 0;
    dcomp buff[MAXBUFSIZE/2];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    if(infile.fail())
        cout << "File could not be opened" << endl;

    while (infile.good())
    {
        string line;
        getline(infile, line);
        if(line.empty())
            break;

        int temp_cols = 0;
        stringstream stream(line);
        string elem;
        while(stream.good()){
            getline(stream,elem,sep);
            istringstream comp_stream(elem);
            comp_stream >> buff[cols*rows+temp_cols++];
            //char character;
            //double real;
            //double imag;
            //comp_stream >> character >> real >> character >> imag >> character;

            //buff[cols*rows+temp_cols++] = real + 1.0i*imag;
        }
        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;
        rows++;
    }

    infile.close();

    //rows--;
    //
    //Map<Matrix<dcomp,Dynamic,Dynamic, RowMajor>>
    //result(buff,rows,cols);
    //return result;



    // Populate matrix with numbers.
    MatrixXcd result(rows,cols);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result(i,j) = buff[ cols*i+j ];
        }
    }
    return result;
};


map<string,string> load_file_to_string_map(const string& file_path, const string& sep)
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

bool save_string_map_to_file(const map<string,string>& params, const string& sep,const string& path, const string& file_name)
{
    ofstream file(path+"/"+file_name);
    if(!file.is_open()){
        cout << "File : " << path+"/"+file_name << " couldn't be opened." << endl;
        return false;
    }

    for(auto it = params.begin();it != params.end();it++)
        file << it->first  << sep << it->second << endl;

    file.close();
    return true;
}


// Function to separate a string to parts based on a separator string
vector<string> split_str(string input, string sep)
{
    vector<string> output;
    while(!input.empty()){
        size_t sep_pos = input.find(sep);
        if(sep_pos == string::npos){
            output.push_back(input);
            input.erase(input.begin(),input.end());
        }
        else{
            output.push_back(input.substr(0,sep_pos));
            input.erase(input.begin(),input.begin()+sep_pos+1);
        }
    }
    return output;
}

void write_MatrixXcd_to_file(H5File* file, const MatrixXcd& mat, const H5std_string dset_path)
{
    const H5std_string dset_real_path = dset_path+"_r";
    const H5std_string dset_imag_path = dset_path+"_i";

    hsize_t rows = mat.rows(), cols = mat.cols();

    hsize_t dims[2] = {rows,cols};

    // DataSpaces and DataSets for the real and imaginary parts
    DataSpace space(2,dims);
    DataSet* dset_r = new DataSet(file->createDataSet(dset_real_path,PredType::IEEE_F64LE, space));
    DataSet* dset_i = new DataSet(file->createDataSet(dset_imag_path,PredType::IEEE_F64LE, space));

    // Fill buffer with data
    double* buff_r = new double[rows*cols];
    double* buff_i = new double[rows*cols];
    Map<MatrixXd_row>(buff_r, rows,cols) = mat.real();
    Map<MatrixXd_row>(buff_i, rows,cols) = mat.imag();

    // Write to the datasets
    dset_r->write(buff_r,PredType::NATIVE_DOUBLE);
    dset_i->write(buff_i,PredType::NATIVE_DOUBLE);

    // Release resources
    delete dset_r;
    delete dset_i;
    delete[] buff_r;
    delete[] buff_i;
}

MatrixXcd get_MatrixXcd_from_file(H5File* file, const H5std_string dset_path)
{
    MatrixXcd output;
    const H5std_string dset_real_path = dset_path+"_r";
    const H5std_string dset_imag_path = dset_path+"_i";
    DataSet* dset_r = new DataSet(file->openDataSet(dset_real_path));
    DataSet* dset_i = new DataSet(file->openDataSet(dset_imag_path));

    DataSpace r_space = dset_r->getSpace();
    //DataSpace i_space = dset_i->getSpace();
    //
    int n_dims = r_space.getSimpleExtentNdims();
    hsize_t dims[n_dims];
    r_space.getSimpleExtentDims(&dims[0]);

    double* buff_r = new double[dims[0]*dims[1]];
    double* buff_i = new double[dims[0]*dims[1]];

    dset_r->read(buff_r, PredType::NATIVE_DOUBLE);
    dset_i->read(buff_i, PredType::NATIVE_DOUBLE);

    output = Map<MatrixXd_row>(buff_r, dims[0], dims[1]) +      
    Map<MatrixXd_row>(buff_i, dims[0], dims[1])*1.0i;

    delete dset_r;
    delete dset_i;
    delete[] buff_r;
    delete[] buff_i;

    return output;
}

void write_MatrixXd_to_file(H5File* file, const MatrixXd& mat, const H5std_string dset_path)
{
    hsize_t rows = mat.rows(), cols = mat.cols();

    hsize_t dims[2] = {rows,cols};

    // DataSpaces and DataSets for the real and imaginary parts
    DataSpace space(2,dims);
    DataSet* dset = new DataSet(file->createDataSet(dset_path,PredType::IEEE_F64LE, space));
    double* buff = new double[rows*cols];
    Map<MatrixXd_row>(buff, rows,cols) = mat;

    // Write to the datasets
    dset->write(buff,PredType::NATIVE_DOUBLE);

    // Release resources
    delete dset;
    delete[] buff;
}
MatrixXd get_MatrixXd_from_file(H5File* file, const H5std_string dset_path)
{
    DataSet* dset = new DataSet(file->openDataSet(dset_path));

    DataSpace space = dset->getSpace();
    int n_dims = space.getSimpleExtentNdims();
    hsize_t dims[n_dims];
    space.getSimpleExtentDims(&dims[0]);

    MatrixXd output = MatrixXd::Zero(dims[0],dims[1]);

    dset->read(output.data(), PredType::NATIVE_DOUBLE);

    delete dset;

    return output;
}

void write_MatrixXcd_to_group(Group* group, const MatrixXcd& mat, const H5std_string dset_path)
{
    const H5std_string dset_real_path = dset_path+"_r";
    const H5std_string dset_imag_path = dset_path+"_i";

    hsize_t rows = mat.rows(), cols = mat.cols();

    hsize_t dims[2] = {rows,cols};

    // DataSpaces and DataSets for the real and imaginary parts
    DataSpace space(2,dims);
    DataSet* dset_r;
    DataSet* dset_i;
    if(!group->nameExists(dset_real_path)){
        dset_r = new DataSet(group->createDataSet(dset_real_path,PredType::IEEE_F64LE, space));
        dset_i = new DataSet(group->createDataSet(dset_imag_path,PredType::IEEE_F64LE, space));
    } else {
        cout << "Dataset " << dset_path << " already exists. Overwriting" << endl;
        dset_r = new DataSet(group->openDataSet(dset_real_path));
        dset_i = new DataSet(group->openDataSet(dset_imag_path));
    }

    // Fill buffer with data
    double* buff_r = new double[rows*cols];
    double* buff_i = new double[rows*cols];
    Map<MatrixXd_row>(buff_r, rows,cols) = mat.real();
    Map<MatrixXd_row>(buff_i, rows,cols) = mat.imag();

    // Write to the datasets
    dset_r->write(buff_r,PredType::NATIVE_DOUBLE);
    dset_i->write(buff_i,PredType::NATIVE_DOUBLE);

    // Release resources
    delete dset_r;
    delete dset_i;
    delete[] buff_r;
    delete[] buff_i;
}
MatrixXcd get_MatrixXcd_from_group(Group* group, const H5std_string dset_path)
{
    MatrixXcd output;
    const H5std_string dset_real_path = dset_path+"_r";
    const H5std_string dset_imag_path = dset_path+"_i";
    DataSet* dset_r = new DataSet(group->openDataSet(dset_real_path));
    DataSet* dset_i = new DataSet(group->openDataSet(dset_imag_path));

    DataSpace r_space = dset_r->getSpace();
    //DataSpace i_space = dset_i->getSpace();
    //
    int n_dims = r_space.getSimpleExtentNdims();
    hsize_t dims[n_dims];
    r_space.getSimpleExtentDims(&dims[0]);

    double* buff_r = new double[dims[0]*dims[1]];
    double* buff_i = new double[dims[0]*dims[1]];

    dset_r->read(buff_r, PredType::NATIVE_DOUBLE);
    dset_i->read(buff_i, PredType::NATIVE_DOUBLE);

    output = Map<MatrixXd_row>(buff_r, dims[0], dims[1]) +      
    Map<MatrixXd_row>(buff_i, dims[0], dims[1])*1.0i;

    delete dset_r;
    delete dset_i;
    delete[] buff_r;
    delete[] buff_i;

    return output;
}
void write_MatrixXd_to_group(Group* group, const MatrixXd& mat, const H5std_string dset_path)
{
    hsize_t rows = mat.rows(), cols = mat.cols();

    hsize_t dims[2] = {rows,cols};

    // DataSpaces and DataSets for the real and imaginary parts
    DataSpace space(2,dims);
    DataSet* dset = new DataSet(group->createDataSet(dset_path,PredType::IEEE_F64LE, space));

    double* buff = new double[rows*cols];
    Map<MatrixXd_row>(buff, rows,cols) = mat;

    // Write to the datasets
    dset->write(buff,PredType::NATIVE_DOUBLE);

    // Release resources
    delete dset;
}
MatrixXd get_MatrixXd_from_group(Group* group, const H5std_string dset_path)
{
    DataSet* dset = new DataSet(group->openDataSet(dset_path));

    DataSpace space = dset->getSpace();
    int n_dims = space.getSimpleExtentNdims();
    hsize_t dims[n_dims];
    space.getSimpleExtentDims(dims);

    MatrixXd output = MatrixXd::Zero(dims[0],dims[1]);

    dset->read(output.data(), PredType::NATIVE_DOUBLE);

    delete dset;

    return output;
}

bool get_data_from_h5attr(Group* group, string name, string type, void* mem)
{
    if(!group->attrExists(name))
        return false;
    //try{
    //    if(!group->attrExists(name))
    //        throw(false);
    //} catch (bool succ) {
    //    cout << "Cannot open attribute " << name << ": it does not exists." << endl;
    //    return;
    //}
    Attribute attr = group->openAttribute(name);
    if(type == "double")
        attr.read(PredType::NATIVE_DOUBLE,mem);
    if(type == "string"){
        StrType dt(PredType::C_S1,H5T_VARIABLE);
        attr.read(dt,*(string*)mem);
    }
    if(type == "int")
        attr.read(PredType::NATIVE_INT,mem);
    attr.close();
    return true;
}

void save_data_to_h5attr(Group* group, string name, string type, const void* mem)
{
    Attribute attr;

    DataSpace ds(H5S_SCALAR);

    if(type == "string"){
        StrType st(PredType::C_S1,H5T_VARIABLE);
        if( group->attrExists(name))
            attr = group->openAttribute(name);
        else
            attr = group->createAttribute(name,st,ds);
        attr.write(st,*(string*)mem);
        st.close();
    } else if (type == "int"){
        IntType it(PredType::STD_I64LE);
        if( group->attrExists(name))
            attr = group->openAttribute(name);
        else
            attr = group->createAttribute(name,it,ds);
        attr.write(it,mem);
        it.close();
    } else if (type == "double"){
        FloatType ft(PredType::IEEE_F64LE);
        if( group->attrExists(name))
            attr = group->openAttribute(name);
        else
            attr = group->createAttribute(name,ft,ds);
        attr.write(ft,mem);
        ft.close();
    }

    ds.close();
    attr.close();
}
