#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <tuple>
#include <complex>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include"pade_frequencies.h"

using namespace std;
using namespace Eigen;

#include <filesystem>
namespace fs = std::filesystem;


#define MAXBUFSIZE  ((int) 1e6)

// Function for sorting frequencies and residues
bool compare(int i, int j, VectorXd freqs)
{
    return freqs(i) < freqs(j);
}


// Calculate Pade (N-1)/N frequencies of Fermi-Dirac distribution
// Note that Pade frequenies are purely imaginary but here considered real
// Input: 
// int n number of frequency. n=1,2,3,...
// int n_approx, how many frequencies in total considered
// Output:
// double, frequency n
// double, residue n
// Note that frequencies are symmetrically on +- imaginary axes
tuple<double,double> pade_frequency(int n, int n_approx)
{
    using namespace std::placeholders;
    static int n_approx_old = 0;
    int nm = 2*n_approx;
    static VectorXd freqs = VectorXd::Zero(1);
    static VectorXd residues = VectorXd::Zero(1);
    VectorXd freq_temp;
    VectorXd resi_temp;
    // Calculate and save if not already calculated
    if(n_approx != n_approx_old){
        // Try to load data from file
        bool success = false;
        cout << "Trying to load data..." << endl;
        tie(success,freq_temp,resi_temp) = load_data("PadeCoefficients",n_approx);
        if(success){
            n_approx_old = n_approx;
            freqs = freq_temp;
            residues = resi_temp;
            return make_tuple(freqs(n-1),residues(n-1));
        }



        freqs.conservativeResize(n_approx);
        residues.conservativeResize(n_approx);

        MatrixXd A = MatrixXd::Zero(nm,nm);
        MatrixXd B = MatrixXd::Zero(nm,nm);
        for(int i = 0; i < nm; i++)
            A(i,i) = -1.0 - i*2.0;
        for(int i = 0; i < nm-1; i++){
            B(i,i+1) = 0.5;
            B(i+1,i) = 0.5;
        }
        MatrixXd Binv = B.inverse();

        GeneralizedEigenSolver<MatrixXd> ges(nm);
        ges.compute(A,B);
        VectorXd eigs = ges.eigenvalues().real();
        MatrixXcd V = ges.eigenvectors();
        MatrixXcd Vinv = V.inverse();
        VectorXd resi = VectorXd::Zero(nm);
        for(int i = 0; i <nm;i++)
            resi(i) = real((eigs(i)*V(0,i)*Vinv.row(i)*Binv.col(0)/4.0)(0,0));

        // sort
        VectorXi idx = VectorXi::LinSpaced(nm,0,nm-1);
        sort(idx.data(),idx.data()+idx.size(),bind(compare,_1,_2,eigs));
        VectorXd tmp1 = VectorXd::Zero(nm);
        VectorXd tmp2 = VectorXd::Zero(nm);
        for(int i = 0; i < nm; i++){
            tmp1(i) = eigs(idx(i));
            tmp2(i) = resi(idx(i));
        }
        freqs = tmp1.segment(n_approx,n_approx);
        residues = tmp2.segment(n_approx,n_approx);
        save_data("PadeCoefficients",freqs,residues);
    }

    n_approx_old = n_approx;


    return make_tuple(freqs(n-1),residues(n-1));
}

void save_data(string base_path, const VectorXd& freqs, const VectorXd& residues)
{
    IOFormat full(FullPrecision, 0, ", ", "\n", "", "","","");
    string folder_name  = to_string(freqs.size());
    if(!fs::exists(base_path))
        fs::create_directory(base_path);
    string path = base_path + "/" + folder_name;
    if(!fs::exists(path))
        fs::create_directory(path);
    ofstream freqs_file, residues_file;
    freqs_file.open(path+"/freqs.csv");
    residues_file.open(path+"/residues.csv");
    if(freqs_file.is_open())
        freqs_file << freqs.format(full);
    if(residues_file.is_open())
        residues_file << residues.format(full);
    freqs_file.close();
    residues_file.close();
}

tuple<bool,VectorXd,VectorXd> load_data(string base_path, int n_approx)
{
    cout << "Loading data" << endl;
    VectorXd freqs = VectorXd::Zero(1);
    VectorXd residues = VectorXd::Zero(1);

    string path = base_path + "/" + to_string(n_approx);
    if(!fs::exists(path+"/freqs.csv")||!fs::exists(path+"/residues.csv")){
        cout << "Files do not exist" << endl;
        return make_tuple(false,freqs,residues);
    }
    cout << "File found!" << endl;

    ifstream freqs_file, residues_file;
    freqs_file.open(path + "/freqs.csv");
    residues_file.open(path + "/residues.csv");

    int freqs_cols = 0, freqs_rows = 0;
    double buff[MAXBUFSIZE];
    
    while (freqs_file.good())
    {
        string line;
        getline(freqs_file, line);
        if(line.empty())
            break;

        int temp_cols = 0;
        stringstream stream(line);
        string elem;
        while(stream.good()){
            getline(stream,elem,',');
            buff[freqs_cols*freqs_rows+temp_cols++] = stod(elem);
        }
        if (temp_cols == 0)
            continue;

        if (freqs_cols == 0)
            freqs_cols = temp_cols;
        freqs_rows++;
    }

    freqs_file.close();

    freqs.resize(freqs_rows);
    for(int i = 0; i < freqs_rows; i++){
        freqs(i) = buff[i];
    }

    int residues_cols = 0, residues_rows = 0;
    
    while (residues_file.good())
    {
        string line;
        getline(residues_file, line);
        if(line.empty())
            break;

        int temp_cols = 0;
        stringstream stream(line);
        string elem;
        while(stream.good()){
            getline(stream,elem,',');
            buff[residues_cols*residues_rows+temp_cols++] = stod(elem);
        }
        if (temp_cols == 0)
            continue;

        if (residues_cols == 0)
            residues_cols = temp_cols;
        residues_rows++;
    }

    residues_file.close();

    residues.resize(residues_rows);
    for(int i = 0; i < residues_rows; i++){
        residues(i) = buff[i];
    }

    return make_tuple(true,freqs,residues);
}
