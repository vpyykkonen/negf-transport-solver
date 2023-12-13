#include <vector>
#include <Eigen/Dense>
#include <string>
#include <map>
#include <iostream>
#include <algorithm>


#include "ScfSolver.h"
#include "ScfMethod.h"
#include "config_parser.h"
#include "file_io.h"

using namespace std;
using namespace Eigen;

typedef complex<double> dcomp;

ScfSolver::ScfSolver(VectorXcd* X, vector<ScfMethod*>& methods, const string save_path):
    X{X}, methods{methods}, save_path{save_path}
{
    iterations = 0;
    converged = false;
    for(auto it = methods.begin(); it != methods.end(); ++it)
        (*it)->set_save_path(save_path);
}

ScfSolver::ScfSolver(VectorXcd* X, const string scf_cfg_path, const string save_path):
    X{X},save_path{save_path}
{
    iterations =  0;
    converged = false;

    vector<map<string,string>> raw_methods;
    int saving_frequency;
    map<string,string> method_strs = load_config_file(scf_cfg_path);
    saving_frequency = stoi(method_strs["saving_frequency"]);
    int n_scf_methods = stoi(method_strs["n_scf_methods"]);
    for(int n = 0; n < n_scf_methods; n++){
        string method_str = method_strs["scf_method"+to_string(n+1)];
        vector<string> method_params = split_str(method_str, ";");
        map<string,string> method;
        method["scf_method"] = method_params[0];
        //cout << "scf_method" << " " << methods_params[0] << endl;
        for(int m = 0; m < (int)method_params.size()-1;m++){
            vector<string> param = split_str(method_params[m+1], "=");
            //cout << param[0] << " " << param[1] << endl;
            method[param[0]] = param[1];
        }
        raw_methods.push_back(method);
    }
    
    try{
        if(raw_methods.empty())
            throw(0);
    }
    catch(int n_methods){
        cout << "Number of methods zero. Quitting. " << "\n";
        return;
    }
    MatrixXcd Jinv;
    int Xsize = X->size();
    for(vector<map<string,string>>::const_iterator it = raw_methods.begin(); it != raw_methods.end();it++){
        int iter = 0;
        string scf_method = it->at("scf_method");
        int max_iter = stoi(it->at("max_iter"));
        double scf_tol = stod(it->at("scf_tol"));
        if(scf_method == "Mixing"){
            double alpha = stod(it->at("alpha"));
            Mixing* mx = new Mixing(X,alpha, iter, max_iter, scf_tol,scf_tol,save_path,saving_frequency);
            methods.push_back(mx);
            cout << "mixing set" << endl;
        } else if(scf_method == "AdaptiveMixing"){
            double alpha = stod(it->at("alpha"));
            double max_alpha = stod(it->at("max_alpha"));
            double min_alpha = stod(it->at("min_alpha"));
            AdaptiveMixing* mx = new AdaptiveMixing(X,alpha,max_alpha,min_alpha, iter, max_iter, scf_tol,scf_tol,save_path,saving_frequency);
            methods.push_back(mx);
            cout << "adaptive mixing set" << endl;

        } else if(scf_method == "PulayPeriodic") {
            double alpha = stod(it->at("alpha"));
            int m = stoi(it->at("m"));
            int p = stoi(it->at("p"));
            PulayPeriodic* pu = new PulayPeriodic(X,alpha,m,p, iter, max_iter, scf_tol, scf_tol,save_path,saving_frequency);
            methods.push_back(pu);
            cout << "pulay periodic set" << endl;
        } else if(scf_method == "BroydenGood") {
            double Jinv0 = stod(it->at("Jinv0"));
            Jinv = Jinv0*MatrixXcd::Identity(Xsize,Xsize);
            BroydenGood* brg = new BroydenGood(X,&Jinv, iter, max_iter, scf_tol, scf_tol,save_path,saving_frequency);
            methods.push_back(brg);
            cout << "broyden good set" << endl;
        } else if(scf_method == "BroydenBad") {
            double Jinv0 = stod(it->at("Jinv0"));
            Jinv = Jinv0*MatrixXcd::Identity(Xsize,Xsize);
            BroydenBad* brb = new BroydenBad(X,&Jinv, iter, max_iter, scf_tol, scf_tol,save_path,saving_frequency);
            methods.push_back(brb);
            cout << "broyden bad set" << endl;
        } else {
            cerr << "Scf method chosen poorly" << endl;
        }
    }
}

ScfSolver::~ScfSolver()
{
    for(auto it = this->methods.begin(); it != this->methods.end(); ++it)
        delete *it;
}

void ScfSolver::iterate(function<VectorXcd(const VectorXcd&)> FX){
    iterations = 0;
    converged = false;
    VectorXcd X0 = *this->X;
    int n = 0;
    for(auto it = methods.begin(); it != methods.end(); ++it){
        n++;
        if(!(*it)->get_converged()){
            (*it)->set_X(this->X);
            (*it)->iterate(FX);
            iterations += (*it)->get_iter();
            converged = (*it)->get_converged();
            if(converged)
                break;
            else{
                cout << "Method " << n << " failed. Continuing...\n";
                *X = X0;
            }
        }
    }
    if(!converged)
        cout << "All methods failed. No convergence." << endl;
}

