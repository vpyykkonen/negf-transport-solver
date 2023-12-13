#include "ScfMethod.h"
#include <iostream>
#include <functional>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <cstdlib>
#include <future>

#include "file_io.h"
#include "H5Cpp.h"


using namespace std;
using namespace Eigen;
using namespace H5;

typedef complex<double> dcomp;

void ScfMethod::save_iteration(const string path)
{
    if(path.size() == 0)
        return;
    if(path.substr(path.size()-3)== ".h5"){
        H5File* file = new H5File(path,H5F_ACC_RDWR);
        Group scf_group(file->openGroup("/scf_iterations"));
        string iter_str = "iteration"+to_string(iter);
        write_MatrixXcd_to_group(&scf_group,*X,iter_str);

        DataSet dset = scf_group.openDataSet(iter_str +"_r");
        Attribute attr;
        FloatType ft(PredType::IEEE_F64LE);
        DataSpace ds(H5S_SCALAR);
        if(dset.attrExists("error_rel"))
            attr = dset.openAttribute("error_rel");
        else
            attr = dset.createAttribute("error_rel",ft,ds);
        attr.write(ft,&error_rel);
        ft.close();
        ds.close();
        attr.close();
        dset.close();
        scf_group.close();
        delete file;
    } else {
        ofstream iteration_file(path+"/"+to_string(this->iter)+".csv");
        iteration_file << this->error_rel << "\n"; 
        iteration_file << scientific << setprecision(10) << *this->X;
        iteration_file.close();
    }

}

void Mixing::iterate(function<VectorXcd(const VectorXcd&)> FX )
{
    MatrixXcd GX = VectorXcd::Zero(X->size());

    //auto f = std::async(std::launch::async, [&]() {
    //    string character;
    //    cout << "Press X to interrupt\n" << flush;
    //    while( character != "X")
    //        cin >> character;
    //    });


    while(!converged){
        //if(f.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
        //    cout << "Loop manually interrupted\n"; 
        //    cout << "Do you want to stop (X) or change parameter (C)?\n";

        //    string input;
        //    cin >> input;
        //    while(input != "X" && input != "C"){
        //        cout << "Invalid choice. Try again...\n";
        //        cin >> input;
        //    }
        //    if(input == "X")
        //        break;
        //    if(input == "C"){
        //        cout << "Which parameter you want to change?\n";
        //        cout << "Options: tol_rel, tol_abs, max_iter,alpha\n";
        //        cin >> input;
        //        while(input != "tol_rel" && input != "tol_abs" && 
        //                input != "max_iter" && input != "alpha"){
        //            cout << "Invalid choice. Try again...\n";
        //        }
        //        cout << "Give new value: \n";
        //        cin >> input;
        //        char* end = nullptr;
        //        double val = strtod(input.c_str(), &end);
        //        while( end == input.c_str() || *end != '\0' || val == HUGE_VAL){
        //            cout << "Invalid numeric value. Try again...\n";
        //            cin >> input;
        //            val  = strtod(input.c_str(), &end);
        //        }

        //        if(input == "tol_rel")
        //            this->set_tol_rel(val);
        //        if(input == "tol_abs")
        //            this->set_tol_abs(val);
        //        if(input == "max_iter")
        //            this->set_max_iter(int(val));
        //        if(input == "alpha")
        //            this->set_alpha(val);

        //        f = std::async(std::launch::async, [&]() {
        //            string character;
        //            cout << "Press X to interrupt" << flush;
        //            while( character != "X")
        //                cin >> character;
        //            });
        //    }
        //}

        iter++;

        GX = FX(*X)-*X;
        
        
        *X += alpha*GX;

        double error_abs_old = error_abs;
        error_abs = GX.lpNorm<Infinity>();
        error_rel = error_abs/X->lpNorm<Infinity>();
        //if(abs((error_abs_old-error_rel)/error_abs_old) < 1.0e-5){
        //    cout << "Error remains constant. Oscillating. Abort.\n";
        //    break;
        //}
            
        if(iter % saving_frequency == 0){
            cout << "At iteration " << iter << " rel error " << error_rel << endl;
            cout << "At iteration " << iter << " abs error " << error_abs << endl;
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
        }


        if(error_abs < tol_abs || error_rel < tol_rel){
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
            converged = true;
        }
        if(iter >= max_iter)
            break;
    }
}

void AdaptiveMixing::iterate(function<VectorXcd(const VectorXcd&)> FX )
{
    MatrixXcd GX = VectorXcd::Zero(X->size());
    double error_old = 1.0;
    double alphar = 0.0;
    error_abs = 1.0;
    while(!converged){
        iter++;

        GX = FX(*X)-*X;
        
        
        *X += alpha*GX;

        error_old = error_abs;
        double error_rel_old = error_rel;
        error_abs = GX.lpNorm<Infinity>();
        error_rel = error_abs/X->lpNorm<Infinity>();

        if(abs((error_rel_old-error_rel)/error_rel_old) < 1.0e-5){
            cout << "Error remains constant. Oscillating. Abort.\n";
            break;
        }

        if(iter % saving_frequency == 0){
            cout << "At iteration " << iter << " rel error " << error_rel << endl;
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
        }

        if(error_abs < tol_abs || error_rel < tol_rel){
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
            converged = true;
        }
        if(iter >= max_iter)
            break;

        // Update alpha according to errors
        alphar = alpha/(1.0-error_abs/error_old);
        if(alphar < 1e-5)
            alpha = 1e-5;
        else if(alphar > 0.8)
            alpha = 0.8;
        else
            alpha = alphar;
    }
}


void BroydenGood::iterate(function<VectorXcd(const VectorXcd&)> FX)
{
    VectorXcd DeltaX;
    VectorXcd DeltaG;
    VectorXcd GX = FX(*X)-*X;
    VectorXcd Xold;
    VectorXcd GXold;
    MatrixXcd Jinv;

    error_abs = GX.lpNorm<Infinity>();
    error_rel = error_abs/X->lpNorm<Infinity>();
    //if(error_abs < tol_abs || error_rel < tol_rel)
    //    converged = true;

    while(!converged){
        iter++;
        if(iter >= max_iter)
            break;

        Jinv = *inv_Jacobian; // temp to prevent Eigen to mix matrix elements

        Xold = *X;
        (*X) -= Jinv*GX;

        DeltaX = (*X)-Xold;
        
        GXold = GX;
        GX = FX(*X)-*X;

        DeltaG = GX-GXold;

        // predictor - corrector, run twice with same Jinv
        //Xold = *X;
        //*X -= Jinv*GX;
        //DeltaX = *X-Xold;
        //
        //GXold = GX;
        //GX = FX(*X)-*X;

        //// run a third time
        //Xold = *X;
        //*X -= Jinv*GX;
        //DeltaX = *X-Xold;
        //
        //GXold = GX;
        //GX = FX(*X)-*X;
        //

        double error_rel_old = error_rel;

        error_abs =  GX.lpNorm<Infinity>();
        error_rel =  error_abs/X->lpNorm<Infinity>();

        if(abs((error_rel_old-error_rel)/error_rel_old) < 1.0e-5){
            cout << "Error remains constant. Oscillating. Abort.\n";
            break;
        }

        if(iter % saving_frequency == 0){
            cout << "At iteration " << iter << " rel error " << error_rel << endl;
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
        }

        if(error_abs < tol_abs || error_rel < tol_rel){
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
            converged = true;
        }


        // update inverse Jacobian
        // correct formula by complex optimization
        *inv_Jacobian = Jinv + (DeltaX-Jinv*DeltaG)*DeltaX.adjoint()*Jinv/(DeltaX.adjoint()*Jinv*DeltaG);
        
        if(iter >= max_iter)
            break;

    }
}

void BroydenBad::iterate(function<VectorXcd(const VectorXcd&)> FX)
{
    VectorXcd DeltaX;
    VectorXcd DeltaG;
    VectorXcd GX = FX(*X)-*X;
    VectorXcd Xold;
    VectorXcd GXold;
    MatrixXcd Jinv;

    error_abs = GX.lpNorm<Infinity>();
    error_rel = error_abs/X->lpNorm<Infinity>();
    if(error_abs < tol_abs || error_rel < tol_rel)
        converged = true;

    while(!converged){
        iter++;
        if(iter >= max_iter)
            break;

        Jinv = *inv_Jacobian; // temp to prevent Eigen to mix matrix elements

        Xold = *X;
        *X -= Jinv*GX;
        DeltaX = *X-Xold;
        
        GXold = GX;
        GX = FX(*X)-*X;

        // predictor - corrector, run twice with same Jinv
        //*X -= Jinv*GX;
        //DeltaX = *X-Xold;
        //
        //GXold = GX;
        //GX = FX(*X)-*X;

        double error_rel_old = error_rel;
        error_abs =  GX.lpNorm<Infinity>();
        error_rel =  error_abs/X->lpNorm<Infinity>();

        if(abs((error_rel_old-error_rel)/error_rel_old) < 1.0e-5){
            cout << "Error remains constant. Oscillating. Abort.\n";
            break;
        }

        if(iter% saving_frequency == 0){
            cout << "At iteration " << iter << " rel error " << error_rel << endl;
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
        }

        if(error_abs < tol_abs || error_rel < tol_rel){
            if(!this->get_save_path().empty()){
                save_iteration(this->get_save_path());
                cout << "saved iteration " << iter << "\n";
            }
            converged = true;
        }
        if(iter >= max_iter)
            break;

        DeltaG = GX-GXold;

        //update inverse Jacobian
        //correct formula with adjoint
        *inv_Jacobian = Jinv + (DeltaX-Jinv*DeltaG)*DeltaG.adjoint()/(DeltaG.adjoint()*DeltaG);

        if(iter >= max_iter)
            break;


    }
}

void PulayPeriodic::iterate(function<VectorXcd(const VectorXcd&)> FX)
{
    int Xsize = X->size();

    VectorXcd GX = *X;
    VectorXcd Xold = VectorXcd::Zero(Xsize);
    VectorXcd Gold = VectorXcd::Zero(Xsize);
    //MatrixXcd A = MatrixXcd::Zero(m,m);
    //VectorXcd b = VectorXcd::Zero(m);
    //VectorXcd Ainvb = VectorXcd::Zero(m);

    // memory of previous iterations, start at size 1
    MatrixXcd Xs = MatrixXcd::Zero(Xsize,1);
    MatrixXcd Gs = MatrixXcd::Zero(Xsize,1);

    MatrixXcd A;
    VectorXcd b;
    VectorXcd Ainvb;

    int counter = 0; // tells how many elements are in Xs and Gs
    // First step, give first value to GX and second value for X
    iter++;
    GX = FX(*X)-*X;
    error_abs = GX.lpNorm<Infinity>();
    error_rel = error_abs/X->lpNorm<Infinity>();
    //if(error_abs < tol_abs || error_rel < tol_Rel)
    //    converged = true;

    Xold = *X;
    *X += alpha*GX;

    while(!converged){
        iter++;
        Gold = GX;
        GX = FX(*X)-*X;

        double error_rel_old = error_rel;

        error_abs = GX.lpNorm<Infinity>();
        error_rel = error_abs/X->lpNorm<Infinity>();


        //if(abs((error_rel_old-error_rel)/error_rel_old) < 1.0e-5){
        //    cout << "Error remains constant. Oscillating. Abort.\n";
        //    break;
        //}

        if(iter % saving_frequency == 0){
            cout << "At iteration " << iter << " rel error " << error_rel << endl;    
            save_iteration(this->get_save_path());
        }

        if(error_abs < tol_abs || error_rel < tol_rel){
            cout << "Converged with error_abs " << error_abs << " and error_rel " << error_rel << "\n";
            save_iteration(this->get_save_path());
            cout << "Saved converged solution " << iter << "\n";
            converged = true;
            break;
        }

        if(counter < m){
            // Update Xs and GS
            Xs.conservativeResize(Xsize,counter+1);
            Gs.conservativeResize(Xsize,counter+1);
            Xs.col(counter) = *X-Xold;
            Gs.col(counter) = GX-Gold;
            counter ++;
        }
        else{
            // Update Xs and GS
            for(int i = 0; i < m-1; i++){
                Xs.col(i) = Xs.col(i+1);
                Gs.col(i) = Gs.col(i+1);
            }
            Xs.col(m-1) = *X-Xold;
            Gs.col(m-1) = GX-Gold;
        }

        // apply Pulay periodically, otherwise mixing
        if(iter % p == 0){
            //A.noalias() = Gs.transpose()*Gs;
            //b.noalias() = Gs.transpose()*GX;
            A.noalias() = Gs.adjoint()*Gs;
            b.noalias() = Gs.adjoint()*GX;
            Ainvb.noalias() = A.colPivHouseholderQr().solve(b);
            Xold = *X;
            X->noalias() += alpha*GX - (Xs+alpha*Gs)*Ainvb;
        }
        else{
            Xold = *X;
            *X +=  alpha*GX;
        }

        
        if(iter >= max_iter)
            break;
    }
}

