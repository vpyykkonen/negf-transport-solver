#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <algorithm>
#include <vector>

#include "connectivity.h"

using namespace Eigen;

// Depth first search
void DFS(const int site, VectorXi& visit_count, const MatrixXcd& mat, const double tresh)
{
    visit_count(site) += 1;
    // Determine the neighbors
    std::vector<int> neighbors;
    for(int j = 0; j < mat.rows(); j++)
        if((abs(mat(site,j)) > tresh || abs(mat(j,site)) > tresh) && site != j)
            neighbors.push_back(j);
    for(auto it = neighbors.begin(); it != neighbors.end(); ++it)
        if(visit_count[*it] == 0)
            DFS(*it,visit_count,mat,tresh);
}


std::vector<std::vector<int>> unconnected_branches_DFS(const MatrixXcd& mat, const double tresh)
{
    std::vector<std::vector<int>> unconnected;
    VectorXi visit_count = VectorXi::Zero(mat.rows());
    for(int i = 0; i < mat.rows(); i++){
        if(visit_count(i) == 0){
            std::vector<int> connected_branch;
            DFS(i,visit_count,mat,tresh);
            for(int j = 0; j < mat.rows(); j++)
                if(visit_count[j] == 1){
                    connected_branch.push_back(j);
                    visit_count[j] = -1;
                }
            std::sort(connected_branch.begin(),connected_branch.end());
            unconnected.push_back(connected_branch);
        }
    }
    return unconnected;
}

int find_class(int idx, const std::vector<std::vector<int>>& a)
{
    int count = 0;
    for(auto it = a.begin(); it != a.end(); ++it){
        count++;
        if((*it)[0] > idx || (*it)[it->size()-1] < idx)
            continue;
        for(auto jt = it->begin(); jt != it->end(); ++jt)
            if(*jt == idx)
                return count;
    }
    return -1;
}

Eigen::MatrixXi class_division_table(const std::vector<std::vector<int>>& a)
{
    int size = 0;
    for(int i = 0; i < a.size(); ++i)
        size += a[i].size();
    MatrixXi result = MatrixXi::Zero(size,2);
    for(int i = 0; i < a.size(); ++i){
        for(int j = 0; j < a[i].size(); ++j){
            result(a[i][j],0) = i;
            result(a[i][j],1) = j;
        }
    }
    return result;
}

// Get the unconnected components of two merged connectivities
// Idea: loop over classes of a and find for each class the classes b connected to it
// Since connection is an equivalence relation, then set of b classes found are connected 
// and the remaining b classes are not.
std::vector<std::vector<int>> unconnected_branches_merge(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b)
{
    std::vector<std::vector<int>> result;
    VectorXi checked_b = VectorXi::Zero(b.size()); // Already included branches b
    for(auto it = a.begin(); it != a.end(); ++it){
        std::vector<int> b_classes;
        for(auto jt = it->begin(); jt != it->end(); ++jt){
            bool found = false;
            for(int i = 0; i < b.size(); i++){
                if(checked_b(i) == 0){
                    for(int j = 0; j < b[i].size(); ++j){
                        if(b[i][j] == *jt){
                            b_classes.push_back(i);
                            checked_b(i) = 1;
                            found = true;
                            break;
                        }
                    }
                }
                if(found)
                    break;
            }
        }
        if(b_classes.size() == 0) 
            continue;
        std::vector<int> joined_classes;
        for(auto jt = b_classes.begin(); jt != b_classes.end(); ++jt){
            joined_classes.insert(joined_classes.end(),b[*jt].begin(),b[*jt].end());
        }
        std::sort(joined_classes.begin(),joined_classes.end());
        result.push_back(joined_classes);
    }
    return result;
}






//int main(void)
//{
//    int size = 100;
//    MatrixXcd con = MatrixXcd::Zero(size,size);
//    for(int i = 0; i < size-3; i++){
//        con(i,i+3)=1.0;
//        con(i+3,i)=1.0;
//    }
//    con(25,28) = 0.0;
//    con(28,25) = 0.0;
//
//    con(57,60) = 0.0;
//    con(60,57) = 0.0;
//
//    con(77,80) = 0.0;
//    con(80,77) = 0.0;
//
//    MatrixXcd con2 = MatrixXcd::Zero(size,size);
//    con2(0,1) = 1;
//    con2(1,0) = 1;
//
//    //for(int i = 0; i < size-1; i++){
//    //    con(i,i+1) = 1;
//    //    con(i+1,i) = 1;
//    //}
//    std::vector<std::vector<int>> unconnected1 = unconnected_branches_DFS(con,1.0e-10);
//    std::vector<std::vector<int>> unconnected2 = unconnected_branches_DFS(con2,1.0e-10);
//    std::cout << "Connectivities 1\n";
//    std::cout << "Connected branches as rows\n";
//    for(auto it = unconnected1.begin(); it != unconnected1.end(); ++it){
//        for(auto jt = it->begin(); jt != it->end(); ++jt)
//            std::cout << *jt << " ";
//        std::cout << "\n";
//    }
//    std::cout << "Connectivities 2\n";
//    std::cout << "Connected branches as rows\n";
//    for(auto it = unconnected2.begin(); it != unconnected2.end(); ++it){
//        for(auto jt = it->begin(); jt != it->end(); ++jt)
//            std::cout << *jt << " ";
//        std::cout << "\n";
//    }
//
//    std::vector<std::vector<int>> unconnected_joined = unconnected_branches_merge(unconnected1,unconnected2);
//    std::cout << "Joined connectivites\n";
//    std::cout << "Connected branches as rows\n";
//    for(auto it = unconnected_joined.begin(); it != unconnected_joined.end(); ++it){
//        for(auto jt = it->begin(); jt != it->end(); ++jt)
//            std::cout << *jt << " ";
//        std::cout << "\n";
//    }
//}
