#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include <Eigen/Dense>
#include <vector>

void DFS(const int site, Eigen::VectorXi& visit_count, const Eigen::MatrixXcd& mat, const double tresh);
std::vector<std::vector<int>> unconnected_branches_DFS(const Eigen::MatrixXcd& mat, const double tresh);
int find_class(int idx, const std::vector<std::vector<int>>& a);
Eigen::MatrixXi class_division_table(const std::vector<std::vector<int>>& a);
std::vector<std::vector<int>> unconnected_branches_merge(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);


#endif
