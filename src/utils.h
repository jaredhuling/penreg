#ifndef _oem2_UTILS_H
#define _oem2_UTILS_H


#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>
#include <numeric>


using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseMatrix;
using Eigen::Lower;
using Eigen::Upper;
using Eigen::Ref;
typedef Eigen::Triplet<double> T;
typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<ArrayXd>  MapArrayd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;
typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SpMatIntR;

   
double threshold(double num);

VectorXd cumsum(const VectorXd& x);

VectorXd cumsumrev(const VectorXd& x);

//computes X'WX where W is diagonal (input w as vector)
MatrixXd XtWX(const MatrixXd& xx, const MatrixXd& ww);

//computes XWX' where W is diagonal (input w as vector)
MatrixXd XWXt(const MatrixXd& xx, const MatrixXd& ww);

//SpMat X'WX where W is diagonal (input w as vector)
SpMat XtWX(const SpMat& xx, const MatrixXd& ww);

//computes XWX' where W is diagonal (input w as vector)
SpMat XWXt(const SpMat& xx, const MatrixXd& ww);

//computes X'X 
MatrixXd XtX(const MatrixXd& xx);

//computes XX'
MatrixXd XXt(const MatrixXd& xx);

//computes X'X 
SpMat XtX(const SpMat& xx);

//computes XX'
SpMat XXt(const SpMat& xx);



bool stopRule(const VectorXd& cur, const VectorXd& prev, const double& tolerance);

void createC(SpMatR &C, const SpMat& group, const int& M);

#endif