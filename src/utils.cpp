

#include "utils.h"

double threshold(double num) 
{
  return num > 0 ? num : 0;
}

// computes cumulative sum of vector x
VectorXd cumsum(const VectorXd& x) {
  const int n(x.size());
  VectorXd cmsm(n);
  //cmsm = std::partial_sum(x.data(), x.data() + x.size(), cmsm.data(), std::plus<double>());
  cmsm(0) = x(0);
  
  for (int i = 1; i < n; i++) {
    cmsm(i) = cmsm(i-1) + x(i);
  }
  return (cmsm);
}

// computes reverse cumulative sum of vector x
VectorXd cumsumrev(const VectorXd& x) {
  const int n(x.size());
  VectorXd cmsm(n);
  //std::reverse(x.data(), x.data() + x.size());
  //cmsm = std::partial_sum(x.data(), x.data() + x.size(), cmsm.data(), std::plus<double>());
  cmsm(0) = x(n-1);
  //double tmpsum = 0;
  
  for (int i = 1; i < n; i++) {
    //tmpsum += cmsm(i-1);
    cmsm(i) = cmsm(i-1) + x(n-i-1);
  }
  std::reverse(cmsm.data(), cmsm.data() + cmsm.size());
  return (cmsm);
}



//computes X'WX where W is diagonal (input w as vector)
MatrixXd XtWX(const MatrixXd& xx, const MatrixXd& ww) {
  const int n(xx.cols());
  MatrixXd AtWA(MatrixXd(n, n).setZero().
    selfadjointView<Lower>().rankUpdate(xx.adjoint() * ww.array().sqrt().matrix().asDiagonal()));
  return (AtWA);
}

//computes X'WX where W is diagonal (input w as vector)
MatrixXd XWXt(const MatrixXd& xx, const MatrixXd& ww) {
  const int n(xx.rows());
  MatrixXd AWAt(MatrixXd(n, n).setZero().
    selfadjointView<Lower>().rankUpdate(xx * ww.array().sqrt().matrix().asDiagonal()));
  return (AWAt);
}

//computes X'X
MatrixXd XtX(const MatrixXd& xx) {
  const int n(xx.cols());
  MatrixXd AtA(MatrixXd(n, n).setZero().
    selfadjointView<Lower>().rankUpdate(xx.adjoint()));
  return (AtA);
}

//computes XX'
MatrixXd XXt(const MatrixXd& xx) {
  const int n(xx.rows());
  MatrixXd AAt(MatrixXd(n, n).setZero().
    selfadjointView<Lower>().rankUpdate(xx));
  return (AAt);
}

//computes X'X
SpMat XtX(const SpMat& xx) {
  const int n(xx.cols());
  SpMat AtA(SpMat(n, n).selfadjointView<Upper>().rankUpdate(xx.adjoint()));
  return (AtA);
}

//computes XX'
SpMat XXt(const SpMat& xx) {
  const int n(xx.rows());
  SpMat AAt(SpMat(n, n).selfadjointView<Upper>().rankUpdate(xx));
  return (AAt);
}

//computes X'WX where W is diagonal (input w as vector)
SpMat XtWX(const SpMat& xx, const MatrixXd& ww) {
  const int n(xx.cols());
  SpMat AtWA(SpMat(n, n).
    selfadjointView<Lower>().rankUpdate(xx.adjoint() * ww.array().sqrt().matrix().asDiagonal()));
  return (AtWA);
}

//computes X'WX where W is diagonal (input w as vector)
SpMat XWXt(const SpMat& xx, const MatrixXd& ww) {
  const int n(xx.rows());
  SpMat AWAt(SpMat(n, n).
    selfadjointView<Lower>().rankUpdate(xx * ww.array().sqrt().matrix().asDiagonal()));
  return (AWAt);
}


bool stopRule(const VectorXd& cur, const VectorXd& prev, const double& tolerance) {
  for (unsigned i = 0; i < cur.rows(); i++) {
    if ( (cur(i) != 0 && prev(i) == 0) || (cur(i) == 0 && prev(i) != 0) ) {
      return 0;
    }
    if (cur(i) != 0 && prev(i) != 0 && 
        std::abs( (cur(i) - prev(i)) / prev(i)) > tolerance) {
  	  return 0;
    }
  }
  return 1;
}

//   C_{i,j} = 1 if y_i is a replicate of x_j
//           = 0 otherwise 
void createC(SpMatR &C, const SpMat& group, const int& M) {  

  int row_idx = 0;
  for (int k=0; k < group.outerSize(); ++k) 
  {
    for (SpMat::InnerIterator it(group, k); it; ++it)
    {
      C.insert(row_idx, it.row()) = 1.0;
      ++row_idx;
    }
  }
  
  C.makeCompressed();  
}

//computes X'WX where W is diagonal (input w as vector)
/*SparseMatrix<double> XtWX_sparse(const SparseMatrix<double>& xx, const MatrixXd& ww) {
  const int n(xx.cols());
  SparseMatrix<double> AtWA(n, n);
  AtWA = AtWA.selfadjointView<Lower>().rankUpdate(xx.adjoint() * ww.asDiagonal());
  return (AtWA);
}*/

