#define EIGEN_DONT_PARALLELIZE

#include "CoordMCP.h"
#include "DataStd.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;
using Eigen::Map;

using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::IntegerVector;

typedef Map<VectorXd> MapVecd;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

inline void write_beta_matrix(SpMat &betas, int col, double beta0, SpVec &coef)
{
    betas.insert(0, col) = beta0;
    
    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}

RcppExport SEXP coord_mcp(SEXP x_, 
                          SEXP y_, 
                          SEXP lambda_,
                          SEXP gamma_,
                          SEXP penalty_factor_,
                          SEXP nlambda_, 
                          SEXP lmin_ratio_,
                          SEXP standardize_, 
                          SEXP intercept_,
                          SEXP opts_)
{
    BEGIN_RCPP
    
    //Rcpp::NumericMatrix xx(x_);
    //Rcpp::NumericVector yy(y_);
    
    
    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);
    
    const int n = xx.rows();
    const int p = xx.cols();
    
    MatrixXd datX(n, p);
    VectorXd datY(n);
    
    // Copy data and convert type from double to float
    std::copy(xx.begin(), xx.end(), datX.data());
    std::copy(yy.begin(), yy.end(), datY.data());
    
    //MatrixXd datX(as<MatrixXd>(x_));
    //VectorXd datY(as<VectorXd>(y_));
    
    //const int n = datX.rows();
    //const int p = datX.cols();
    
    //MatrixXf datX(n, p);
    //VectorXf datY(n);
    
    // Copy data and convert type from double to float
    //std::copy(xx.begin(), xx.end(), datX.data());
    //std::copy(yy.begin(), yy.end(), datY.data());
    
    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();
    ArrayXd gamma(as<ArrayXd>(gamma_));
    int ngamma = gamma.size();
    
    ArrayXd penalty_factor(as<ArrayXd>(penalty_factor_));
    
    
    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    const double tol       = as<double>(opts["tol"]);
    const bool standardize = as<bool>(standardize_);
    const bool intercept   = as<bool>(intercept_);
    
    DataStd<double> datstd(n, p, standardize, intercept);
    datstd.standardize(datX, datY);
    
    CoordMCP *solver;
    solver = new CoordMCP(datX, datY, tol);
    
    
    
    if (nlambda < 1) {
        
        double lmax = 0.0;
        lmax = solver->get_lambda_zero() / n * datstd.get_scaleY();
        
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }
    
    
    
    
    //SpMat beta(p + 1, nlambda);
    //beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));
    
    MatrixXd beta(p+1, nlambda);
    
    IntegerVector niter(nlambda);
    double ilambda = 0.0;
    
    for (int g = 0; g < ngamma; g++)
    {
        for(int i = 0; i < nlambda; i++)
        {
            ilambda = lambda[i] * n / datstd.get_scaleY();
            
            if(i == 0)
                solver->init(ilambda, gamma[g], penalty_factor);
            else
                solver->init_warm(ilambda, gamma[g]);
            
            niter[i] = solver->solve(maxit);
            VectorXd res = solver->get_beta();
            double beta0 = 0.0;
            datstd.recover(beta0, res);
            beta(0,i) = beta0;
            beta.block(1, i, p, 1) = res;
            //write_beta_matrix(beta, i, beta0, res);
            
        }
    }
    
    delete solver;
    
    //beta.makeCompressed();
    
    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("niter") = niter);
    
    END_RCPP
}

