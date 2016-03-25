#define EIGEN_DONT_PARALLELIZE

#include "ADMMGenLassoTall.h"
//#include "ADMMGenLassoWide.h"
#include "DataStd.h"

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::VectorXf;
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
typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseVector<float> SpVecf;
typedef Eigen::SparseMatrix<float> SpMatf;
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

RcppExport SEXP admm_genlasso(SEXP x_, 
                              SEXP y_, 
                              SEXP D_,
                              SEXP lambda_,
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
    
    const SpMat D(as<MSpMat>(D_));
    
    const int M = D.rows();
    
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
    
    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    const double eps_abs   = as<double>(opts["eps_abs"]);
    const double eps_rel   = as<double>(opts["eps_rel"]);
    const double rho       = as<double>(opts["rho"]);
    const bool standardize = as<bool>(standardize_);
    const bool intercept   = as<bool>(intercept_);
    
    DataStd<double> datstd(n, p, standardize, intercept);
    datstd.standardize(datX, datY);
    
    ADMMGenLassoTall *solver_tall;
    //ADMMGenLassoWide *solver_wide;
    
    if(n > p)
    {
        solver_tall = new ADMMGenLassoTall(datX, datY, D, eps_abs, eps_rel);
    }
    else
    {
        //solver_wide = new ADMMGenLassoWide(datX, datY, eps_abs, eps_rel);
    }
    
    if(nlambda < 1)
    {
        double lmax = 0.0;
        if(n > p)
        {
            lmax = solver_tall->get_lambda_zero() / n * datstd.get_scaleY();
        }
        else
        {
            lmax = solver_tall->get_lambda_zero() / n * datstd.get_scaleY();
        }
            //lmax = solver_wide->get_lambda_zero() / n * datstd.get_scaleY();
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }
    
    MatrixXd beta(p + 1, nlambda);
    SpMat beta_aug(M + 1, nlambda);
    //beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));
    beta_aug.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, M)));
    
    IntegerVector niter(nlambda);
    double ilambda = 0.0;
    
    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i] * n / datstd.get_scaleY();
        if(n > p)
        {
            if(i == 0)
                solver_tall->init(ilambda, rho);
            else
                solver_tall->init_warm(ilambda);
            
            niter[i] = solver_tall->solve(maxit);
            SpVec res = solver_tall->get_z();
            VectorXd restrue = solver_tall->get_x();
            double beta0 = 0.0;
            double beta0a = 0.0;
            datstd.recover(beta0, restrue);
            datstd.recover(beta0a, res);
            //write_beta_matrix(beta, i, beta0, restrue);
            beta(0,i) = beta0;
            beta.block(1, i, p, 1) = restrue;
            write_beta_matrix(beta_aug, i, beta0a, res);
        } else {
            /*
            if(i == 0)
                solver_wide->init(ilambda, rho);
            else
                solver_wide->init_warm(ilambda);
            
            niter[i] = solver_wide->solve(maxit);
            SpVec res = solver_wide->get_x();
            double beta0 = 0.0;
            datstd.recover(beta0, res);
            write_beta_matrix(beta, i, beta0, res);
             */
        }
    }
    
    if(n > p)
    {
        delete solver_tall;
    }
    else
    {
        //delete solver_wide;
    }
    
    beta_aug.makeCompressed();
    
    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("beta.aug") = beta_aug,
                        Named("niter") = niter);
    
    END_RCPP
}

