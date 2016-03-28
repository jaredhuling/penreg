

#define EIGEN_DONT_PARALLELIZE

#include "ADMMogLassoTall.h"
#include "ADMMogLassoLogisticTall.h"
//#include "ADMMogLassoWide.h"
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
using Rcpp::CharacterVector;

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

RcppExport SEXP admm_oglasso_dense(SEXP x_,
                                   SEXP y_,
                                   SEXP group_,
                                   SEXP family_,
                                   SEXP nlambda_,
                                   SEXP lambda_,
                                   SEXP lambda_min_ratio_,
                                   SEXP group_weights_,
                                   SEXP group_idx_,
                                   SEXP ngroups_,
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
    
    List opts(opts_);
    const int maxit          = as<int>(opts["maxit"]);
    const int irls_maxit     = as<int>(opts["irls_maxit"]);
    const double irls_tol    = as<double>(opts["irls_tol"]);
    const double eps_abs     = as<double>(opts["eps_abs"]);
    const double eps_rel     = as<double>(opts["eps_rel"]);
    const double rho         = as<double>(opts["rho"]);
    const double dynamic_rho = as<double>(opts["dynamic_rho"]);
    bool standardize   = as<bool>(standardize_);
    bool intercept     = as<bool>(intercept_);
    bool intercept_bin = intercept;
    
    const SpMat group(as<MSpMat>(group_));
    CharacterVector family(as<CharacterVector>(family_));
    const MapVec group_weights(as<MapVec>(group_weights_));
    IntegerVector group_idx(group_idx_);
    
    const int ngroups(as<int>(ngroups_));
    
    
    // don't standardize if not linear model. 
    // fit intercept the dumb way if it is wanted
    bool fullbetamat = false;
    int add = 0;
    if (family(0) != "gaussian")
    {
        standardize = false;
        intercept = false;
        
        if (intercept_bin)
        {
            fullbetamat = true;
            add = 1;
            
            VectorXd v(n);
            v.fill(1);
            MatrixXd datX_tmp(n, p+1);
            
            datX_tmp << v, datX;
            datX.swap(datX_tmp);
            
            datX_tmp.resize(0,0);
        }
    }
    
    
    
    // total size of all groups
    const int M(group.sum());
    
    // create C matrix
    //   C_{i,j} = 1 if y_i is a replicate of x_j
    //           = 0 otherwise 
    Eigen::SparseMatrix<double,Eigen::RowMajor> C(Eigen::SparseMatrix<double,Eigen::RowMajor>(M, p));
    C.reserve(VectorXi::Constant(M,1));
    createC(C, group, M);
    
    
    DataStd<double> datstd(n, p + add, standardize, intercept);
    datstd.standardize(datX, datY);
    
    FADMMBase<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> *solver_tall = NULL; // obj doesn't point to anything yet
    //ADMMogLassoTall *solver_tall;
    //ADMMogLassoWide *solver_wide;
    
    
    if(n > 2 * p)
    {
        
        if (family(0) == "gaussian")
        {
            solver_tall = new ADMMogLassoTall(datX, datY, C, n, p, M, ngroups, 
                                              family, group_weights, group_idx, 
                                              dynamic_rho, irls_tol, irls_maxit, 
                                              eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
            solver_tall = new ADMMogLassoLogisticTall(datX, datY, C, n, p, M, ngroups, 
                                                      family, group_weights, group_idx, 
                                                      dynamic_rho, irls_tol, irls_maxit, 
                                                      eps_abs, eps_rel);
        }
    }
    else
    {
        /*
        if (family(0) == "gaussian")
        {
            solver_wide = new ADMMogLassoTallWide(datX, datY, C, n, p, M, ngroups, 
                                              family, group_weights, group_idx, 
                                              dynamic_rho, irls_tol, irls_maxit, 
                                              eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
            solver_wide = new ADMMogLassoLogisticWide(datX, datY, C, n, p, M, ngroups, 
                                                      family, group_weights, group_idx, 
                                                      dynamic_rho, irls_tol, irls_maxit, 
                                                      eps_abs, eps_rel);
        }
         */
    }
    
    if(nlambda < 1)
    {
        double lmax = 0.0;
        if(n > 2 * p)
        {
            lmax = solver_tall->get_lambda_zero() / n * datstd.get_scaleY();
        }
        else
        {
            //lmax = solver_wide->get_lambda_zero() / n * datstd.get_scaleY();
        }
        
        double lmin = as<double>(lambda_min_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }
    
    MatrixXd beta(p + 1, nlambda);
    IntegerVector niter(nlambda);
    double ilambda = 0.0;
    
    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i] * n / datstd.get_scaleY();
        if(n > 2 * p)
        {
            if(i == 0)
                solver_tall->init(ilambda, rho);
            else
                solver_tall->init_warm(ilambda);
            
            niter[i] = solver_tall->solve(maxit);
            VectorXd res = solver_tall->get_gamma();
            double beta0 = 0.0;
            
            // if the design matrix includes the intercept
            // then don't back into the intercept with
            // datastd and include it to beta directly.
            if (fullbetamat)
            {
                beta.block(0, i, p+1, 1) = res;
                datstd.recover(beta0, res);
            } else 
            {
                datstd.recover(beta0, res);
                beta(0,i) = beta0;
                beta.block(1, i, p, 1) = res;
            }
            

            
        } else {
            /*
            if(i == 0)
            solver_wide->init(ilambda, rho);
            else
            solver_wide->init_warm(ilambda);
            
            niter[i] = solver_wide->solve(maxit);
            SpVec res = solver_wide->get_x();
            double beta0 = 0.0;
            if (!fullbetamat)
            {
                datstd.recover(beta0, res);
            }
            write_beta_matrix(beta, i, beta0, res);
            */
        }
    }
    
    // need to deallocate dynamic object
    if(n > 2 * p)
    {
        delete solver_tall;
    }
    else
    {
        //delete solver_wide;
    }
    
    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("niter") = niter);
    
    END_RCPP
}






