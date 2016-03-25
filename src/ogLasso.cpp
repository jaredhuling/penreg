

#define EIGEN_DONT_PARALLELIZE

#include "ADMMogLassoTall.h"
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
    const bool standardize   = as<bool>(standardize_);
    const bool intercept     = as<bool>(intercept_);
    
    const SpMat group(as<MSpMat>(group_));
    CharacterVector family(as<CharacterVector>(family_));
    const MapVec group_weights(as<MapVec>(group_weights_));
    IntegerVector group_idx(group_idx_);
    
    const int ngroups(as<int>(ngroups_));
    
    // total size of all groups
    const int M(group.sum());
    
    // create C matrix
    //   C_{i,j} = 1 if y_i is a replicate of x_j
    //           = 0 otherwise 
    Eigen::SparseMatrix<double,Eigen::RowMajor> C(Eigen::SparseMatrix<double,Eigen::RowMajor>(M, p));
    C.reserve(VectorXi::Constant(M,1));
    createC(C, group, M);
    
    
    DataStd<double> datstd(n, p, standardize, intercept);
    datstd.standardize(datX, datY);
    
    ADMMogLassoTall *solver_tall;
    //ADMMogLassoWide *solver_wide;
    
    if(n > 2 * p)
    {
        solver_tall = new ADMMogLassoTall(datX, datY, C, n, p, M, ngroups, 
                                          family, group_weights, group_idx, 
                                          dynamic_rho, irls_tol, irls_maxit, 
                                          eps_abs, eps_rel);
    }
    else
    {
        //solver_wide = new ADMMogLassoWide(datX, datY, eps_abs, eps_rel);
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
            VectorXd res = solver_tall->get_z();
            double beta0 = 0.0;
            double beta0a = 0.0;
            
            datstd.recover(beta0a, res);
            
            beta(0,i) = beta0;
            beta.block(1, i, p, 1) = res;
            
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
    
    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("niter") = niter);
    
    END_RCPP
}



/*
#include "ADMMogLassoTall.h"
#include "DataStd.h"

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;



using Rcpp::Function;
using Rcpp::Named;
using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::CharacterVector;
using Rcpp::NumericMatrix;
using Rcpp::Environment;
using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;

typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<ArrayXd>  MapArray;


inline void write_beta_matrix(SpMat &betas, int col, double beta0, SpVec &coef)
{
    betas.insert(0, col) = beta0;
    
    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}

RcppExport SEXP  oglasso_fit_dense(SEXP x_,
                                   SEXP y_,
                                   SEXP group_,
                                   SEXP family_,
                                   SEXP nlambda_,
                                   SEXP lambda_,
                                   SEXP lambda_min_ratio_,
                                   SEXP group_weights_,
                                   SEXP group_idx_,
                                   SEXP method_,
                                   SEXP irls_tol_,
                                   SEXP eps_,
                                   SEXP inner_tol_,
                                   SEXP irls_maxit_,
                                   SEXP outer_maxit_,
                                   SEXP inner_maxit_,
                                   SEXP nvars_,
                                   SEXP nobs_,
                                   SEXP ngroups_,
                                   SEXP compute_lambda_,
                                   SEXP standardize_,
                                   SEXP intercept_,
                                   SEXP dynamic_rho_)
{
    try {
        
        MatrixXd x(as<MatrixXd>(x_));
        VectorXd y(as<VectorXd>(y_));
        
        // In glmnet, we minimize
        //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
        // which is equivalent to minimizing
        //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
        
        
        const SpMat group(as<MSpMat>(group_));
        CharacterVector family(as<CharacterVector>(family_));
        const int nlambda(as<int>(nlambda_));
        ArrayXd lambda(as<ArrayXd>(lambda_));
        const double lambda_min_ratio(as<double>(lambda_min_ratio_));
        const MapVec group_weights(as<MapVec>(group_weights_));
        CharacterVector method(method_);
        IntegerVector group_idx(group_idx_);
        
        const double irls_tol(as<double>(irls_tol_));
        const double eps(as<double>(eps_));
        const double inner_tol(as<double>(inner_tol_));
        const int irls_maxit(as<int>(irls_maxit_));
        const int outer_maxit(as<int>(outer_maxit_));
        const int inner_maxit(as<int>(inner_maxit_));
        const int nobs(as<int>(nobs_));
        const int nvars(as<int>(nvars_));
        const int ngroups(as<int>(ngroups_));
        const bool compute_lambda(as<bool>(compute_lambda_));
        const bool dynamic_rho(as<bool>(dynamic_rho_));
        
        bool standardize = as<bool>(standardize_);
        bool intercept = as<bool>(intercept_);
        
        
        // total size of all groups
        const int M(group.sum());
        
        // create C matrix
        //   C_{i,j} = 1 if y_i is a replicate of x_j
        //           = 0 otherwise 
        Eigen::SparseMatrix<double,Eigen::RowMajor> C(Eigen::SparseMatrix<double,Eigen::RowMajor>(M, nvars));
        C.reserve(VectorXi::Constant(M,1));
        createC(C, group, M);
        
        standardize_data data_standardizer(nobs, nvars, standardize, intercept);
        data_standardizer.standardize(x, y);
        
        
        // initialize pointers 
        ADMMogLassoTall *solver_tall;
        ADMMogLassoTall *solver_wide;
        
        // initialize classes
        if(n > p)
        {
            solver_tall = new ADMMogLassoTall(datX, datY, C, eps_abs, eps_rel);
        } else
        {
            solver_wide = new ADMMogLassoTall(datX, datY, C, eps_abs, eps_rel);
        }
        
        
        //ogLassoDense optimizer(x, y, C, nobs, nvars, M, ngroups, 
        //                       family, penalty, method, group_weights, 
        //                       group_idx, eps, dynamic_rho,
        //                       irls_tol, inner_tol);
        
        
        if(compute_lambda) {
            double lmax = optimizer.get_lambda_zero() * data_standardizer.return_scale_y()/ (nobs * alpha);
            double lmin = lambda_min_ratio * lmax;
            lambda.setLinSpaced(nlambda, log(lmax), log(lmin));
            lambda = lambda.exp();
        }
        
        MatrixXd beta(nvars + 1, nlambda);
        //beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(nobs, nvars)));
        
        IntegerVector niter(nlambda);
        double cur_lambda = 0.0;
        double cur_lambda2 = 0.0;
        VectorXd res(nvars);
        
        // fit model for each lambda value and store the results
        for(int i = 0; i < nlambda; i++) {
            cur_lambda  = lambda[i] * nobs * alpha / data_standardizer.return_scale_y();
            cur_lambda2 = lambda[i] * nobs * (1 - alpha) / data_standardizer.return_scale_y();
            if(i == 0) {
                optimizer.init(cur_lambda, cur_lambda2, gamma_mcp);
            } else {
                optimizer.init_warm(cur_lambda, res, cur_lambda2, gamma_mcp);
            }
            
            niter[i] = optimizer.fit(outer_maxit, irls_maxit, inner_maxit);
            res = optimizer.return_beta();
            // initialize intercept to be computed in unstandardize_est
            double beta0 = 0.0;
            data_standardizer.unstandardize_est(beta0, res);
            //write_beta_matrix(beta, i, beta0, res);
            beta(0,i) = beta0;
            beta.block(1, i, nvars, 1) = res;
        }
        
        
        //beta.makeCompressed();
        
        return List::create(Named("beta") = beta,
                            Named("lambda") = lambda,
                            Named("family") = family,
                            Named("penalty") = penalty,
                            Named("niter") = niter,
                            Named("standardized") = standardize,
                            Named("intercept.fit") = intercept);
        
        
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch (...) {
        ::Rf_error("C++ exception (unknown reason)");
    }
    return R_NilValue; //-Wall
}

*/




