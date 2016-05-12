#define EIGEN_DONT_PARALLELIZE

#include "ADMMLassoTallPrecond.h"
#include "ADMMLassoLogisticTall.h"
#include "ADMMLassoWide.h"
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
using Rcpp::CharacterVector;

typedef Map<VectorXd> MapVecd;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

inline void write_beta_matrix(SpMat &betas, int col, double beta0, SpVec &coef, bool startatzero)
{
    
    int add = 0;
    if (!startatzero)
    {
        add = 1;
        betas.insert(0, col) = beta0;
    }
    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + add, col) = iter.value();
    }
}

RcppExport SEXP admm_lasso_precond(SEXP x_, 
                                   SEXP y_, 
                                   SEXP family_,
                                   SEXP lambda_,
                                   SEXP nlambda_, 
                                   SEXP lmin_ratio_,
                                   SEXP penalty_factor_,
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
    
    // Copy data 
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
    const int maxit        = as<int>(opts["maxit"]);
    const int irls_maxit     = as<int>(opts["irls_maxit"]);
    const double irls_tol    = as<double>(opts["irls_tol"]);
    const double eps_abs   = as<double>(opts["eps_abs"]);
    const double eps_rel   = as<double>(opts["eps_rel"]);
    const double rho       = as<double>(opts["rho"]);
    bool standardize   = as<bool>(standardize_);
    bool intercept     = as<bool>(intercept_);
    bool intercept_bin = intercept;
    
    CharacterVector family(as<CharacterVector>(family_));
    ArrayXd penalty_factor(as<ArrayXd>(penalty_factor_));
    
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
            // dont penalize the intercept
            ArrayXd penalty_factor_tmp(p+1);
            
            penalty_factor_tmp << 0, penalty_factor;
            penalty_factor.swap(penalty_factor_tmp);
            
            VectorXd v(n);
            v.fill(1);
            MatrixXd datX_tmp(n, p+1);
            
            datX_tmp << v, datX;
            datX.swap(datX_tmp);
            
            datX_tmp.resize(0,0);
        }
    }
    
    DataStd<double> datstd(n, p + add, standardize, intercept);
    datstd.standardize(datX, datY);
    
    // initialize pointers 
    FADMMBasePrecond<Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd> *solver_tall = NULL; // obj doesn't point to anything yet
    ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd, Eigen::VectorXd> *solver_wide = NULL; // obj doesn't point to anything yet
    //ADMMLassoTall *solver_tall;
    //ADMMLassoWide *solver_wide;
    

    // initialize classes
    if(n > 2 * p)
    {
        solver_tall = new ADMMLassoTallPrecond(datX, datY, penalty_factor, eps_abs, eps_rel);
    } else
    {
        solver_wide = new ADMMLassoWide(datX, datY, penalty_factor, eps_abs, eps_rel);
    }

    
    if (nlambda < 1) {
        
        double lmax = 0.0;
        
        if(n > 2 * p) 
        {
            lmax = solver_tall->get_lambda_zero() / n * datstd.get_scaleY();
        } else
        {
            lmax = solver_wide->get_lambda_zero() / n * datstd.get_scaleY();
        }
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }


    SpMat beta(p + 1, nlambda);
    beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));

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
            SpVec res = solver_tall->get_gamma();
            double beta0 = 0.0;
            if (!fullbetamat)
            {
                datstd.recover(beta0, res);
            }
            write_beta_matrix(beta, i, beta0, res, fullbetamat);
        } else {
            
            if(i == 0)
                solver_wide->init(ilambda, rho);
            else
                solver_wide->init_warm(ilambda, i);

            niter[i] = solver_wide->solve(maxit);
            SpVec res = solver_wide->get_beta();
            double beta0 = 0.0;
            if (!fullbetamat)
            {
                datstd.recover(beta0, res);
            }
            write_beta_matrix(beta, i, beta0, res, fullbetamat);
            
        }
    }
    

    if(n > 2 * p) 
    {
        delete solver_tall;
    }
    else
    {
        delete solver_wide;
    }

    beta.makeCompressed();

    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("niter") = niter);

END_RCPP
}
