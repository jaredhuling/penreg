#ifndef COORDBASE_H
#define COORDBASE_H

#include <RcppEigen.h>
#include "Linalg/BlasWrapper.h"
#include "utils.h"


template<typename VecTypeX>
class CoordBase
{
protected:
    
    const int nvars;      // dimension of beta
    const int nobs;       // number of rows
    
    VecTypeX beta;        // parameters to be optimized
    VecTypeX beta_prev;   // auxiliary parameters
    
    double tol;           // tolerance for convergence
    
    virtual void next_beta(VecTypeX &res) = 0;
    
    virtual bool converged()
    {
        return (stopRule(beta, beta_prev, tol));
    }
    
    
    void print_row(int iter)
    {
        const char sep = ' ';
        
        Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << iter;
        Rcpp::Rcout << std::endl;
    }
    void print_footer()
    {
        const int width = 80;
        Rcpp::Rcout << std::string(width, '=') << std::endl << std::endl;
    }
    
public:
    CoordBase(int n_, int p_,
              double tol_ = 1e-6) :
    nvars(p_), nobs(n_),
    beta(p_), beta_prev(p_), // allocate space but do not set values
    tol(tol_)
    {}
    
    virtual ~CoordBase() {}
    
    void update_beta()
    {
        //VecTypeX newbeta(nvars);
        next_beta(beta);
        //beta.swap(newbeta);
    }
    
    int solve(int maxit)
    {
        int i;
        
        for(i = 0; i < maxit; ++i)
        {
            beta_prev = beta;
            // old_y = dual_y;
            //std::copy(dual_y.data(), dual_y.data() + dim_dual, old_y.data());
            
            update_beta();
            
            // print_row(i);
            
            if(converged())
                break;
            
        }
        
        // print_footer();
        
        return i + 1;
    }
    
    virtual VecTypeX get_beta() { return beta; }
};



#endif // COORDBASE_H

