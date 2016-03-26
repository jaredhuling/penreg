#ifndef ADMMOGLASSOTALL_H
#define ADMMOGLASSOTALL_H

#include "FADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Spectra/SymEigsSolver.h"
#include "ADMMMatOp.h"
#include "utils.h"

using Rcpp::IntegerVector;
using Rcpp::CharacterVector;

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// x => beta
// z => -X * beta
// A => X
// b => y
// f(x) => 1/2 * ||Ax - b||^2
// g(z) => lambda * ||z||_1
class ADMMogLassoTall: public FADMMBase<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<Double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SpMatIntR;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::LLT<Matrix> LLT;
    
    MapMat datX;                  // data matrix
    MapVec datY;                  // response vector
    const SpMatR C;               // pointer to C matrix
    //const MapVec D;             // pointer D vector
    // VectorXd D;
    
    int nobs;                 // number of observations
    int nvars;                // number of variables
    int ngroups;              // number of groups
    int M;                    // length of nu (total size of all groups)
    
    Vector XY;                    // X'Y
    MatrixXd XX;                  // X'X
    VectorXd CC;                  // C'C diagonal
    VectorXd Cbeta;               // C * beta
    VectorXd savedEigs;           // saved eigenvalues
    VectorXd group_weights;       // group weight multipliers
    CharacterVector family;       // model family (gaussian, binomial, or Cox PH)
    IntegerVector group_idx;      // indices of groups
    
    LLT solver;                   // matrix factorization
    double newton_tol;            // tolerance for newton iterations
    int newton_maxit;             // max # iterations for newton-raphson
    bool dynamic_rho;
    bool rho_unspecified;         // was rho unspecified? if so, we must set it
    
    Scalar lambda;                // L1 penalty
    Scalar lambda0;               // minimum lambda to make coefficients all zero
    
    //Eigen::DiagonalMatrix<double, Eigen::Dynamic> one_over_D_diag; // diag(1/D)
    SparseMatrix<double,Eigen::ColMajor> CCol;
    
    
    virtual void block_soft_threshold(VectorXd &gammavec, VectorXd &d, 
                                      const double &lam, const double &step_size) 
    {
        // This thresholding function is for the most
        // basic overlapping group penalty, the 
        // l1/l2 norm penalty, ie 
        //     lambda * sqrt(beta_1 ^ 2 + beta_2 ^ 2 + ...)
        
        // d is the vector to be thresholded
        // gammavec is the vector to be written to
        
        int itrs = 0;
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double ds_norm = (d.segment(group_idx(g), group_idx(g+1) - group_idx(g))).norm();
            double thresh_factor = std::max(0.0, 1 - step_size * lam * group_weights(g) / (ds_norm) );
            
            for (int gr = group_idx(g); gr < group_idx(g+1); ++gr) 
            {
                gammavec(itrs) = thresh_factor * d(gr);
                ++itrs;
            }
        }
    }   
    
    
    // x -> Ax
    void A_mult (Vector &res, Vector &x)  { res.swap(x); }
    // y -> A'y
    void At_mult(Vector &res, Vector &y)  { res.swap(y); }
    // z -> Bz
    void B_mult (Vector &res, Vector &z) { res = -z; }
    // ||c||_2
    double c_norm() { return 0.0; }
    
    
    
    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }
    
    void next_x(Vector &res)
    {
        Vector rhs = XY - CCol.adjoint() * adj_y;
        rhs += rho * (CCol.adjoint() * adj_z); 
        
        res.noalias() = solver.solve(rhs);
    }
    virtual void next_z(Vector &res)
    {
        Cbeta = CCol * main_x;
        Vector vec = Cbeta + adj_y / rho;
        block_soft_threshold(res, vec, lambda, 1/rho);
    }
    void next_residual(Vector &res)
    {
        res = Cbeta;
        res -= aux_z;
    }
    void rho_changed_action() {}
    void update_rho() {}
    
    
    
    // Calculate ||v1 - v2||^2 when v1 and v2 are sparse
    static double diff_squared_norm(const SparseVector &v1, const SparseVector &v2)
    {
        const int n1 = v1.nonZeros(), n2 = v2.nonZeros();
        const double *v1_val = v1.valuePtr(), *v2_val = v2.valuePtr();
        const int *v1_ind = v1.innerIndexPtr(), *v2_ind = v2.innerIndexPtr();
        
        Scalar r = 0.0;
        int i1 = 0, i2 = 0;
        while(i1 < n1 && i2 < n2)
        {
            if(v1_ind[i1] == v2_ind[i2])
            {
                Scalar val = v1_val[i1] - v2_val[i2];
                r += val * val;
                i1++;
                i2++;
            } else if(v1_ind[i1] < v2_ind[i2]) {
                r += v1_val[i1] * v1_val[i1];
                i1++;
            } else {
                r += v2_val[i2] * v2_val[i2];
                i2++;
            }
        }
        while(i1 < n1)
        {
            r += v1_val[i1] * v1_val[i1];
            i1++;
        }
        while(i2 < n2)
        {
            r += v2_val[i2] * v2_val[i2];
            i2++;
        }
        
        return r;
    }
    
    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(Cbeta.norm(), aux_z.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    
    
public:
    ADMMogLassoTall(ConstGenericMatrix &datX_, 
                     ConstGenericVector &datY_,
                     const SpMatR &C_,// const VectorXd &D_, 
                     int nobs_, int nvars_, int M_,
                     int ngroups_,
                     Rcpp::CharacterVector family_,
                     VectorXd group_weights_,
                     Rcpp::IntegerVector group_idx_,
                     bool dynamic_rho_,
                     double newton_tol_ = 1e-5,
                     int newton_maxit_ = 100,
                     double eps_abs_ = 1e-6,
                     double eps_rel_ = 1e-6) :
    FADMMBase<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
             (datX_.cols(), C_.rows(), C_.rows(),
              eps_abs_, eps_rel_),
              datX(datX_.data(), datX_.rows(), datX_.cols()),
              datY(datY_.data(), datY_.size()),
              C(C_),
              nobs(nobs_),
              nvars(nvars_),
              M(M_),
              ngroups(ngroups_),
              newton_tol(newton_tol_),
              newton_maxit(newton_maxit_),
              dynamic_rho(dynamic_rho_),
              group_weights(group_weights_),
              family(family_),
              group_idx(group_idx_),
              XY(datX.transpose() * datY),
              XX(XtX(datX)),
              CCol(Eigen::SparseMatrix<double>(M_, nvars_)),
              CC(nvars_),
              Cbeta(C_.rows()),
              lambda0(XY.cwiseAbs().maxCoeff())
    { }
                       
    double get_lambda_zero() const { return lambda0; }
   
    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
                  
        adj_z.setZero();
        adj_y.setZero();
        
        lambda = lambda_;
        rho = rho_;
        
        
        // store ColMajor version of C
        CCol = C;
        
        // create vector CC, whose elements are the number of times
        // each variable is in a group
        for (int k=0; k < CCol.outerSize(); ++k)
        {
            double tmp_val = 0;
            for (SparseMatrix<double>::InnerIterator it(CCol,k); it; ++it)
            { 
                tmp_val += it.value();
            }
            CC(k) = tmp_val;
        } 
        
        //Linalg::cross_prod_lower(XX, datX);
        
        if(rho <= 0)
        {
            rho_unspecified = true;
            MatOpSymLower<Double> op(XX);
            //Spectra::SymEigsSolver< Double, Spectra::LARGEST_ALGE, MatOpSymLower<Double> > eigs(&op, 1, 3);
            Spectra::SymEigsSolver< Double, Spectra::BOTH_ENDS, MatOpSymLower<Double> > eigs(&op, 2, 5);
            srand(0);
            eigs.init();
            eigs.compute(1000, 0.01);
            Vector evals = eigs.eigenvalues();
            savedEigs = evals;
            
            float lam_fact = lambda;
            //rho = std::pow(evals[0], 1.0 / 3) * std::pow(lambda, 2.0 / 3);
            if (lam_fact < savedEigs[1])
            {
                rho = std::sqrt(savedEigs[1] * std::pow(lam_fact * 4, 1.35));
            } else if (lam_fact * 0.25 > savedEigs[0])
            {
                rho = std::sqrt(savedEigs[1] * std::pow(lam_fact * 0.25, 1.35));
            } else 
            {
                rho = std::pow(lam_fact, 1.05);
            }
        } else {
            rho_unspecified = false;
        }
        
        //XX.diagonal().array() += rho;
        
        MatrixXd matToSolve(XX);
        matToSolve.diagonal() += rho * CC;
        
        // precompute LLT decomposition of (X'X + rho * D'D)
        solver.compute(matToSolve.selfadjointView<Eigen::Lower>());
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 1e30;
        resid_dual = 1e30;
        
        adj_a = 1.0;
        adj_c = 1e30;
        
        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        
        if (rho_unspecified)
        {
            float lam_fact = lambda;
            //rho = std::pow(evals[0], 1.0 / 3) * std::pow(lambda, 2.0 / 3);
            if (lam_fact < savedEigs[1])
            {
                rho = std::sqrt(savedEigs[1] * std::pow(lam_fact * 4, 1.35));
            } else if (lam_fact * 0.25 > savedEigs[0])
            {
                rho = std::sqrt(savedEigs[1] * std::pow(lam_fact * 0.25, 1.35));
            } else 
            {
                rho = std::pow(lam_fact, 1.05);
            }
            
        }
        MatrixXd matToSolve(XX);
        matToSolve.diagonal() += rho * CC;
        
        // precompute LLT decomposition of (X'X + rho * C'C)
        solver.compute(matToSolve.selfadjointView<Eigen::Lower>());
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 1e30;
        resid_dual = 1e30;
        
        // adj_a = 1.0;
        // adj_c = 9999;
        rho_changed_action();
    }
    
    virtual VectorXd get_z() { 
        VectorXd beta_return(nvars);
        for (int k=0; k < CCol.outerSize(); ++k)
        {
            int rowidx;
            bool current_zero = false;
            bool already_idx = false;
            for (SparseMatrix<double>::InnerIterator it(CCol,k); it; ++it)
            {
                
                if (aux_z(it.row()) == 0.0 && !current_zero)
                {
                    rowidx = it.row();
                    current_zero = true;
                } else if (!current_zero && !already_idx)
                {
                    rowidx = it.row();
                    already_idx = true;
                }
                
                
            }
            beta_return(k) = aux_z(rowidx);
        }
        return beta_return; 
    }
    
};



#endif // ADMMOGLASSOTALL_H