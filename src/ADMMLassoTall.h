#ifndef ADMMLASSOTALL_H
#define ADMMLASSOTALL_H

#include "FADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Spectra/SymEigsSolver.h"
#include "ADMMMatOp.h"
#include "utils.h"

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
class ADMMLassoTall: public FADMMBase<Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::LLT<Matrix> LLT;
    typedef Eigen::LDLT<Matrix> LDLT;
    
    MapMat datX;                  // data matrix
    MapVec datY;                  // response vector
    Vector XY;                    // X'Y
    MatrixXd XX;                  // X'X
    LDLT solver;                  // matrix factorization
    VectorXd savedEigs;           // saved eigenvalues
    bool rho_unspecified;         // was rho unspecified? if so, we must set it
    ArrayXd penalty_factor;       // penalty multiplication factors 
    
    Scalar lambda;                // L1 penalty
    Scalar lambda0;               // minimum lambda to make coefficients all zero
    
    
    // x -> Ax
    void A_mult (Vector &res, Vector &beta)  { res.swap(beta); }
    // y -> A'y
    void At_mult(Vector &res, Vector &nu)  { res.swap(nu); }
    // z -> Bz
    void B_mult (Vector &res, SparseVector &gamma) { res = -gamma; }
    // ||c||_2
    double c_norm() { return 0.0; }
    
    
    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty, const Vector &pen_fact)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            double total_pen = pen_fact(i) * penalty;
            
            if(ptr[i] > total_pen)
                res.insertBack(i) = ptr[i] - total_pen;
            else if(ptr[i] < -total_pen)
                res.insertBack(i) = ptr[i] + total_pen;
        }
    }
    
    void next_beta(Vector &res)
    {
        Vector rhs = XY - adj_nu;
        // rhs += rho * adj_gamma;
        
        // manual optimization
        for(SparseVector::InnerIterator iter(adj_gamma); iter; ++iter)
            rhs[iter.index()] += rho * iter.value();
        
        res.noalias() = solver.solve(rhs);
    }
    
    virtual void next_gamma(SparseVector &res)
    {
        Vector vec = main_beta + adj_nu / rho;
        soft_threshold(res, vec, lambda / rho, penalty_factor);
    }
    
    void next_residual(Vector &res)
    {
        // res = main_beta;
        // res -= aux_gamma;
        
        // manual optimization
        std::copy(main_beta.data(), main_beta.data() + dim_main, res.data());
        for(SparseVector::InnerIterator iter(aux_gamma); iter; ++iter)
            res[iter.index()] -= iter.value();
    }
    void rho_changed_action() 
    {
        MatrixXd matToSolve(XX);
        matToSolve.diagonal().array() += rho;
        
        // precompute LLT decomposition of (X'X + rho * I)
        solver.compute(matToSolve.selfadjointView<Eigen::Lower>());
    }
    //void update_rho() {}
    
    
    // Calculate ||v1 - v2||^2 when v1 and v2 are sparse
    static double diff_squared_norm(const SparseVector &v1, const SparseVector &v2)
    {
        const int n1 = v1.nonZeros(), n2 = v2.nonZeros();
        const double *v1_val = v1.valuePtr(), *v2_val = v2.valuePtr();
        const int *v1_ind = v1.innerIndexPtr(), *v2_ind = v2.innerIndexPtr();
        
        double r = 0.0;
        int i1 = 0, i2 = 0;
        while(i1 < n1 && i2 < n2)
        {
            if(v1_ind[i1] == v2_ind[i2])
            {
                double val = v1_val[i1] - v2_val[i2];
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
        double r = std::max(main_beta.norm(), aux_gamma.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return dual_nu.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual()
    {
        return rho * std::sqrt(diff_squared_norm(aux_gamma, old_gamma));
    }
    double compute_resid_combined()
    {
        // SparseVector tmp = aux_gamma - adj_gamma;
        // return rho * resid_primal * resid_primal + rho * tmp.squaredNorm();
        
        // manual optmization
        return rho * resid_primal * resid_primal + rho * diff_squared_norm(aux_gamma, adj_gamma);
    }
    
public:
    ADMMLassoTall(ConstGenericMatrix &datX_, 
                  ConstGenericVector &datY_,
                  ArrayXd &penalty_factor_,
                  double eps_abs_ = 1e-6,
                  double eps_rel_ = 1e-6) :
    FADMMBase<Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd>
               (datX_.cols(), datX_.cols(), datX_.cols(),
              eps_abs_, eps_rel_),
              datX(datX_.data(), datX_.rows(), datX_.cols()),
              datY(datY_.data(), datY_.size()),
              penalty_factor(penalty_factor_),
              XY(datX.transpose() * datY),
              XX(XtX(datX)),
              lambda0(XY.cwiseAbs().maxCoeff())
    {}
    
    double get_lambda_zero() const { return lambda0; }
    
    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_beta.setZero();
        aux_gamma.setZero();
        dual_nu.setZero();
        
        adj_gamma.setZero();
        adj_nu.setZero();
        
        lambda = lambda_;
        rho = rho_;
        
        //MatrixXd XX(XtX(datX));
        //Matrix XX;
        //Linalg::cross_prod_lower(XX, datX);
        
        if(rho <= 0)
        {
            MatOpSymLower<Double> op(XX);
            Spectra::SymEigsSolver< Double, Spectra::LARGEST_ALGE, MatOpSymLower<Double> > eigs(&op, 1, 3);
            srand(0);
            eigs.init();
            eigs.compute(100, 0.1);
            savedEigs = eigs.eigenvalues();
            rho = std::pow(savedEigs[0], 1.0 / 3) * std::pow(lambda, 2.0 / 3);
        }
        
        //XX.diagonal().array() += rho;
        //solver.compute(XX.selfadjointView<Eigen::Lower>());
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        
        adj_a = 1.0;
        adj_c = 9999;
        
        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_beta, aux_gamma, dual_nu and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        
        // adj_a = 1.0;
        // adj_c = 9999;
    }
};



#endif // ADMMLASSOTALL_H