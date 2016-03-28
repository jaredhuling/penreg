#ifndef ADMMLASSOLOGISTICTALL_H
#define ADMMLASSOLOGISTICTALL_H

#include "FADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Spectra/SymEigsSolver.h"
#include "ADMMMatOp.h"
#include "utils.h"
#include <Eigen/Geometry>

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
class ADMMLassoLogisticTall: public FADMMBase<Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd>
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
    MatrixXd HH;                  // X'WX
    LDLT solver;                  // matrix factorization
    VectorXd savedEigs;           // saved eigenvalues
    double newton_tol;            // tolerance for newton iterations
    int newton_maxit;             // max # iterations for newton-raphson
    bool rho_unspecified;         // was rho unspecified? if so, we must set it
    
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
    
    void next_beta(Vector &res)
    {
        Vector rhs = XY - adj_nu;
        // rhs += rho * adj_gamma;
        
        // manual optimization
        for(SparseVector::InnerIterator iter(adj_gamma); iter; ++iter)
            rhs[iter.index()] += rho * iter.value();
        
        res.noalias() = solver.solve(rhs);
    }
    
    void next_beta_logistic(Vector &res)
    {
        // this function doesn't work quite right
        res = main_beta;
        //LDLT solver_logreg;
        int maxit_newton = 100;
        double tol_newton = 1e-5;
        
        for (int i = 0; i < maxit_newton; ++i)
        {
            // calculate gradient
            
            VectorXd prob = 1 / (1 + (-1 * (datX * res).array()).exp().array());
            
            VectorXd grad = (-1 * XY.array()).array() + (datX.adjoint() * prob).array() + 
                adj_nu.array() + (rho * res.array()).array();
            
            
            for(SparseVector::InnerIterator iter(adj_gamma); iter; ++iter)
                grad[iter.index()] -= rho * iter.value();
            
            //calculate Jacobian
            VectorXd W = prob.array() * (1 - prob.array());
            HH = XtWX(datX, W);
            HH.diagonal().array() += rho;
            
            VectorXd dx = HH.ldlt().solve(grad);
            res.noalias() -= dx;
            if (std::abs(grad.adjoint() * dx) < tol_newton)
            {
                //std::cout << "iters:\n" << i+1 << std::endl;
                break;
            }
        }
        
    }
    virtual void next_gamma(SparseVector &res)
    {
        Vector vec = main_beta + adj_nu / rho;
        soft_threshold(res, vec, lambda / rho);
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
    
    void compute_rho()
    {
        if (rho_unspecified)
        {
            MatOpSymLower<Double> op(XX);
            Spectra::SymEigsSolver< Double, Spectra::LARGEST_ALGE, MatOpSymLower<Double> > eigs(&op, 1, 3);
            srand(0);
            eigs.init();
            eigs.compute(100, 0.1);
            savedEigs = eigs.eigenvalues();
            rho = std::pow(savedEigs[0], 1.0 / 3) * std::pow(lambda, 2.0 / 3);
        }
    }
    
    
    
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
    ADMMLassoLogisticTall(ConstGenericMatrix &datX_, 
                          ConstGenericVector &datY_,
                          double newton_tol_ = 1e-5,
                          int newton_maxit_ = 100,
                          double eps_abs_ = 1e-6,
                          double eps_rel_ = 1e-6) :
    FADMMBase<Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd>
             (datX_.cols(), datX_.cols(), datX_.cols(),
              eps_abs_, eps_rel_),
              newton_tol(newton_tol_),
              newton_maxit(newton_maxit_),
              datX(datX_.data(), datX_.rows(), datX_.cols()),
              datY(datY_.data(), datY_.size()),
              XY(datX.transpose() * datY),
              XX(datX_.rows(), datX_.cols()),
              lambda0(XY.cwiseAbs().maxCoeff())
    {}
    
    virtual double get_lambda_zero() const { return lambda0; }
    
    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double rho_)
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
            rho_unspecified = true;
        } else 
        {
            rho_unspecified = false;
        }
        
        
        //XX.diagonal().array() += rho;
        //solver.compute(XX.selfadjointView<Eigen::Lower>());
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        
        adj_a = 1.0;
        adj_c = 9999;
        
    }
    // when computing for the next lambda, we can use the
    // current main_beta, aux_gamma, dual_nu and rho as initial values
    virtual void init_warm(double lambda_)
    {
        lambda = lambda_;
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        
        // adj_a = 1.0;
        // adj_c = 9999;
    }
    
    virtual int solve(int maxit)
    {
        
        VectorXd beta_prev;
        
        int i;
        int j;
        for (int i = 0; i < newton_maxit; ++i)
        {
            
            
            VectorXd W;
            VectorXd prob;
            VectorXd grad;
            
            beta_prev = main_beta;
            
            // calculate gradient
            prob = 1 / (1 + (-1 * (datX * main_beta).array()).exp().array());
            
            grad = (-1 * XY.array()).array() + (datX.adjoint() * prob).array();
            
            // calculate Jacobian
            W = prob.array() * (1 - prob.array());
            
            // compute X'WX
            XX = XtWX(datX, W);
            
            // compute X'Wz
            XY = XX * main_beta + datX.adjoint() * (datY.array() - prob.array()).matrix();
            
            // compute rho after X'WX is computed
            compute_rho();
            
            // reset LDLT solver with new XX
            rho_changed_action();
            
            
            if (i > 0) 
            {
                // reset values that need to be reset 
                // for ADMM
                // and keep lambda the same
                init_warm(lambda);
            }
            
            for(j = 0; j < maxit; ++j)
            {
                old_gamma = aux_gamma;
                // old_nu = dual_nu;
                std::copy(dual_nu.data(), dual_nu.data() + dim_dual, old_nu.data());
                
                update_beta();
                update_gamma();
                update_nu();
                
                // print_row(i);
                
                if(converged())
                    break;
                
                double old_c = adj_c;
                adj_c = compute_resid_combined();
                
                if(adj_c < 0.999 * old_c)
                {
                    double old_a = adj_a;
                    adj_a = 0.5 + 0.5 * std::sqrt(1 + 4.0 * old_a * old_a);
                    double ratio = (old_a - 1.0) / adj_a;
                    adj_gamma = (1 + ratio) * aux_gamma - ratio * old_gamma;
                    adj_nu.noalias() = (1 + ratio) * dual_nu - ratio * old_nu;
                } else {
                    adj_a = 1.0;
                    adj_gamma = old_gamma;
                    // adj_nu = old_nu;
                    std::copy(old_nu.data(), old_nu.data() + dim_dual, adj_nu.data());
                    adj_c = old_c / 0.999;
                }
                // only update rho after a few iterations and after every 40 iterations.
                // too many updates makes it slow.
                if(i > 5 && i % 2500 == 0)
                    update_rho();
            }
            
            VectorXd dx = beta_prev - main_beta;
            if (std::abs(XY.adjoint() * dx) < newton_tol)
            {
                //std::cout << "iters:\n" << i+1 << std::endl;
                break;
            }
            
        }
        // print_footer();
        
        return i + 1;
    }
};



#endif // ADMMLASSOLOGISTICTALL_H