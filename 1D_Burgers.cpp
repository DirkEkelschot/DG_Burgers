#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/Polylib.h"
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
using namespace std;
using namespace polylib;
void CalculateRHS_Modal(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt);
void CalculateRHS(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis);
void *negatednormals(int Nel, double *n);
int **iarray(int n,int m);
std::vector<std::vector<double> > getNodalBasis(std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getModalBasis(std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);

std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
void GetFwdBwd(int eln, int Nel, int np, double *bc, double *quad, double *UtL, double *UtR);
void GetAllFwdBwd(int Nel, int np, double *bc, double *quad, std::vector<double> &UtL, std::vector<double> &UtR);
void TraceMap(int np, int Nel, int **trace);
std::vector<double> BackwardTransformLagrange(int P, std::vector<double> zquad, std::vector<double> wquad, int nq, double J, std::vector<double> coeff, std::vector<double> z, int np);
void InnerProductWRTDerivBasis(int np, int P, double J, double *w, double *z, double **D, double *F_DG, double *coeff);
void GetGlobalMassMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double *Jac, int **map, int Mdim, double **MassMatGlobal);
void GetGlobalMassMatrix_Modal(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad,  std::vector<double> z, double *Jac, int **map, int Mdim, double **MassMatGlobal);
void *basis(int np, int P, int i, double *z, double *phi);
void getbasis(int np, int P, int ncoeff, double *z, double **basis);
void getdbasis(int np, int P, double J, double **D, double **basis, double **dbasis);
void *CalcResidual(int np, int Nel, int P, int **map, double *Jac, double *x_G, double *u_DG, double *bound,double *u_Res);
void *diff(int np, double **D, double *p, double *pd, double J);
void *elbound(int Nel, double *bound,double Ldom,double Rdom);
int **imatrix(int nrow, int ncol);
void *mapping(int Nel, int P, int **map);
double integr(int np, double *w, double *phi1, double *phi2);
double *dvector(int np);
int *ivector(int n);
void *chi(int np, int eln, double *x,double *z, double *Jac, double *bound);
void *basis(int np, int P, int i, double *z, double *phi);
double **dmatrix(int Mdim);
double **darray(int n,int m);
void evaluateflux(int np, double *u_DG, double *Flux_DG);
void GetElementStiffnessMatrix(int np, int nq, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double J, double **StiffMatElem);
void GetElementStiffnessMatrixNew(int P, std::vector<double> wquad, double **D, double J, std::vector<std::vector<double> > basis, double **StiffMatElem);
void GetGlobalStiffnessMatrixNew(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal);

void GetGlobalStiffnessMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, int Mdim, double **StiffnessGlobal);
std::vector<double> modal_basis(int np, int P, int i, std::vector<double> z);
void GetGlobalStiffnessMatrix_Modal(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, int Mdim, double **StiffnessGlobal);
void GetElementStiffnessMatrix_Modal(int np, int nq, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double J, double **StiffMatElem);
void GetGlobalMassMatrixNew(int Nel, int P, std::vector<double> wquad, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **MassMatGlobal);
std::vector<double> ForwardTransform(int P, int np, std::vector<std::vector<double> > basis,  std::vector<double>wquad, int nq, double J, std::vector<double> input_quad);

extern "C" {extern void dgetrf_(int *, int *, double (*), int *, int [], int*);}
extern "C" {extern void dgetrs_(unsigned char *, int *, int *, double (*), int *, int [], double [], int *, int *);}
extern "C" {extern void dgemv_(unsigned char *, int *, int *, double *, double (*), int *, double [], int *, double *, double [], int *);}
extern "C" {extern void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);}



void *lagrange_basis(int np, int P, int i, double *z, double *zwgll, double *phi)
{
    for (int q = 0; q < np; ++q)
    {
        phi[q] = hglj(i, z[q], zwgll, P, 0.0, 0.0);
    }

    return 0;
}


std::vector<double> getLagrangeBasisFunction(int i, std::vector<double> zquad, int nq, std::vector<double> z, int np, int P)
{

    if(nq != zquad.size())
    {
        std::cout << "error: nq != zquad.size() " << std::endl;
    }
    int numModes = P + 1;

    std::vector<double> phi1(nq);
    for (int q = 0; q < nq; ++q)
    {
        phi1[q] = hglj(i, zquad[q], z.data(), numModes, 0.0, 0.0);
    }
    return phi1;
}









std::vector<double> GetElementMassMatrix(int P, 
                            std::vector<std::vector<double> > basis,
                            std::vector<double> wquad,
                            double J)
{
    std::vector<double> MassMatElem((P+1)*(P+1),0.0);
    int np = wquad.size();
    for(int i=0;i<P+1;i++)
    {
        std::vector<double> phi1 = basis[i];
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = basis[j];
            MassMatElem[i*(P+1)+j] = J*integr(np, wquad.data(), phi1.data(), phi2.data());
        }
    }
    return MassMatElem;
}





















std::vector<double> ForwardTransform(int P, 
                                     int np,
                                     std::vector<std::vector<double> > basis, 
                                     std::vector<double>wquad, int nq,
                                     double J, 
                                     std::vector<double> input_quad)
{
    
    std::vector<double> coeff(P+1);
    int ncoeffs     = P+1;
    double *Icoeff  = dvector(ncoeffs);
    
    for(int j=0;j<P+1;j++)
    {
        std::vector<double> phi1 = basis[j];

        Icoeff[j] = J*integr(nq, wquad.data(), phi1.data(), input_quad.data());
    }

    std::vector<double> MassMatElem = GetElementMassMatrix(P,basis,wquad,J);

    int ONE_INT=1;
    double ONE_DOUBLE=1.0;
    double ZERO_DOUBLE=0.0;
    unsigned char TR = 'T';
    int INFO;
    int LWORK = ncoeffs*ncoeffs;
    double *WORK = new double[LWORK];
    int *ip = ivector(ncoeffs);
    // Create inverse Mass matrix.
    dgetrf_(&ncoeffs, &ncoeffs, MassMatElem.data(), &ncoeffs, ip, &INFO);
    dgetri_(&ncoeffs, MassMatElem.data(), &ncoeffs, ip, WORK, &LWORK, &INFO);
    // Apply InvMass to Icoeffs hence M^-1 Icoeff = uhat
    dgemv_(&TR,&ncoeffs,&ncoeffs,&ONE_DOUBLE,MassMatElem.data(),&ncoeffs,Icoeff,&ONE_INT,&ZERO_DOUBLE,coeff.data(),&ONE_INT);
    return coeff;
}


















std::vector<double> BackwardTransform(int P, 
                                      int np, 
                                      std::vector<std::vector<double> > basis,  
                                      std::vector<double> input_coeff)
{

    std::vector<double> quad(np,0.0);
    double sum = 0.0;
    for(int i = 0;i<P+1;i++)
    {
        std::vector<double> phi1 =basis[i];
        for( int j=0;j<np;j++)
        {
            quad[j] = quad[j]+input_coeff[i]*phi1[j];
        }
    }

    return quad;
}



















std::vector<double> FilterNodalCoeffs(std::vector<double> zquad, 
                                      std::vector<double> wquad, 
                                      std::vector<double> z, 
                                      int np, int nq, 
                                      std::vector<double> coeffs_modal, int P, int Pf, double J)
{



    std::vector<double> coeffs_update(P+1,0.0);

    int ncoeffs     = P+1;
    
    for(int n=0;n<(Pf+1);n++)
    {
        coeffs_update[n]=coeffs_modal[n];
    }
    
    return coeffs_update;

}






std::vector<std::vector<double> > getModalBasis(std::vector<double> zquad, int nq, int np, int P)
{

    if(nq != zquad.size())
    {
        std::cout << "error: nq != zquad.size() " << std::endl;
    }
    int numModes = P + 1;


    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(np,0.0);
        if(n == 0)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = (1 - zquad[k])/2;
            }
        }
        else if(n == P)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = (1 + zquad[k])/2;
            }
        }
        else
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, n-1, 1.0, 1.0);

            for(int k=0;k<np;k++)
            {
                phi1[k] = ((1-zquad[k])/2)*((1+zquad[k])/2)*phi1[k];
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}






std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

   
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(np,0.0);

        if(n == 0)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = (1 - zquad[k])/2;
            }
        }
        else if(n == P)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = (1 + zquad[k])/2;
            }
        }
        else
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, n-1, 1.0, 1.0);

            for(int k=0;k<np;k++)
            {
                phi1[k] = ((1-zquad[k])/2)*((1+zquad[k])/2)*phi1[k];
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}






std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

   
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(np,0.0);

        if(n == 0)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = 1.0;
            }
        }
        else if(n == 1)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = zquad[k];
            }
        }
        else
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);

            for(int k=0;k<np;k++)
            {
                phi1[k] = phi1[k];
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}



std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(zquad_eval.size(),0.0);

        if(n == 0)
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, 1, 0.0, 0.0);
            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = 0.0;
            }
        }
        else
        {
            std::vector<double> phi2(zquad_eval.size(),0.0);

            jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);
            jacobfd(np, zquad.data(), phi2.data(), NULL, n-1, 0.0, 0.0);

            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = 0.5*(phi1[k] + phi2[k]);
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}


std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

   
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(zquad_eval.size(),0.0);

        if(n == 0)
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, 1, 0.0, 0.0);
            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = 0.0;
            }
        }
        else
        {
            std::vector<double> phi2(zquad_eval.size(),0.0);

            jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);
            jacobfd(np, zquad.data(), phi2.data(), NULL, n-1, 0.0, 0.0);

            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = pow(-1, n)*0.5*(phi1[k] - phi2[k]);
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}









std::vector<std::vector<double> > getNodalBasis(std::vector<double> zquad, int nq, int np, int P)
{

    if(nq != zquad.size())
    {
        std::cout << "error: nq != zquad.size() " << std::endl;
    }
    int numModes = P + 1;


    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {
        std::vector<double> phi1(nq);
        for (int q = 0; q < nq; ++q)
        {
            phi1[q] = hglj(n, zquad[q], zquad.data(), numModes, 0.0, 0.0);
        }
        basis.push_back(phi1);
    }
    return basis;
}



std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P)
{

    if(nq != zquad.size())
    {
        std::cout << "error: nq != zquad.size() " << std::endl;
    }
    int numModes = P + 1;


    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {
        std::vector<double> phi1(zquad_eval.size());
        for (int q = 0; q < zquad_eval.size(); ++q)
        {
            phi1[q] = hglj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
        }
        basis.push_back(phi1);
    }
    return basis;
}











// This member function determines the traces.
void *elbound(int Nel, double *bound,double Ldom,double Rdom)
{
  double Ledom = Rdom-Ldom;
  // assuming equally distributed elements.
  double Elsize = Ledom/Nel;
  double acul = 0.0;
  for(int i=0;i<Nel+1;i++)
  {
    bound[i] = acul;
    acul += Elsize;
    //std::cout << bound[i] << std::endl;
  }
  return 0;
}



int main(int argc, char* argv[])
{

    int     P   = atoi(argv[1]);
    int     Nel = atoi(argv[2]);
    double  dt  = atof(argv[3]);
    int     nt  = atoi(argv[4]);
    int     modal = atoi(argv[5]);
    int Q = P;

    int np = P + 1;
    int nq = Q + 1;
    // int nt = 1500;
    // double dt = 0.001;
    double** D          = dmatrix(np);
    double** Dt         = dmatrix(np);
    std::vector<double> z(np,0.0);
    std::vector<double> w(np,0.0);
    zwgll(z.data(), w.data(), np);
    std::vector<double> zq(nq,0.0);
    std::vector<double> wq(nq,0.0);
    zwgll(zq.data(), wq.data(), nq);
    Dgll(D, Dt, z.data(), np);

    //============================================
    std::vector<double> zqrm(np,0.0);
    std::vector<double> wqrm(np,0.0);
    zwgrjm(zqrm.data(), wqrm.data(), nq, 0.0, 0.0);
    double** Drm          = dmatrix(np);
    double** Drmt         = dmatrix(np);
    Dgrlm(Drm, Drmt, zqrm.data(), np);
    std::vector<double> zqrp(np,0.0);
    std::vector<double> wqrp(np,0.0);
    zwgrjp(zqrp.data(), wqrp.data(), nq, 0.0, 0.0);
    double** Drp          = dmatrix(np);
    double** Drpt         = dmatrix(np);
    Dgrlm(Drp, Drpt, zqrp.data(), np);

    std::vector<double> zgauss(np,0.0);
    std::vector<double> wgauss(np,0.0);
    zwgj(zgauss.data(), wgauss.data(), nq, 0.0, 0.0);
    //=============================================
    // std::vector<std::vector<double> > basis = getLagrangeBasis(zq,nq,z,np,P);
    // double J = 1.0;
    // std::vector<double> M = GetElementMassMatrix_Lagrange(P, z, np, zq, wq, nq, J);

    // for(int i=0;i<nq;i++)
    // {
    //     std::cout << "w[" << i << "]=" << wq[i] << std::endl;
    // }

    // for(int i=0;i<(P+1);i++)
    // {
    //     for(int j=0;j<(P+1);j++)
    //     {
    //         std::cout << M[i*(P+1)+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }



    
    // std::vector<double> coeffs = ForwardTransformLagrange(P, zq, wq, nq, J, z, np);
    
    // for(int i=0;i<P+1;i++)
    // {
    //     std::cout << "c["<<i<<"]=" << coeffs[i] << std::endl;
    // }

    // std::vector<double> quad = BackwardTransformLagrange(P, zq, wq, nq, J, coeffs, z, np);

    // for(int i=0;i<Q+1;i++)
    // {
    //     std::cout << "quad["<<i<<"]=" << quad[i] << std::endl;
    // }

    // int Pf = P - 1;
    // std::vector<double> coeffs_filtered = FilterNodalCoeffs(zq, wq, z, np, nq, coeffs, P, Pf, J);




    //=====================================================================

    double* bound           = dvector(Nel+1);
    elbound(Nel, bound, 0.0, 1.0);
    int** map  = imatrix(Nel, P+1);
    mapping(Nel, P, map);

    double* X_DG_i              = dvector(Nel*(np));
    double* U_DG_i              = dvector(Nel*(np));
    double* F_DG_i              = dvector(Nel*(np));
    double* X_DG                = dvector(Nel*(np));
    double* U_DG                = dvector(Nel*(np));
    double* F_DG                = dvector(Nel*(np));
    double* Jac                 = dvector(Nel);
    double* x                   = dvector(np);
    // Construct the global (initial) state vector u_DG;
    for(int eln=0;eln<Nel;eln++)
    {
        // Determine the coordinates in each element x.
        chi(np, eln, x, z.data(), Jac, bound);
        // Places the element coordinates x into the right place in
        // the global coordinate vector.
        for(int i=0;i<np;i++)
        {
            // construct global coordinates for each quadrature point.
            X_DG_i    [i + eln*np]    = x[i];
            // construct initial solution at each quadrature point.
            U_DG_i    [i + eln*np]    = 2.0+sin(2.0*M_PI*x[i]);
            // Evaluate the flux at each quadrature point.
            F_DG_i    [i + eln*np]    = 0.5*U_DG[i + eln*np]*U_DG[i + eln*np];

            X_DG [i + eln*np] = X_DG_i    [i + eln*np];
            U_DG [i + eln*np] = U_DG_i    [i + eln*np];
            F_DG [i + eln*np] = F_DG_i    [i + eln*np];
        }
    }

    ofstream sol;
    sol.open("dgdata.in");
    for(int i = 0;i < (Nel*np);i++)
    {
        sol << X_DG_i[i] << " " << U_DG_i[i] << endl;
    }
    sol.close();

    double* U_DG_new        = dvector(Nel*(np));
    double* R_DG0           = dvector(Nel*(np));
    double* R_DG1           = dvector(Nel*(np));
    double* R_DG2           = dvector(Nel*(np));
    double* R_DG3           = dvector(Nel*(np));
    double* k1              = dvector(Nel*(np));
    double* k2              = dvector(Nel*(np));
    double* k3              = dvector(Nel*(np));
    double* k4              = dvector(Nel*(np));
    double* k1input         = dvector(Nel*(np));
    double* k2input         = dvector(Nel*(np));
    double* k3input         = dvector(Nel*(np));
    double* k4input         = dvector(Nel*(np));
    int RKstages            = 4;
    double* a               = dvector(RKstages);
    a[0]=1.0/6.0;
    a[1]=1.0/3.0;
    a[2]=1.0/3.0;
    a[3]=1.0/6.0;
    double *bc=dvector(2);
    bc[0]=0.0;
    bc[1]=0.0;
    double time = 0.0;
    // timeScheme = 0 -> Forward Euler
    // timeScheme = 1 -> Runge Kutta 4
    int timeScheme = 1; 


    // ================================== TEST MODAL BASIS ==============================================
    std::vector<std::vector<double> > basis_m;

    std::vector<double> X_DG_test(np,0.0);
    std::vector<double> U_DG_test(np,0.0);



    if (modal == 0)
    {
        std::cout << "Running nodal " << std::endl;

        ofstream solout;
        solout.open("nodal_basis.out");

        std::vector<double> zplot(10*np,0.0);
        std::vector<double> wplot(10*np,0.0);
        zwgll(zplot.data(), wplot.data(), 10*np);
    
        std::vector<std::vector<double> > basis_plot = getNodalBasisEval(zplot, z, nq, np, P);
        
        for(int i=0;i<basis_plot.size();i++)
        {
            for(int j=0;j<basis_plot[i].size();j++)
            {
                 solout << zplot[j] << " " << basis_plot[i][j] << endl;
            }
        }

        solout.close();

        basis_m = getNodalBasis(z, nq, np, P);

        chi(np, 0, x, z.data(), Jac, bound);

        for(int i=0;i<np;i++)
        {
            // construct global coordinates for each quadrature point.
            X_DG_test [i]    = x[i];
            // construct initial solution at each quadrature point.
            U_DG_test [i]    = 2.0+sin(2.0*M_PI*x[i]);
        }

        std::vector<double> coeff_e = ForwardTransform(P, np, basis_m, w, nq, Jac[0], U_DG_test);
       
        std::vector<double> quad_e  = BackwardTransform(P, np,  basis_m,  coeff_e);

        double L2norm = 0.0;
        for(int i=0;i<quad_e.size();i++)
        {
            L2norm = L2norm+(quad_e[i]-U_DG_test[i])*(quad_e[i]-U_DG_test[i]);
        }
        if(abs(L2norm)<1.0e-12)
        {
            std::cout << "Nodal basis Fwd and Bwd test PASSED :: L2norm = " << L2norm << std::endl; 
        }
        else
        {
            std::cout << "Nodal basis Fwd and Bwd test FAILED :: L2norm =" << L2norm << std::endl; 
        }

    }
    
    if (modal == 1)
    {
        std::cout << "Running modal " << std::endl;

        ofstream solout;
        solout.open("modal_basis.out");

        std::vector<double> zplot(10*np,0.0);
        std::vector<double> wplot(10*np,0.0);
        zwgll(zplot.data(), wplot.data(), 10*np);
    
        // std::vector<std::vector<double> > basis_plot = getModalBasisEval(zplot, zplot, nq, zplot.size(), P);
       
        //std::vector<std::vector<double> > basis_plot = getLegendreBasisEval(zplot, zplot, nq, zplot.size(), P);

        std::vector<std::vector<double> > basis_plot = getLegendreBasisEval(zplot, zplot, nq, zplot.size(), P);

        for(int i=0;i<basis_plot.size();i++)
        {
            for(int j=0;j<basis_plot[i].size();j++)
            {
                 solout << zplot[j] << " " << basis_plot[i][j] << endl;
            }
        }

        solout.close();

        basis_m = getLegendreBasisEval(z, z, nq, np, P);

        chi(np, 0, x, z.data(), Jac, bound);

        std::vector<double> MassMatElem = GetElementMassMatrix(P,basis_m,w,Jac[0]);

        std::cout << "M=[";
        for(int i=0;i<P+1;i++)
        {
            std::cout << "[";
            for(int j=0;j<P+1;j++)
            {
                if(j<P)
                {
                    std::cout << MassMatElem[i*(P+1)+j] << ", ";
                }
                else
                {
                    std::cout << MassMatElem[i*(P+1)+j];
                }
                
            }
            if(i<P)
            {
                std::cout <<"],"<< std::endl;
            }
            else
            {
                std::cout <<"]]"<< std::endl;
            }
            
        }


        std::cout << "Running Gauss-Radau-Legendre Minus " << std::endl;

        ofstream solout2;
        solout2.open("radaum_basis.out");

        std::vector<double> zradaum(np,0.0);
        std::vector<double> wradaum(np,0.0);
        zwgrjm(zradaum.data(), wradaum.data(), np, 0.0, 0.0);

        std::vector<double> zradaumplot(10*np,0.0);
        std::vector<double> wradaumplot(10*np,0.0);
        zwgrjm(zradaumplot.data(), wradaumplot.data(), 10*np, 0.0, 0.0);
        std::vector<std::vector<double> > basisradaum_plot = getRadauMinusBasisEval(zradaumplot, zradaumplot, nq, zradaumplot.size(), P);

        //std::vector<std::vector<double> > basisradau_plot = getRadauMinusBasisEvalV2(zradaumplot, zradaum, nq, 10*np, P);
        
        for(int i=0;i<basisradaum_plot.size();i++)
        {   
            std::cout << "basisradaum_plot[i].size() " << basisradaum_plot[i].size() << "  " << zradaumplot.size() << std::endl;
            for(int j=0;j<basisradaum_plot[i].size();j++)
            {
                 solout2 << zradaumplot[j] << " " << basisradaum_plot[i][j] << endl;
            }
        }

        solout2.close();


        ofstream solout3;
        solout3.open("radaup_basis.out");

        std::vector<double> zradaup(np,0.0);
        std::vector<double> wradaup(np,0.0);
        zwgrjp(zradaup.data(), wradaup.data(), np, 0.0, 0.0);

        std::vector<double> zradaupplot(10*np,0.0);
        std::vector<double> wradaupplot(10*np,0.0);
        zwgrjp(zradaupplot.data(), wradaupplot.data(), 10*np, 0.0, 0.0);
        std::vector<std::vector<double> > basisradaup_plot = getRadauPlusBasisEval(zplot, zplot, nq, zplot.size(), P);

        //std::vector<std::vector<double> > basisradau_plot = getRadauMinusBasisEvalV2(zradaumplot, zradaum, nq, 10*np, P);
        
        for(int i=0;i<basisradaup_plot.size();i++)
        {   
            std::cout << "basisradaup_plot[i].size() " << basisradaup_plot[i].size() << "  " << zradaumplot.size() << std::endl;
            for(int j=0;j<basisradaup_plot[i].size();j++)
            {
                 solout3 << zradaumplot[j] << " " << basisradaup_plot[i][j] << endl;
            }
        }

        solout3.close();
        


        for(int i=0;i<np;i++)
        {
            // construct global coordinates for each quadrature point.
            X_DG_test [i]    = x[i];
            // construct initial solution at each quadrature point.
            U_DG_test [i]    = 2.0+sin(2.0*M_PI*x[i]);
        }

        std::vector<double> coeff_e = ForwardTransform(P, np, basis_m, w, nq, Jac[0], U_DG_test);
        
        std::vector<double> quad_e  = BackwardTransform(P, np,  basis_m,  coeff_e);
        double L2norm = 0.0;
        for(int i=0;i<quad_e.size();i++)
        {
            L2norm = L2norm+(quad_e[i]-U_DG_test[i])*(quad_e[i]-U_DG_test[i]);
        }

        if(abs(L2norm)<1.0e-12)
        {
            std::cout << "Modal basis Fwd and Bwd test PASSED :: L2norm = " << L2norm << std::endl; 
        }
        else
        {
            std::cout << "Modal basis Fwd and Bwd test FAILED :: L2norm =" << L2norm << std::endl; 
        }
    }



    for(int t = 0; t < nt; t++)
    {
        
        //================================================================================
        //Forward Euler time integration
        if(timeScheme==0)
        {
            CalculateRHS(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, U_DG, R_DG0, dt, basis_m);
            for(int i=0;i<(Nel*np);i++)
            {
                k1[i] = U_DG[i]+dt*R_DG0[i];
                U_DG[i] = k1[i];
            }
        }
        
        //================================================================================
        //Runge-Kutta 4 time integration
        // //Calculate Stage 1;
        if(timeScheme==1)
        {
            CalculateRHS(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, U_DG, R_DG0,dt, basis_m);
            for(int i=0;i<(Nel*np);i++)
            {
                k1[i] = dt*R_DG0[i];
                k1input[i] = U_DG[i]+dt*R_DG0[i];
            }
            //Calculate Stage 2;
            CalculateRHS(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, k1input, R_DG1,dt, basis_m);
            for(int i=0;i<(Nel*np);i++)
            {
                k2[i] = dt*R_DG1[i];
                k2input[i] = U_DG[i]+dt*R_DG1[i]*0.5;
            }
            //Calculate Stage 3;
            CalculateRHS(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, k2input, R_DG2,dt, basis_m);
            for(int i=0;i<(Nel*np);i++)
            {
                k3[i] = dt*R_DG2[i];
                k3input[i] = U_DG[i]+dt*R_DG2[i]*0.5;
            }
            //Calculate Stage 4;
            CalculateRHS(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, k3input, R_DG3,dt, basis_m);
            for(int i=0;i<(Nel*np);i++)
            {
                k4[i] = dt*R_DG3[i];
                U_DG[i] = U_DG[i]+1.0/6.0*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
                if(std::isnan(U_DG[i]))
                {
                    std::cout << "NaN " <<std::endl;
                    t = nt;
                }
            }


        }
        
    
    
        bc[0]=U_DG[Nel*(np)-1];
        bc[1]=U_DG[0];
        std::cout << "time = " << time << std::endl;
        time = time + dt;       
    }
    ofstream solout;
    solout.open("dgdata.out");
    for(int i = 0;i < (Nel*np);i++)
    {
        solout << X_DG[i] << " " << U_DG[i] << endl;
    }
    solout.close();
    


    
    





    /**/

    




    return 0;
}

/*void CalculateNumericalFlux(np, Nel, trace, U, X_t, **Ut)
{
    for(int eln=0;eln<Nel*(np);eln++)
    {
        U
    }
}

void AddTraceIntegral(np, Nel, trace, U, X_t, **Ut)
{
    for(int eln=0;eln<Nel*(np);eln++)
    {
        U
    }
}*/

// This member function calculates the inner product of the flux with the derivative of the basis functions.




// void CalculateRHS(int np, int Nel, int P, double *z, double *w, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG)

void CalculateRHS(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;
    
    int Mdim                        = (P+1)*Nel;
    double *F_DG                    = dvector(Nel*np);
    // double *quad_e                  = dvector(np);
    std::vector<double> quad_e(np);
    ipiv                            = ivector(Mdim);
    // double *coeff_e                 = dvector(P+1);
    std::vector<double> coeff_e(P+1);
    double *tmp                     = dvector(Mdim);
    double *Fcoeff                  = dvector(Mdim);
    double *Ucoeff                  = dvector(Mdim);
    double **MassMatGlobal          = dmatrix(Mdim);
    double **StiffnessMatGlobal     = dmatrix(Mdim);
    double *numfluxcoeffL  = dvector(Mdim);
    double *numfluxcoeffR  = dvector(Mdim);
    double *numfluxcoeff   = dvector(Mdim);
    double *du      = dvector(2);
    double* phi1 = dvector(np);

    
    
    for(int eln=0;eln<Nel;eln++)
    {
        for(int i=0;i<np;i++)
        {
            // Evaluate the flux at each quadrature point.
            F_DG    [i + eln*np] = 0.5*U_DG[i + eln*np]*U_DG[i + eln*np];
        }
    }
    // Transform fluxes forward into coefficient space.
    //==========================================================
    std::vector<double> dx(Nel);
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        double xstart = X_DG[eln*np];
        double xend   = X_DG[eln*np+np-1];

        dx[eln] = xend-xstart; 
        //std::cout << eln << " " << dx[eln] << std::endl;
        for(int i = 0;i < np;i++)
        {
            quad_e[i] = F_DG[i+eln*np];
        }
        //ForwardTransform(np, P, z, w, Jac[eln], quad_e, coeff_e);
        //std::vector<double> coeff_e = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e, np);

        std::vector<double> coeff_e = ForwardTransform(P, np, basis, wquad, nq, J, quad_e);
        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff[i+eln*(P+1)] = coeff_e[i];
        }
    }
    //==========================================================
    // Calculate the numerical flux;
    double *quads = dvector(np);
    
    double *n        = dvector(Nel*2);
    std::vector<double> UtL(Nel*2);
    std::vector<double> UtR(Nel*2);
    double *numfluxL = dvector(Nel*2);
    double *numfluxR = dvector(Nel*2);
    double *numf = dvector(Nel*2);
    // GetFwdBwd(Nel, np, bc, U_DG, UtL, UtR);
    GetAllFwdBwd(Nel, np, bc, U_DG, UtL, UtR);
    
    negatednormals(Nel, n);
    int cnt = 0;
    
    double *alpha     = dvector(Nel+1);
    double *alphaL    = dvector(Nel+1);
    double *alphaR    = dvector(Nel+1);
    double *FtLn      = dvector(Nel+1);
    double *FtRn      = dvector(Nel+1);
    double *FLUX      = dvector(Nel+1);
    std::vector<double> DeltaF_l(Nel+1,0.0);
    std::vector<double> DeltaF_r(Nel+1,0.0);
    cnt = 0;
    int nL = -1;int nR=1;
    for(int i = 0;i < (Nel+1);i++)
    {
        FtLn[i]      =  nL*(UtL[i]*UtL[i])*0.5;
        FtRn[i]      =  nR*(UtR[i]*UtR[i])*0.5;

        alphaL[i]   = (UtL[i]);
        alphaR[i]   = (UtR[i]);

        FLUX[i]     = 0.5*(FtLn[i]+FtRn[i])-max(fabs(alphaL[i]),fabs(alphaR[i]))*(UtR[i]-UtL[i]);

        DeltaF_l[i] = FLUX[i]-FtLn[i];
        DeltaF_r[i] = FLUX[i]-FtRn[i];

        //std::cout << DeltaF_l[i] << " " << DeltaF_r[i] << " " <<  FLUX[i] << " " << FtLn[i] << " " << FtRn[i] << std::endl;
    }

    double *numcoeff = dvector(Mdim);
    cnt = 0;
    
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff[i] = 0.0;
    }
    
    std::map<int,std::vector<double> > element2flux;
    std::vector<double> flux_res(Nel);

    for(int i = 0;i < Nel;i++)
    {
        std::vector<double> F_Corr_L(np);
        std::vector<double> F_Corr_R(np);
        for(int j=0;j<np;j++)
        {
            // construct global coordinates for each quadrature point.
            F_Corr_L[j]    = 0.0;
            F_Corr_R[j]    = 0.0;
            if(j==0)
            {
                F_Corr_L[j] =  FLUX[i];
            }
            if(j==(np-1))
            {
                F_Corr_R[j] = -FLUX[i+1];
            }
        }
        
        std::vector<double> coeff_e_L = ForwardTransform(P, np, basis, wquad, nq, Jac[0], F_Corr_L);
        std::vector<double> coeff_e_R = ForwardTransform(P, np, basis, wquad, nq, Jac[0], F_Corr_R);

        if(i == 0)
        {
            numcoeff[0]                 =   FLUX[0];
            numcoeff[P]                 =  -FLUX[1];
        }
        else if(i == Nel-1)
        {
            numcoeff[(Nel-1)*(P+1)]     =   FLUX[i];
            numcoeff[Nel*(P+1)-1    ]   =  -FLUX[i+1];
        }
        else
        {
            numcoeff[i*(P+1)]           =   FLUX[i];
            numcoeff[i*(P+1)+P]         =  -FLUX[i+1];
        }
        // for(int j=0;j<np;j++)
        // {
        //     std::cout << "FLUX " << F_Corr_L[j] << " " << F_Corr_R[j] << std::endl;
        // }
        // for(int j=0;j<coeff_e_L.size();j++)
        // {
        //     std::cout << "coeff " << coeff_e_L[i] << " " << coeff_e_R[i] << " " << numcoeff[i*(P+1)+j] << std::endl;
        // }
    }

    GetGlobalStiffnessMatrixNew(Nel, P, wquad, D, Jac, map, Mdim, basis, StiffnessMatGlobal);

    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,Fcoeff,&ONE_INT,&ZERO_DOUBLE,tmp,&ONE_INT);
    
    for(int i = 0;i < Mdim; i++)
    {
        Ucoeff[i] = -tmp[i]+numcoeff[i];

        // std::cout << "numcoeff["<<i<<"]="<<numcoeff[i]<<std::endl;
    }
    std::cout << std::endl;
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, basis, MassMatGlobal);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal[0], &Mdim, ipiv, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff, &Mdim, &INFO);
    
    // Transform back onto quadrature points.
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        for(int i = 0;i<(P+1);i++)
        {
            coeff_e[i] = Ucoeff[i+eln*(P+1)];
        }
        for(int i=0;i<np;i++)
        {
            quad_e[i]=0.0;
        }

        std::vector<double> quad_e = BackwardTransform(P,  np,  basis,  coeff_e);
        int Pf = P - 1;

        // std::vector<double> coeffs_filtered = FilterNodalCoeffs(zquad, wquad, z, np, nq, coeff_e, P, Pf, J);
        // std::vector<double> quad_e_filter = BackwardTransformLagrange(P, zquad, wquad, nq, J, coeffs_filtered, z, np);
        // for(int i = 0;i < np;i++)
        // {
        //     quad_e[i] = quad_e_filter[i];
        // }

        // if(eln == 7 || eln == 8)
        // {
        //     int Pf = P - 1;

        //     std::vector<double> coeffs_filtered = FilterNodalCoeffs(zquad, wquad, z, np, nq, coeff_e, P, Pf, J);

        //     std::vector<double> quad_e_filter = BackwardTransformLagrange(P, zquad, wquad, nq, J, coeffs_filtered, z, np);
        //     for(int i = 0;i < np;i++)
        //     {
        //         quad_e[i] = quad_e_filter[i];
        //     }
        // }

        for(int i = 0;i < np;i++)
        {
            R_DG[i+np*eln] = quad_e[i];
        }
    }
}









void *negatednormals(int Nel, double *n)
{
    
    for(int i = 0;i < (2.0*Nel);i++)
    {
        n[i] = 1.0;
        
        if(i>1 && (i % 2 == 0))
        {
            n[i] = -1.0;
            
        }
    }
    return 0;
}


void GetAllFwdBwd(int Nel, int np, double *bc, double *quad, std::vector<double> &UtL, std::vector<double> &UtR)
{
    
    for(int i=0;i<Nel+1;i++)
    {
        if(i == 0)
        {
            //UtL[0] = bc[0];
            UtL[0] = quad[0];
            UtR[0] = quad[0];

        }
        else if(i == Nel)
        {
            UtL[i] = quad[Nel*np-1]; 
            UtR[i]=quad[Nel*np-1];
            
        }
        else
        {
            UtL[i] = quad[i*np-1];
            //UtR[i]=bc[1];
            UtR[i] = quad[i*np];
            //std::cout << "fluxie " << quad[i*np-1] << " " << quad[i*np] <<std::endl;
        }
    }
}



void GetGlobalStiffnessMatrixNew(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal)
{
    double **StiffnessElem = dmatrix(P+1);
    // Construct global Mass matrix.
    for(int i=0;i<Mdim;i++)
    {
        for(int j=0;j<Mdim;j++){
            StiffnessGlobal[i][j] = 0;
        }
    }
    
    for(int eln=0;eln<Nel;eln++)
    {
        // Determine elemental mass matrix;
        GetElementStiffnessMatrixNew(P, wquad, D, Jac[eln], basis, StiffnessElem);
        //std::cout << std::endl;
        for(int a=0;a<P+1;a++)
        {
            for(int b=0;b<P+1;b++)
            {
                // Assemble the global mass matrix;
                StiffnessGlobal[map[eln][a]][map[eln][b]] = StiffnessGlobal[map[eln][a]][map[eln][b]] + StiffnessElem[a][b];
            }
        }   
    }

    // std::cout << std::endl;
    // for(int i=0;i<Mdim;i++)
    // {
    //     for(int j=0;j<Mdim;j++)
    //     {
    //         std::cout << StiffnessGlobal[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

}



void GetGlobalStiffnessMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, int Mdim, double **StiffnessGlobal)
{
    double **StiffnessElem = dmatrix(P+1);
    // Construct global Mass matrix.
    for(int i=0;i<Mdim;i++)
    {
        for(int j=0;j<Mdim;j++){
            StiffnessGlobal[i][j] = 0;
        }
    }
    std::vector<std::vector<double> > basis = getNodalBasis(zquad, nq, np, P);
    
    for(int eln=0;eln<Nel;eln++)
    {
        // Determine elemental mass matrix;
        GetElementStiffnessMatrixNew(P, wquad, D, Jac[eln], basis, StiffnessElem);
        //std::cout << std::endl;
        for(int a=0;a<P+1;a++)
        {
            for(int b=0;b<P+1;b++)
            {
                // Assemble the global mass matrix;
                StiffnessGlobal[map[eln][a]][map[eln][b]] = StiffnessGlobal[map[eln][a]][map[eln][b]] + StiffnessElem[a][b];
            }
        }   
    }

    // std::cout << std::endl;
    // for(int i=0;i<Mdim;i++)
    // {
    //     for(int j=0;j<Mdim;j++)
    //     {
    //         std::cout << StiffnessGlobal[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

}









void GetElementStiffnessMatrixNew(int P, std::vector<double> wquad, double **D, double J, std::vector<std::vector<double> > basis, double **StiffMatElem)
{
    for(int i=0;i<P+1;i++)
    {
        for(int j=0;j<P+1;j++){
            StiffMatElem[i][j] = 0;
        }
    }
    
    // double *phi1  = dvector(np);
    // double *dphi1 = dvector(np);
    // double *phi2  = dvector(np);
    int np = basis[0].size();
    double *dphi2 = dvector(np);

    //std::cout << "Stiffness Matrix " << std::endl;
    for(int i=0;i<P+1;i++)
    {
        //std::vector<double> phi1 = getLagrangeBasisFunction(i,zquad,nq,z,np,P);
        std::vector<double> phi1 = basis[i];
        for(int j=0;j<P+1;j++)
        {
            // std::vector<double> phi2 = getLagrangeBasisFunction(j,zquad,nq,z,np,P);
            std::vector<double> phi2 = basis[j];
            //lagrange_basis(np, P, j, z, phi2);
            diff( np, D, phi2.data(), dphi2, J);
            
            StiffMatElem[i][j] = J*integr(np, wquad.data(), phi1.data(), dphi2);
        }
    }
}








void GetFwdBwd(int eln, int Nel, int np, double *bc, double *quad, double *Fwd, double *Bwd)
{
    
    if(eln == 0)
    {
        // Bwd and Fwd states at the left side of the element.
        Bwd[0]     = bc[0];
        Fwd[0]     = quad[0];
        // Bwd and Fwd states at the right side of the element.
        Bwd[1]     = quad[np];
        Fwd[1]     = quad[np-1];
    }
    else if(eln == Nel-1)
    {
        // Bwd and Fwd states at the left side of the element.
        Bwd[0]     = quad[(eln-1)*np-1];
        Fwd[0]     = quad[eln*np];
        // Bwd and Fwd states at the right side of the element.
        Bwd[1]     = bc[1];
        Fwd[1]     = quad[eln*np-1];
    }
    else
    {
        //cout << (eln)*np-1 << " " << (eln)*np << endl;
        // Bwd and Fwd states at the left side of the element.
        Bwd[0]     = quad[(eln)*np-1];
        Fwd[0]     = quad[(eln)*np];
        // Bwd and Fwd states at the right side of the element.
        Bwd[1]     = quad[(eln)*np+np+1];
        Fwd[1]     = quad[(eln)*np+np];
    }
}

void TraceMap(int np, int Nel, int **trace)
{
    int cnt = 0;
    for(int eln=0;eln<Nel;eln++)
    {
        trace[eln][cnt]   = eln*np;
        trace[eln][cnt+1] = eln*np+np-1;
    }
    cnt = cnt+2;
}




// This member function calculates the inner product of the flux with the derivative of the basis functions.
void InnerProductWRTDerivBasis(int np, int P, double J, double *w, double *z, double **D, double *quad, double *coeff)
{
    int ONE_INT=1;
    double ONE_DOUBLE=1.0;
    double ZERO_DOUBLE=0.0;
    unsigned char TR = 'T';
    
    int ncoeff = P;
    double *tmp   = dvector(np);
    double *phi1  = dvector(np);
    double *dphi1 = dvector(np);
    
    for(int n=0;n<P+1;n++)
    {
        
        //basis(np, P, n, z, phi1);
        lagrange_basis(np, P, n, z, z, phi1);
        diff(np, D, phi1, dphi1, J);
        coeff[n] = J*integr(np, w, dphi1, quad);
    }
}




void GetGlobalMassMatrixNew(int Nel, int P, std::vector<double> wquad, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **MassMatGlobal)
{
    // double **MassMatElem = dmatrix(P+1);
    // Construct global Mass matrix.

    for(int i=0;i<Mdim;i++)
    {
        for(int j=0;j<Mdim;j++){
            MassMatGlobal[i][j] = 0;
        }
    }
    
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];
        // Determine elemental mass matrix;
        // GetElementMassMatrix_Lagrange(np, P, z, w, Jac[eln], MassMatElem);
        //std::vector<double> MassMatElem = GetElementMassMatrix_Lagrange(P,zquad,np,zquad,wquad,nq,J);
        std::vector<double> MassMatElem = GetElementMassMatrix(P, basis, wquad, J);
        for(int a=0;a<P+1;a++)
        {
            for(int b=0;b<P+1;b++)
            {
                // Assemble the global mass matrix;
                MassMatGlobal[map[eln][a]][map[eln][b]] = MassMatGlobal[map[eln][a]][map[eln][b]] + MassMatElem[a*(P+1)+b];//MassMatElem[a][b];
            }
        }
    }
}







// This member function allocates memory and sets up for a double vector.
double *dvector(int n)
{
  double *v;

  v = (double *)malloc(n*sizeof(double));
  return v;
}

// This member function allocates memory and sets up for a integer vector.
int *ivector(int n)
{
  int *v;
  
  v = (int *)malloc(n*sizeof(int));
  return v;
}

// This member function allocates memory and sets up for a integer matrix.
int **imatrix(int nrow, int ncol)
{
    register int **A;
    
    A = (int **)malloc(nrow*sizeof(int *));
    A[0] = (int *)malloc(nrow*ncol*sizeof(int));
    
    for(int i=1;i<nrow;i++){
        A[i] = A[i-1]+ncol;
    }
    return A;
}

// This member function allocates memory and sets up for a double matrix.
double **dmatrix(int n)
{
  double **A;
  A = (double **)malloc(n*sizeof(double *));
  A[0] = (double *)malloc(n*n*sizeof(double));

  for(int i=1;i<n;i++){
    A[i] = A[i-1]+n;
  }
  return A;
}

// This member function allocates memory and sets up for a double array.
double **darray(int n,int m)
{
    double **A;
    A = (double **)malloc(n*sizeof(double *));
    A[0] = (double *)malloc(n*m*sizeof(double));
    
    for(int i=1;i<n;i++){
        A[i] = A[i-1]+m;
    }
    return A;
}

int **iarray(int n,int m)
{
    int **A;
    A = (int **)malloc(n*sizeof(int *));
    A[0] = (int *)malloc(n*m*sizeof(int));
    
    for(int i=1;i<n;i++){
        A[i] = A[i-1]+m;
    }
    return A;
}

void getbasis(int np, int P, int ncoeff, double *z, double **basis)
{
    //cout << "i="<< i << endl;
    for(int i=0;i<ncoeff;i++)
    {
        if(i == 0){
            for(int k=0;k<np;k++){
                basis[k][i] = (1 - z[k])/2;
            }
        }else if(i == P-1){
            for(int k=0;k<np;k++){
                basis[k][i] = (1 + z[k])/2;
            }
        }else{
            double *tmp = dvector(np);
            jacobfd(np, z, tmp, NULL, i-1, 1.0, 1.0);
            for(int k=0;k<np;k++){
                basis[k][i] = ((1-z[k])/2)*((1+z[k])/2)*tmp[k];
            }
        }
    }
}



// This member function perform numerical quadrature.
double integr(int np, double *w, double *phi1, double *phi2)
{
  register double sum = 0.;

  for(int i=0;i<np;i++){
    sum = sum + phi1[i]*phi2[i]*w[i]; 
  }
  return sum;
}


// This member function performs numerical differentiation.
void *diff(int np, double **D, double *p, double *pd, double J)
{
    for(int i=0;i<np;i++){
        pd[i] = 0;
        for(int j=0;j<np;j++){
            pd[i] = pd[i] + D[i][j]*p[j];
        }
        pd[i] = pd[i]/J;
        
    }
    return 0;
}

// This member function maps the standard region to the local region.
void *chi(int np, int eln, double *x, double *z, double *Jac, double *bound)
{
  for(int i=0;i<np;i++){
    x[i] = ((1 - z[i])/2)*bound[eln] + ((1 + z[i])/2)*bound[eln+1];
  }
  Jac[eln] =  (-bound[eln]/2) + (bound[eln+1]/2);
  return 0;
}

// This member function determines the mapping.
void *mapping(int Nel, int P, int **map)
{
    for(int e=0;e<Nel;e++)
    {
        for(int p=0;p<P+1;p++)
        {
            map[e][p] = p + (e*(P+1));
        }
    }

    return 0;
}


// This member function determines the ith basis function.
std::vector<double> modal_basis(int np, int P, int i, std::vector<double> z)
{
    std::vector<double> phi(np);
    if(i == 0){
        for(int k=0;k<np;k++){
            phi[k] = (1 - z[k])/2;
        }
    }else if(i == P){
        for(int k=0;k<np;k++){
            phi[k] = (1 + z[k])/2;
        }
    }else{
        jacobfd(np, z.data(), phi.data(), NULL, i-1, 1.0, 1.0);
        for(int k=0;k<np;k++){
            phi[k] = ((1-z[k])/2)*((1+z[k])/2)*phi[k];
        }
    }
    return phi;
}




// This member function evaluates the flux based on the state vector.
void evaluateflux(int np, double *u_DG, double *Flux_DG)
{
    for(int i=0;i<np;i++)
    {
        Flux_DG[i]  = 0.5*u_DG[i]*u_DG[i];
    }
  
}

