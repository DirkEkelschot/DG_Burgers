#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/Polylib.h"
#include "src/basis_functions.h"
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
using namespace std;
using namespace polylib;
std::vector<double> LaxFriedrichsRiemannVec(std::vector<double> Ul, std::vector<double> Ur, double normal);
void GetAllFwdBwdMapVec(int Nel, int np, std::vector<std::vector<double> > quad, std::map<int,std::vector<std::vector<double> > > &Umap, int nvar);
void CalculateRHS_Modal(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt);
void CalculateRHS(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis);
void CalculateRHSWeakFR(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP);
void CalculateRHSStrongFR(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP);
void CalculateRHSStrongFREuler(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, std::vector<std::vector<double> > bc, std::vector<double> X_DG, std::vector<std::vector<double> > U_DG, std::vector<std::vector<double> > &R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP);

void *negatednormals(int Nel, double *n);
int **iarray(int n,int m);
std::vector<std::vector<double> > getNodalBasis(std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getModalBasis(std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);

//std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
void GetFwdBwd(int eln, int Nel, int np, double *bc, double *quad, double *UtL, double *UtR);
void GetAllFwdBwd(int Nel, int np, double *bc, double *quad, std::vector<double> &UtL, std::vector<double> &UtR);
void GetAllFwdBwdMap(int Nel, int np, double *quad, std::map<int,std::vector<double> > &Umap);
void GetAllFwdBwdMapCoeff(int Nel, int P, double *coeff, std::map<int,std::vector<double> > &Umap);
void TraceMap(int np, int Nel, int **trace);
std::vector<double> BackwardTransformLagrange(int P, std::vector<double> zquad, std::vector<double> wquad, int nq, double J, std::vector<double> coeff, std::vector<double> z, int np);
void InnerProductWRTDerivBasis(int np, int P, double J, double *w, double *z, double **D, double *F_DG, double *coeff);
void GetGlobalMassMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double *Jac, int **map, int Mdim, double **MassMatGlobal);
void GetGlobalMassMatrix_Modal(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad,  std::vector<double> z, double *Jac, int **map, int Mdim, double **MassMatGlobal);
void *basis(int np, int P, int i, double *z, double *phi);
void getbasis(int np, int P, int ncoeff, double *z, double **basis);
void getdbasis(int np, int P, double J, double **D, double **basis, double **dbasis);

// void *diff(int np, double **D, double *p, double *pd, double J);
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
void GetGlobalStiffnessMatrixWeakNew(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal);
void GetElementStiffnessMatrix(int np, int nq, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double J, double **StiffMatElem);
void GetElementStiffnessMatrixNew(int P, std::vector<double> wquad, double **D, double J, std::vector<std::vector<double> > basis, double **StiffMatElem);
void GetGlobalStiffnessMatrixNew(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal);
void GetElementStiffnessMatrixWeakNew(int P, std::vector<double> wquad, double **D, double J, std::vector<std::vector<double> > basis, double **StiffMatElem);
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









// std::vector<double> GetElementMassMatrix(int P, 
//                             std::vector<std::vector<double> > basis,
//                             std::vector<double> wquad,
//                             double J)
// {
//     std::vector<double> MassMatElem((P+1)*(P+1),0.0);
//     int np = wquad.size();
//     for(int i=0;i<P+1;i++)
//     {
//         std::vector<double> phi1 = basis[i];
//         for(int j=0;j<P+1;j++)
//         {
//             std::vector<double> phi2 = basis[j];
//             MassMatElem[i*(P+1)+j] = J*integr(np, wquad.data(), phi1.data(), phi2.data());
//         }
//     }
//     return MassMatElem;
// }





















// std::vector<double> ForwardTransform(int P, 
//                                      int np,
//                                      std::vector<std::vector<double> > basis, 
//                                      std::vector<double>wquad, int nq,
//                                      double J, 
//                                      std::vector<double> input_quad)
// {
    
//     std::vector<double> coeff(P+1);
//     int ncoeffs     = P+1;
//     double *Icoeff  = dvector(ncoeffs);
    
//     for(int j=0;j<P+1;j++)
//     {
//         std::vector<double> phi1 = basis[j];

//         Icoeff[j] = J*integr(nq, wquad.data(), phi1.data(), input_quad.data());
//     }

//     std::vector<double> MassMatElem = GetElementMassMatrix(P,basis,wquad,J);

//     int ONE_INT=1;
//     double ONE_DOUBLE=1.0;
//     double ZERO_DOUBLE=0.0;
//     unsigned char TR = 'T';
//     int INFO;
//     int LWORK = ncoeffs*ncoeffs;
//     double *WORK = new double[LWORK];
//     int *ip = ivector(ncoeffs);
//     // Create inverse Mass matrix.
//     dgetrf_(&ncoeffs, &ncoeffs, MassMatElem.data(), &ncoeffs, ip, &INFO);
//     dgetri_(&ncoeffs, MassMatElem.data(), &ncoeffs, ip, WORK, &LWORK, &INFO);
//     // Apply InvMass to Icoeffs hence M^-1 Icoeff = uhat
//     dgemv_(&TR,&ncoeffs,&ncoeffs,&ONE_DOUBLE,MassMatElem.data(),&ncoeffs,Icoeff,&ONE_INT,&ZERO_DOUBLE,coeff.data(),&ONE_INT);
//     return coeff;
// }


















// std::vector<double> BackwardTransform(int P, 
//                                       int np, 
//                                       std::vector<std::vector<double> > basis,  
//                                       std::vector<double> input_coeff)
// {

//     std::vector<double> quad(np,0.0);
//     double sum = 0.0;
//     for(int i = 0;i<P+1;i++)
//     {
//         std::vector<double> phi1 =basis[i];
//         for( int j=0;j<np;j++)
//         {
//             quad[j] = quad[j]+input_coeff[i]*phi1[j];
//         }
//     }

//     return quad;
// }



















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






// std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
// {

   
//     int numModes = P + 1;

//     std::vector<std::vector<double> > basis;
    
//     for(int n=0;n<numModes;n++)
//     {   
//         std::vector<double> phi1(np,0.0);

//         if(n == 0)
//         {
//             for(int k=0;k<np;k++)
//             {
//                 phi1[k] = (1 - zquad[k])/2;
//             }
//         }
//         else if(n == P)
//         {
//             for(int k=0;k<np;k++)
//             {
//                 phi1[k] = (1 + zquad[k])/2;
//             }
//         }
//         else
//         {
//             jacobfd(np, zquad.data(), phi1.data(), NULL, n-1, 1.0, 1.0);

//             for(int k=0;k<np;k++)
//             {
//                 phi1[k] = ((1-zquad[k])/2)*((1+zquad[k])/2)*phi1[k];
//             }
//         }
        
//         basis.push_back(phi1);
//     }
//     return basis;
// }






// std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
// {

   
//     int numModes = P + 1;

//     std::vector<std::vector<double> > basis;
    
//     for(int n=0;n<numModes;n++)
//     {   
//         std::vector<double> phi1(np,0.0);

//         if(n == 0)
//         {
//             for(int k=0;k<np;k++)
//             {
//                 phi1[k] = 1.0;
//             }
//         }
//         else if(n == 1)
//         {
//             for(int k=0;k<np;k++)
//             {
//                 phi1[k] = zquad[k];
//             }
//         }
//         else
//         {
//             jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);

//             for(int k=0;k<np;k++)
//             {
//                 phi1[k] = phi1[k];
//             }
//         }
        
//         basis.push_back(phi1);
//     }
//     return basis;
// }



// std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
// {

//     int numModes = P + 1;

//     std::vector<std::vector<double> > basis;
    
//     for(int n=0;n<numModes;n++)
//     {   
//         std::vector<double> phi1(zquad_eval.size(),0.0);

//         if(n == 0)
//         {
//             jacobfd(np, zquad_eval.data(), phi1.data(), NULL, 1, 0.0, 0.0);
//             for(int k=0;k<zquad_eval.size();k++)
//             {
//                 phi1[k] = 0.0;
//             }
//         }
//         else
//         {
//             std::vector<double> phi2(zquad_eval.size(),0.0);

//             jacobfd(np, zquad_eval.data(), phi1.data(), NULL, n, 0.0, 0.0);
//             jacobfd(np, zquad_eval.data(), phi2.data(), NULL, n-1, 0.0, 0.0);

//             for(int k=0;k<zquad_eval.size();k++)
//             {
//                 phi1[k] = 0.5*(phi1[k] + phi2[k]);
//             }
//         }
        
//         basis.push_back(phi1);
//     }
//     return basis;
// }


// std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
// {

   
//     int numModes = P + 1;

//     std::vector<std::vector<double> > basis;
    
//     for(int n=0;n<numModes;n++)
//     {   
//         std::vector<double> phi1(zquad_eval.size(),0.0);

//         if(n == 0)
//         {
//             jacobfd(np, zquad.data(), phi1.data(), NULL, 1, 0.0, 0.0);
//             for(int k=0;k<zquad_eval.size();k++)
//             {
//                 phi1[k] = 0.0;
//             }
//         }
//         else
//         {
//             std::vector<double> phi2(zquad_eval.size(),0.0);

//             jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);
//             jacobfd(np, zquad.data(), phi2.data(), NULL, n-1, 0.0, 0.0);

//             for(int k=0;k<zquad_eval.size();k++)
//             {
//                 phi1[k] = pow(-1, n)*0.5*(phi1[k] - phi2[k]);
//             }
//         }
        
//         basis.push_back(phi1);
//     }
//     return basis;
// }









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



// std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P)
// {

//     if(nq != zquad.size())
//     {
//         std::cout << "error: nq != zquad.size() " << std::endl;
//     }
//     int numModes = P + 1;


//     std::vector<std::vector<double> > basis;
    
//     for(int n=0;n<numModes;n++)
//     {
//         std::vector<double> phi1(zquad_eval.size());
//         for (int q = 0; q < zquad_eval.size(); ++q)
//         {
//             phi1[q] = hglj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
//         }
//         basis.push_back(phi1);
//     }
//     return basis;
// }











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

    std::vector<double> zradaum(np,0.0);
    std::vector<double> wradaum(np,0.0);
    zwgrjm(zradaum.data(), wradaum.data(), np, 0.0, 0.0);

    std::vector<double> zradaup(np,0.0);
    std::vector<double> wradaup(np,0.0);
    zwgrjp(zradaup.data(), wradaup.data(), np, 0.0, 0.0);


    //=====================================================================

    
    double time = 0.0;
    // timeScheme = 0 -> Forward Euler
    // timeScheme = 1 -> Runge Kutta 4
    int timeScheme = 0; 


    std::vector<double> X_DG_e(Nel*np,0.0);
    std::vector<std::vector<double> > U_DG(3);
    std::vector<double> U_DG_row0(Nel*np,0.0);
    std::vector<double> U_DG_row1(Nel*np,0.0);
    std::vector<double> U_DG_row2(Nel*np,0.0);
    double* bound           = dvector(Nel+1);
    elbound(Nel, bound, 0.0, 1.0);
    int** map  = imatrix(Nel, P+1);
    mapping(Nel, P, map);
    double* Jac                 = dvector(Nel);
    double* x                   = dvector(np);
    double gammaMone = 1.4 - 1.0;

    double pL = 1.0;
    double uL = 0.0;
    double rhoL = 1.0;
    double pR = 1.0;
    double uR = 0.0;
    double rhoR = 0.99;


    for(int eln=0;eln<Nel;eln++)
    {
        // Determine the coordinates in each element x.
        chi(np, eln, x, z.data(), Jac, bound);
        
        
        // Places the element coordinates x into the right place in
        // the global coordinate vector.
        for(int i=0;i<np;i++)
        {
            X_DG_e[eln*np+i] = x[i];

            if(x[i]<0.5)
            {
                double pressure        = pL;
                U_DG_row0[eln*np+i]    = 2.0+sin(2.0*M_PI*x[i]);//rhoL;
                U_DG_row1[eln*np+i]    = 2.0+sin(2.0*M_PI*x[i]);//uL;
                U_DG_row2[eln*np+i]    = 2.0+sin(2.0*M_PI*x[i]);//(pressure/gammaMone);

            }
            else if(x[i]>=0.5)
            {
                double pressure        = pR;
                U_DG_row0[eln*np+i]    = 2.0+sin(2.0*M_PI*x[i]);//rhoR;
                U_DG_row1[eln*np+i]    = 2.0+sin(2.0*M_PI*x[i]);//uR;
                U_DG_row2[eln*np+i]    = 2.0+sin(2.0*M_PI*x[i]);//(pressure/gammaMone);
            }

        }
    }




    U_DG[0] = U_DG_row0;
    U_DG[1] = U_DG_row1;
    U_DG[2] = U_DG_row2;


    ofstream sol_e;
    sol_e.open("dgdataEuler.in");
    for(int i = 0;i < (Nel*np);i++)
    {
        sol_e << X_DG_e[i] << " " << U_DG[0][i] << " " << U_DG[1][i]<< " " << U_DG[2][i] << endl;
    }
    sol_e.close();


    double rho_l = rhoL;
    double u_l   = uL;
    double p_l   = pL;

    double rho_r = rhoR;
    double u_r   = uR;
    double p_r   = pR;
    
    std::vector<std::vector<double> > bc_e(2);
    std::vector<double> bc0(3,0.0);
    bc0[0] = rho_l;
    bc0[1] = rho_l*u_l;
    bc0[2] = p_l/(gammaMone)+0.5*rho_l*(u_l*u_l);
    std::vector<double> bc1(3,0.0);
    bc1[0] = rho_r;
    bc1[1] = rho_r*u_r;
    bc1[2] = p_r/(gammaMone)+0.5*rho_r*(u_r*u_r);

    bc_e[0] = bc0;
    bc_e[1] = bc1;

    std::vector<std::vector<double> > R_DG0(3);


    // ================================== TEST MODAL BASIS ==============================================
    std::vector<std::vector<double> > basis_m;
    std::vector<std::vector<double> > basis_rm;
    std::vector<std::vector<double> > basis_rp;

    std::vector<double> X_DG_test(np,0.0);
    std::vector<double> U_DG_test(np,0.0);


    if (modal == 0)
    {
        run_nodal_test(x, z, w, bound, Jac, np, nq, P);


        basis_m = getNodalBasisEval(z, z, nq, np, P);
    }
    
    if (modal == 1)
    {
        run_modal_test(x, z, w, bound, Jac, np, nq, P);


        basis_m = getModalBasisEval(z, z, nq, np, P);
        //basis_m = getLegendreBasisEval(z, z, nq, np, P);
        
        basis_rm = getRadauMinusBasisEval(zradaum, zradaum, nq, zradaum.size(), P);
        basis_rp = getRadauPlusBasisEval(zradaup, zradaup, nq, zradaup.size(), P);

    }




    for(int t = 0; t < nt; t++)
    {
        
        //================================================================================
        //Forward Euler time integration
        if(timeScheme==0)
        {
            CalculateRHSStrongFREuler(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc_e, X_DG_e, U_DG, R_DG0, dt, basis_m,basis_rm,basis_rp);
            for(int i=0;i<(Nel*np);i++)
            {
                //std::cout << "R_DG0[0][i] " << R_DG0[0][i] << " " << R_DG0[1][i] << " " << R_DG0[2][i] << std::endl; 
                U_DG[0][i]  = U_DG[0][i]+dt*R_DG0[0][i];
                U_DG[1][i]  = U_DG[1][i]+dt*R_DG0[1][i];
                U_DG[2][i]  = U_DG[2][i]+dt*R_DG0[2][i];
            }
        }
        
        
        std::cout << "time = " << time << std::endl;
        time = time + dt;       
    }
    ofstream solout;
    solout.open("dgdataEuler.out");
    for(int i = 0;i < (Nel*np);i++)
    {
            solout << X_DG_e[i] << " " << U_DG[0][i] << " " << U_DG[1][i] << " " << U_DG[2][i] << endl;
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


double LaxFriedrichsRiemann(double Ul, double Ur, double n)
{
    double Fl = Ul*Ul*0.5;
    double Fr = Ur*Ur*0.5;

    double alphaL   = Ul;
    double alphaR   = Ur;

    double Fn = 0.5*(Fl+Fr)*n-max(fabs(alphaL),fabs(alphaR))*(Ur-Ul);

    return Fn;

}



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


    std::map<int,std::vector<double> > Umap;
    GetAllFwdBwdMap(Nel, np, U_DG, Umap);
    std::map<int,std::vector<double> > UmapCoeff;
    GetAllFwdBwdMapCoeff(Nel, P, U_DG, UmapCoeff);
    // for(int i=0;i<UtL.size();i++)
    // {
    //     std::cout << "Ut " << UtL[i] << " " << UtR[i] << std::endl;

    // }

    std::map<int,std::vector<double> >::iterator itm;
    std::map<int,std::vector<double> > Fmap;
    std::vector<double> FLUX_new(Nel+1);
    for(itm=Umap.begin();itm!=Umap.end();itm++)
    {
        double uLFwd = 0.0, uLBwd = 0.0, FLFwd = 0.0;
        double uRFwd = 0.0, uRBwd = 0.0, FRFwd = 0.0;

        double uLFwdCoeff = 0.0, uLBwdCoeff = 0.0, FLFwdCoeff = 0.0, FLBwdCoeff = 0.0;
        double uRFwdCoeff = 0.0, uRBwdCoeff = 0.0, FRFwdCoeff = 0.0, FRBwdCoeff = 0.0;

        int elid = itm->first;

        if(itm->first == 0)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid][0];
            
            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];  

            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid][0];
            
            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid+1][0];   

        }
        else if(itm->first == Nel-1)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid][1];


            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid-1][1];

            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid][1];
        }
        else
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];

            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid-1][1];

            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid+1][0];
        }


        FLFwd = uLFwd*uLFwd*0.5;
        FRFwd = uRFwd*uRFwd*0.5;

        double Fl = LaxFriedrichsRiemann(uLFwd,uLBwd,-1.0);
        double Fr = LaxFriedrichsRiemann(uRFwd,uRBwd,1.0);

        double FlCoeff = LaxFriedrichsRiemann(uLFwdCoeff,uLBwdCoeff,-1.0);
        double FrCoeff = LaxFriedrichsRiemann(uRFwdCoeff,uRBwdCoeff,1.0);
        std::cout << elid << " fluxes " << Fl << " " << Fr << " " << FLFwd << " " << FRFwd << " " << (Fl-FLFwd) << " " << (Fr-FRFwd)<< std::endl;
        Fmap[elid].push_back(Fl-FLFwd);
        Fmap[elid].push_back(Fr-FRFwd);
    }


    negatednormals(Nel, n);
    int cnt = 0;
    
    double *alpha     = dvector(Nel+1);
    double *alphaL    = dvector(Nel+1);
    double *alphaR    = dvector(Nel+1);
    double *FtLn      = dvector(Nel+1);
    double *FtRn      = dvector(Nel+1);
    double *FLUX      = dvector(Nel+1);
    double *FLUX_NoN  = dvector(Nel+1);
    std::vector<double> DeltaF_l(Nel+1,0.0);
    std::vector<double> DeltaF_r(Nel+1,0.0);
    cnt = 0;
    int nL = 1;int nR=-1;
    for(int i = 0;i < (Nel+1);i++)
    {
        FtLn[i]      =  nL*(UtL[i]*UtL[i])*0.5;
        FtRn[i]      =  nR*(UtR[i]*UtR[i])*0.5;

        double fLn   = (UtL[i]*UtL[i])*0.5;
        double fRn   = (UtR[i]*UtR[i])*0.5;

        alphaL[i]    = (UtL[i]);
        alphaR[i]    = (UtR[i]);

        FLUX[i]      = 0.5*(FtLn[i]+FtRn[i])-max(fabs(alphaL[i]),fabs(alphaR[i]))*(UtR[i]-UtL[i]);
        FLUX_NoN[i]  = 0.5*(fLn+fRn)-max(fabs(alphaL[i]),fabs(alphaR[i]))*(UtR[i]-UtL[i]);

        DeltaF_l[i]  = FLUX_NoN[i]-FtLn[i];
        DeltaF_r[i]  = FLUX_NoN[i]+FtRn[i];

        std::cout<<i<<" "<<FLUX[i] <<" "<<FtLn[i]<<" "<< FtRn[i] << " " << FLUX_NoN[i] << " " << DeltaF_l[i] << " " << DeltaF_r[i] << " " << FLUX[i] << std::endl;
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

        double Ul = UtL[i];
        double Ur = UtR[i];
        double Fl = LaxFriedrichsRiemann(Ul,Ur,-1.0);
        double Fr = LaxFriedrichsRiemann(Ul,Ur, 1.0);

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
            std::cout << "Check fluxes i == 0 " << Fmap[i][0] << " " << FLUX[0] << " " << Fmap[i][1] << " " << FLUX[1] << std::endl;
            
            numcoeff[0]                 =   (FLUX[0]);//-Fcoeff[0]);
            numcoeff[P]                 =  -(FLUX[1]);//-Fcoeff[P]);
        }
        else if(i == Nel-1)
        {
            std::cout << "Check fluxes i == Nel-1 " << Fmap[i][0] << " " << FLUX[i] << " " << Fmap[i][1] << " " << FLUX[i+1] << std::endl;

            numcoeff[(Nel-1)*(P+1)]     =     (FLUX[i]);//-Fcoeff[(Nel-1)*(P+1)]);
            numcoeff[Nel*(P+1)-1  ]     =    -(FLUX[i+1]);//-Fcoeff[Nel*(P+1)-1 ]);
        }
        else
        {
            std::cout << "Check fluxes i " << Fmap[i][0] << " " << FLUX[i] << " " << Fmap[i][1] << " " << FLUX[i+1] << std::endl;

            numcoeff[i*(P+1)]           =   (FLUX[i]);//-Fcoeff[i*(P+1)]);
            numcoeff[i*(P+1)+P]         =  -(FLUX[i+1]);//-Fcoeff[i*(P+1)+P]);
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
    //std::cout << std::endl;
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





void CalculateRHSWeakFR(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;
    
    int Mdim                        = (P+1)*Nel;
    double *F_DG                    = dvector(Nel*np);
    // double *quad_e                  = dvector(np);
    std::vector<double> quad_e(np);
    std::vector<double> quad_u(np);
    std::vector<double> quad_eo0(np);
    std::vector<double> quad_eo1(np);
    ipiv                            = ivector(Mdim);
    // double *coeff_e                 = dvector(P+1);
    std::vector<double> coeff_u(P+1);
    std::vector<double> coeff_e(P+1);
    std::vector<double> coeff_eo0(P+1);
    std::vector<double> coeff_eo1(P+1);
    double *tmp                     = dvector(Mdim);
    double *UcoeffU                 = dvector(Mdim);
    double *Fcoeff                  = dvector(Mdim);
    double *Ucoeff                  = dvector(Mdim);
    double *Ucoeff_o0               = dvector(Mdim);
    double *Ucoeff_o1               = dvector(Mdim);
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

        for(int i = 0;i < np;i++)
        {
            quad_u[i] = U_DG[i+eln*np];
            //std::cout << "quad_u[i] " << quad_u[i] << std::endl;
        }

        //ForwardTransform(np, P, z, w, Jac[eln], quad_e, coeff_e);
        //std::vector<double> coeff_e = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e, np);

        std::vector<double> coeff_e = ForwardTransform(P, np, basis, wquad, nq, J, quad_e);
        std::vector<double> coeff_u = ForwardTransform(P, np, basis, wquad, nq, J, quad_u);

        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff[i+eln*(P+1)] = coeff_e[i];
            UcoeffU[i+eln*(P+1)] = coeff_u[i];
            //std::cout << "coeff_u[i] " << coeff_u[i] << std::endl;
        }
        //std::cout << std::endl;
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



    std::map<int,std::vector<double> > Umap;
    GetAllFwdBwdMap(Nel, np, U_DG, Umap);
    std::map<int,std::vector<double> > UmapCoeff;
    GetAllFwdBwdMapCoeff(Nel, P, UcoeffU, UmapCoeff);
    // for(int i=0;i<UtL.size();i++)
    // {
    //     std::cout << "Ut " << UtL[i] << " " << UtR[i] << std::endl;

    // }

    std::map<int,std::vector<double> >::iterator itm;
    std::map<int,std::vector<double> > Fmap;
    std::vector<double> FLUX_new(Nel+1);

    int normalL =  1;
    int normalR =  1;
    for(itm=Umap.begin();itm!=Umap.end();itm++)
    {
        double uLFwd = 0.0, uLBwd = 0.0, FLFwd = 0.0, FLBwd = 0.0;
        double uRFwd = 0.0, uRBwd = 0.0, FRFwd = 0.0, FRBwd = 0.0;

        double uLFwdCoeff = 0.0, uLBwdCoeff = 0.0, FLFwdCoeff = 0.0, FLBwdCoeff = 0.0;
        double uRFwdCoeff = 0.0, uRBwdCoeff = 0.0, FRFwdCoeff = 0.0, FRBwdCoeff = 0.0;

        int elid = itm->first;
        double J = Jac[elid];
        if(itm->first == 0)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[Nel-1][1];
            
            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];  

            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid][0];
            
            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid+1][0];   

        }
        else if(itm->first == Nel-1)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[0][0];


            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid-1][1];

            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid][1];
        }
        else
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];

            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid-1][1];

            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid+1][0];
        }


        FLFwd = uLFwd*uLFwd*0.5;
        FRFwd = uRFwd*uRFwd*0.5;

        FLBwd = uLBwd*uLBwd*0.5;
        FRBwd = uRBwd*uRBwd*0.5;

        FLFwdCoeff = uLFwdCoeff*uLFwdCoeff*0.5;
        FRFwdCoeff = uRFwdCoeff*uRFwdCoeff*0.5;

        FLBwdCoeff = uLBwdCoeff*uLBwdCoeff*0.5;
        FRBwdCoeff = uRBwdCoeff*uRBwdCoeff*0.5;

        double Fl = LaxFriedrichsRiemann(uLFwd,uLBwd,1.0);
        double Fr = LaxFriedrichsRiemann(uRFwd,uRBwd,1.0);

        double FlCoeff = LaxFriedrichsRiemann(uLFwdCoeff,uLBwdCoeff,1.0);
        double FrCoeff = LaxFriedrichsRiemann(uRFwdCoeff,uRBwdCoeff,1.0);
        
        // if (uLFwd >= uLBwd)
        // {
        //     Fmap[elid].push_back((FLFwd));
        // }
        // if(uLFwd < uLBwd)
        // {
        //     Fmap[elid].push_back((FLBwd));
        // }


        // if (uRFwd >= uRBwd)
        // {
        //     Fmap[elid].push_back((FRFwd));
        // }
        // if(uRFwd < uRBwd)
        // {
        //     Fmap[elid].push_back((FRBwd));
        // }

        //std::cout << "ul -> " << uLFwd << "  " << uLBwd << " == uR -> "<< uRFwd << "  " << uRBwd <<std::endl;

        // std::cout << "uLFwdCoeff -- ("<< uLFwdCoeff << ", "<<  uRFwdCoeff << ") (" << uLFwd << ", " << uRFwd <<")" << std::endl;
        // Fmap[elid].push_back(normalL*(0));
        // Fmap[elid].push_back(normalR*(0));

        // Fmap[elid].push_back(normalL*(Fl-FLFwd));
        // Fmap[elid].push_back(normalR*(Fr-FRFwd));

        // Fmap[elid].push_back(normalL*((Fl)-((FLFwd+FLBwd))*0.5));
        // Fmap[elid].push_back(normalR*((Fr)-((FRFwd+FRBwd))*0.5));

        // Fmap[elid].push_back(normalL*(Fl-FLFwd));
        // Fmap[elid].push_back(normalR*(Fr-FRFwd));

        // Fmap[elid].push_back((FLFwd));
        // Fmap[elid].push_back((FRFwd));

        Fmap[elid].push_back((FLBwd));
        Fmap[elid].push_back((FRBwd));
        


        // Fmap[elid].push_back(normalL*(Fl));
        // Fmap[elid].push_back(normalR*(Fr));
        
        // Fmap[elid].push_back(FrCoeff-FRFwdCoeff);

        //std::cout << " Fl -> (" << "("<< Fl << ","<<FLFwd<<") " << " Fr -> (" << Fr << ","<<FRFwd<<") " << std::endl;
    }


    double *numcoeff = dvector(Mdim);
    int cnt = 0;
    
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff[i] = 0.0;
    }
    double *numcoeff2 = dvector(Mdim);
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff2[i] = 0.0;
    }
    
    std::map<int,std::vector<double> > element2flux;
    std::vector<double> flux_res(Nel);
    std::map<int,std::vector<double> > JumpLeftMap;

    std::map<int,std::vector<double> > JumpRightMap;
    for(int i = 0;i < Nel;i++)
    {
        
        if(i == 0)
        {   
            numcoeff[0]                 =       -Fmap[i][0];
            numcoeff[P]                 =       Fmap[i][1];

            numcoeff2[0]                 =      Umap[i][0];
            numcoeff2[P]                 =      Umap[i][1];
        }
        else if(i == Nel-1)
        {

            numcoeff[(Nel-1)*(P+1)]     =       -Fmap[i][0];
            numcoeff[Nel*(P+1)-1  ]     =       Fmap[i][1];

            numcoeff2[(Nel-1)*(P+1)]    =      Umap[i][0];
            numcoeff2[Nel*(P+1)-1]      =      Umap[i][1];

        }
        else
        {

            numcoeff[i*(P+1)]           =       -Fmap[i][0];
            numcoeff[i*(P+1)+P]         =       Fmap[i][1];


            numcoeff2[i*(P+1)]           =      Umap[i][0];
            numcoeff2[i*(P+1)+P]         =      Umap[i][1];


        }

                    //std::cout << "Fmap " << Fmap[i][0] << " " << Fmap[i][1] << std::endl;
        
    }

    // for(int i = 0;i<Mdim;i++)
    // {
    //     std::cout << "numcoeff " << numcoeff[i] << " " << numcoeff2[i]*numcoeff2[i]*0.5 << std::endl;
    // }


    GetGlobalStiffnessMatrixWeakNew(Nel, P, wquad, D, Jac, map, Mdim, basis, StiffnessMatGlobal);

    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,Fcoeff,&ONE_INT,&ZERO_DOUBLE,tmp,&ONE_INT);
    
    for(int i = 0;i < Mdim; i++)
    {
        Ucoeff[i] = tmp[i]-numcoeff[i];
        Ucoeff_o0[i] = -tmp[i];
        Ucoeff_o1[i] =  numcoeff[i];

        // std::cout << "numcoeff["<<i<<"]="<<numcoeff[i]<<std::endl;
    }
    //std::cout << std::endl;
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, basis, MassMatGlobal);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal[0], &Mdim, ipiv, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff, &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_o0, &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_o1, &Mdim, &INFO);
    // Transform back onto quadrature points.
    std::vector<double> R_DG_tmp0(Nel*np,0.0);
    std::vector<double> R_DG_tmp1(Nel*np,0.0);
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        for(int i = 0;i<(P+1);i++)
        {
            coeff_e[i]   = Ucoeff[i+eln*(P+1)];
            coeff_eo0[i] = Ucoeff_o0[i+eln*(P+1)];
            coeff_eo1[i] = Ucoeff_o1[i+eln*(P+1)];
        }
        for(int i=0;i<np;i++)
        {
            quad_e[i]  =0.0;
            quad_eo0[i]=0.0;
            quad_eo1[i]=0.0;
        }

        std::vector<double> quad_e = BackwardTransform(P,  np,  basis,  coeff_e);
        std::vector<double> quad_eo0 = BackwardTransform(P,  np,  basis,  coeff_eo0);
        std::vector<double> quad_eo1 = BackwardTransform(P,  np,  basis,  coeff_eo1);
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
            R_DG_tmp0[i+np*eln] = quad_eo0[i];
            R_DG_tmp1[i+np*eln] = quad_eo1[i];
            // if(i==0)
            // {
            //     R_DG[i+np*eln]=R_DG[i+np*eln]-Fmap[eln][0];
            // }
            // if(i==np-1)
            // {
            //     R_DG[i+np*eln]=R_DG[i+np*eln]+Fmap[eln][1];
            // }
        }


    }


        ofstream solout;
    solout.open("dgRHSdata.out");
    for(int i = 0;i < (Nel*np);i++)
    {
        solout << R_DG[i] << " " << R_DG_tmp0[i] << " " << R_DG_tmp1[i]<< endl;
    }
    solout.close();



    // GetAllFwdBwd(Nel, np, bc, R_DG, UtL, UtR);
    // std::cout << std::endl;
    // for(int i = 0;i < (Nel+1);i++)
    // {
    //     FtLn[i]      =  nL*(UtL[i]*UtL[i])*0.5;
    //     FtRn[i]      =  nR*(UtR[i]*UtR[i])*0.5;

    //     alphaL[i]   = (UtL[i]);
    //     alphaR[i]   = (UtR[i]);

    //     FLUX[i]     = 0.5*(FtLn[i]+FtRn[i])-max(fabs(alphaL[i]),fabs(alphaR[i]))*(UtR[i]-UtL[i]);

    //     DeltaF_l[i] = FLUX[i]-FtLn[i];
    //     DeltaF_r[i] = FLUX[i]-FtRn[i];

    //     std::cout << FLUX[i] << " " << FtLn[i] << " " << FtRn[i] << std::endl;

    // }
}




void CalculateRHSStrongFREuler(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, std::vector<std::vector<double> > bc, std::vector<double> X_DG, std::vector<std::vector<double> > U_DG, std::vector<std::vector<double> > &R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;
    
    int Mdim                        = (P+1)*Nel;
    std::vector<std::vector<double> > F_DG(3);
    ipiv                            = ivector(Mdim);

    std::vector<double> F_DG_row0(Nel*np,0.0);
    std::vector<double> F_DG_row1(Nel*np,0.0);
    std::vector<double> F_DG_row2(Nel*np,0.0);

    for(int eln=0;eln<Nel;eln++)
    {
        for(int i=0;i<np;i++)
        {
            // Evaluate the flux at each quadrature point.
            F_DG_row0[i + eln*np] = 0.5*U_DG[0][i + eln*np]*U_DG[0][i + eln*np];
            F_DG_row1[i + eln*np] = 0.5*U_DG[0][i + eln*np]*U_DG[0][i + eln*np];
            F_DG_row2[i + eln*np] = 0.5*U_DG[0][i + eln*np]*U_DG[0][i + eln*np];

        }
    }

    F_DG[0] = F_DG_row0;
    F_DG[1] = F_DG_row1;
    F_DG[2] = F_DG_row2;
    // Transform fluxes forward into coefficient space.
    //==========================================================

    std::vector<std::vector<double> > F_DG_coeff(3);
    std::vector<double> quad_eq0(np,0.0);
    std::vector<double> quad_eq1(np,0.0);
    std::vector<double> quad_eq2(np,0.0);
    std::vector<double> dx(Nel);

    std::vector<double> Fcoeff_eq0(Nel*(P+1),0.0);
    std::vector<double> Fcoeff_eq1(Nel*(P+1),0.0);
    std::vector<double> Fcoeff_eq2(Nel*(P+1),0.0);
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        double xstart = X_DG[eln*np];
        double xend   = X_DG[eln*np+np-1];

        dx[eln] = xend-xstart; 
        //std::cout << eln << " " << dx[eln] << std::endl;
        for(int i = 0;i < np;i++)
        {
            quad_eq0[i] = F_DG[0][i+eln*np];
            quad_eq1[i] = F_DG[1][i+eln*np];
            quad_eq2[i] = F_DG[2][i+eln*np];
        }

        
        std::vector<double> coeff_eq0 = ForwardTransform(P, np, basis, wquad, nq, J, quad_eq0);
        std::vector<double> coeff_eq1 = ForwardTransform(P, np, basis, wquad, nq, J, quad_eq1);
        std::vector<double> coeff_eq2 = ForwardTransform(P, np, basis, wquad, nq, J, quad_eq2);


        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff_eq0[i+eln*(P+1)] = coeff_eq0[i];
            Fcoeff_eq1[i+eln*(P+1)] = coeff_eq1[i];
            Fcoeff_eq2[i+eln*(P+1)] = coeff_eq2[i];
        }
        //std::cout << std::endl;
    }

    F_DG_coeff[0] = Fcoeff_eq0;
    F_DG_coeff[1] = Fcoeff_eq1;
    F_DG_coeff[2] = Fcoeff_eq2;

    //==========================================================
    std::map<int,std::vector<double> > Umap;
    GetAllFwdBwdMap(Nel, np, U_DG[0].data(), Umap);

    std::map<int,std::vector<std::vector<double> > > Umap_v;
    GetAllFwdBwdMapVec(Nel, np, U_DG, Umap_v, 3);
    
    std::map<int,std::vector<double> >::iterator itm;
    std::map<int,std::vector<double> > Fmap;
    std::vector<double> FLUX_new(Nel+1);

    int normalL =  -1;
    int normalR =  1;
    for(itm=Umap.begin();itm!=Umap.end();itm++)
    {
        double uLFwd = 0.0, uLBwd = 0.0, FLFwd = 0.0, FLBwd = 0.0;
        double uRFwd = 0.0, uRBwd = 0.0, FRFwd = 0.0, FRBwd = 0.0;

        int elid = itm->first;
        double J = Jac[elid];
        if(itm->first == 0)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[Nel-1][1];
            
            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];  



        }
        else if(itm->first == Nel-1)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[0][0];


        }
        else
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];

        }


        FLFwd = uLFwd*uLFwd*0.5;
        FRFwd = uRFwd*uRFwd*0.5;

        FLBwd = uLBwd*uLBwd*0.5;
        FRBwd = uRBwd*uRBwd*0.5;


        double Fl = LaxFriedrichsRiemann(uLFwd,uLBwd,1.0);
        double Fr = LaxFriedrichsRiemann(uRFwd,uRBwd,1.0);

        Fmap[elid].push_back(normalL*(-Fl+FLBwd));
        Fmap[elid].push_back(normalR*(-Fr+FRBwd));
    }


    std::map<int,std::vector<std::vector<double> > > Fmap_v;
    std::map<int,std::vector<std::vector<double> > >::iterator itmv;
    //============================================================
    for(itmv=Umap_v.begin();itmv!=Umap_v.end();itmv++)
    {
        int nvar = 3;
        int elid = itmv->first;
        double J = Jac[elid];
        std::vector<double> uLFwd(nvar,0.0);
        std::vector<double> uLBwd(nvar,0.0);
        std::vector<double> FLFwd(nvar,0.0);
        std::vector<double> FLBwd(nvar,0.0);

        std::vector<double> uRFwd(nvar,0.0);
        std::vector<double> uRBwd(nvar,0.0);
        std::vector<double> FRFwd(nvar,0.0);
        std::vector<double> FRBwd(nvar,0.0);

        for(int n=0;n<nvar;n++)
        {
            if(elid == 0)
            {
                uLFwd[n] = Umap_v[elid][n][0];
                uLBwd[n] = Umap_v[Nel-1][n][1];
                //uLBwd[n] = 2.0*bc[0][n]-uLFwd[n];
                
                uRFwd[n] = Umap_v[elid][n][1];
                uRBwd[n] = Umap_v[elid+1][n][0];  
            }
            else if(elid == Nel-1)
            {
                uLFwd[n] = Umap_v[elid][n][0];
                uLBwd[n] = Umap_v[elid-1][n][1];

                uRFwd[n] = Umap_v[elid][n][1];
                uRBwd[n] = Umap_v[0][n][0];
                //uRBwd[n] = 2.0*bc[1][n]-uRFwd[n];
            }
            else
            {
                uLFwd[n] = Umap_v[elid][n][0];
                uLBwd[n] = Umap_v[elid-1][n][1];

                uRFwd[n] = Umap_v[elid][n][1];
                uRBwd[n] = Umap_v[elid+1][n][0];
            }
        }

        FLFwd[0] = uLFwd[0]*uLFwd[0]*0.5;
        FLFwd[1] = uLFwd[1]*uLFwd[1]*0.5;
        FLFwd[2] = uLFwd[2]*uLFwd[2]*0.5;

        FRFwd[0] = uRFwd[0]*uRFwd[0]*0.5;
        FRFwd[1] = uRFwd[1]*uRFwd[1]*0.5;
        FRFwd[2] = uRFwd[2]*uRFwd[2]*0.5;

        FLBwd[0] = uLBwd[0]*uLBwd[0]*0.5;
        FLBwd[1] = uLBwd[1]*uLBwd[1]*0.5;
        FLBwd[2] = uLBwd[2]*uLBwd[2]*0.5;

        FRBwd[0] = uRBwd[0]*uRBwd[0]*0.5;
        FRBwd[1] = uRBwd[1]*uRBwd[1]*0.5;
        FRBwd[2] = uRBwd[2]*uRBwd[2]*0.5;
        
        std::vector<double> Fl = LaxFriedrichsRiemannVec(uLFwd,uLBwd,1.0);
        std::vector<double> Fr = LaxFriedrichsRiemannVec(uRFwd,uRBwd,1.0);

        std::vector<double> Fleft(nvar,0.0);
        std::vector<double> Fright(nvar,0.0);

        for(int n=0;n<nvar;n++)
        {
            Fleft[n]  = normalL*(-Fl[n]+FLBwd[n]);
            Fright[n] = normalR*(-Fr[n]+FRBwd[n]);
           
        }

        Fmap_v[elid].push_back(Fleft);
        Fmap_v[elid].push_back(Fright);
    }


    std::vector<double> numcoeff_eq0(Mdim,0.0);
    std::vector<double> numcoeff_eq1(Mdim,0.0);
    std::vector<double> numcoeff_eq2(Mdim,0.0);

    for(int i = 0;i < Nel;i++)
    {
        
        if(i == 0)
        {   
            numcoeff_eq0[0]                 =       -Fmap_v[i][0][0];
            numcoeff_eq0[P]                 =       Fmap_v[i][1][0];

            numcoeff_eq1[0]                 =       -Fmap_v[i][0][1];
            numcoeff_eq1[P]                 =       Fmap_v[i][1][1];

            numcoeff_eq2[0]                 =       -Fmap_v[i][0][2];
            numcoeff_eq2[P]                 =       Fmap_v[i][1][2];
        }
        else if(i == Nel-1)
        {

            numcoeff_eq0[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][0];
            numcoeff_eq0[Nel*(P+1)-1  ]     =       Fmap_v[i][1][0];

            numcoeff_eq1[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][1];
            numcoeff_eq1[Nel*(P+1)-1  ]     =       Fmap_v[i][1][1];

            numcoeff_eq2[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][2];
            numcoeff_eq2[Nel*(P+1)-1  ]     =       Fmap_v[i][1][2];


        }
        else
        {

            numcoeff_eq0[i*(P+1)]           =       -Fmap_v[i][0][0];
            numcoeff_eq0[i*(P+1)+P]         =       Fmap_v[i][1][0];

            numcoeff_eq1[i*(P+1)]           =       -Fmap_v[i][0][1];
            numcoeff_eq1[i*(P+1)+P]         =       Fmap_v[i][1][1];

            numcoeff_eq2[i*(P+1)]           =       -Fmap_v[i][0][2];
            numcoeff_eq2[i*(P+1)+P]         =       Fmap_v[i][1][2];

        }        
    }

    double **StiffnessMatGlobal     = dmatrix(Mdim);
    GetGlobalStiffnessMatrixNew(Nel, P, wquad, D, Jac, map, Mdim, basis, StiffnessMatGlobal);
    double *tmp0                     = dvector(Mdim);
    double *tmp1                     = dvector(Mdim);
    double *tmp2                     = dvector(Mdim);

    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,F_DG_coeff[0].data(),&ONE_INT,&ZERO_DOUBLE,tmp0,&ONE_INT);
    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,F_DG_coeff[1].data(),&ONE_INT,&ZERO_DOUBLE,tmp1,&ONE_INT);
    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,F_DG_coeff[2].data(),&ONE_INT,&ZERO_DOUBLE,tmp2,&ONE_INT);


    std::vector<double> Ucoeff_eq0(Mdim,0.0);
    std::vector<double> Ucoeff_eq1(Mdim,0.0);
    std::vector<double> Ucoeff_eq2(Mdim,0.0);
    for(int i = 0;i < Mdim; i++)
    {
        Ucoeff_eq0[i] = -tmp0[i]+numcoeff_eq0[i];
        Ucoeff_eq1[i] = -tmp1[i]+numcoeff_eq1[i];
        Ucoeff_eq2[i] = -tmp2[i]+numcoeff_eq2[i];

    }
    double **MassMatGlobal          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, basis, MassMatGlobal);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal[0], &Mdim, ipiv, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_eq0.data(), &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_eq1.data(), &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_eq2.data(), &Mdim, &INFO);
    // Transform back onto quadrature points.
    std::vector<double> R_DG_row0(Nel*np,0.0);
    std::vector<double> R_DG_row1(Nel*np,0.0);
    std::vector<double> R_DG_row2(Nel*np,0.0);

    std::vector<double> coeff_eq0_tmp(P+1);
    std::vector<double> coeff_eq1_tmp(P+1);
    std::vector<double> coeff_eq2_tmp(P+1);

    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        for(int i = 0;i<(P+1);i++)
        {
            coeff_eq0_tmp[i]   = Ucoeff_eq0[i+eln*(P+1)];
            coeff_eq1_tmp[i]   = Ucoeff_eq1[i+eln*(P+1)];
            coeff_eq2_tmp[i]   = Ucoeff_eq2[i+eln*(P+1)];
        }
        

        std::vector<double> quad_e0 = BackwardTransform(P,  np,  basis,  coeff_eq0_tmp);
        std::vector<double> quad_e1 = BackwardTransform(P,  np,  basis,  coeff_eq0_tmp);
        std::vector<double> quad_e2 = BackwardTransform(P,  np,  basis,  coeff_eq0_tmp);
        int Pf = P - 1;

        for(int i = 0;i < np;i++)
        {

            R_DG_row0[i+np*eln] = quad_e0[i];
            R_DG_row1[i+np*eln] = quad_e1[i];
            R_DG_row2[i+np*eln] = quad_e2[i];

        }


        R_DG[0] = R_DG_row0;
        R_DG[1] = R_DG_row1;
        R_DG[2] = R_DG_row2;

    }

}





std::vector<double> LaxFriedrichsRiemannVec(std::vector<double> Ul, std::vector<double> Ur, double normal)
{

    int nvar = Ul.size();
    std::vector<double> Fn(nvar,0.0);

    std::vector<double> Flvec(3,0.0);
    Flvec[0] = Ul[0]*Ul[0]*0.5;
    Flvec[1] = Ul[1]*Ul[1]*0.5;
    Flvec[2] = Ul[2]*Ul[2]*0.5;
    std::vector<double> Frvec(3,0.0);
    Frvec[0] = Ur[0]*Ur[0]*0.5;
    Frvec[1] = Ur[1]*Ur[1]*0.5;
    Frvec[2] = Ur[2]*Ur[2]*0.5;

   for(int n=0;n<nvar;n++)
    {
        
        double Fl = Flvec[n];
        double Fr = Frvec[n];

        double alphaL   = Ul[n];
        double alphaR   = Ur[n];
        Fn[n] = 0.5*(Fl+Fr)*normal-max(fabs(alphaL),fabs(alphaR))*(Ur[n]-Ul[n]);
        //std::cout <<"flux " << n << " "<< Fn[n] << " " << Fl << " " << Fr << " " << Ur[n] << " " << Ul[n] << " " << std::endl;
    }
    //std::cout << std::endl;
    return Fn;
}






void CalculateRHSStrongFR(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;
    
    int Mdim                        = (P+1)*Nel;
    double *F_DG                    = dvector(Nel*np);
    // double *quad_e                  = dvector(np);
    std::vector<double> quad_e(np);
    std::vector<double> quad_u(np);
    std::vector<double> quad_eo0(np);
    std::vector<double> quad_eo1(np);
    ipiv                            = ivector(Mdim);
    // double *coeff_e                 = dvector(P+1);
    std::vector<double> coeff_u(P+1);
    std::vector<double> coeff_e(P+1);
    std::vector<double> coeff_eo0(P+1);
    std::vector<double> coeff_eo1(P+1);
    double *tmp                     = dvector(Mdim);
    double *UcoeffU                 = dvector(Mdim);
    double *Fcoeff                  = dvector(Mdim);
    double *Ucoeff                  = dvector(Mdim);
    double *Ucoeff_o0               = dvector(Mdim);
    double *Ucoeff_o1               = dvector(Mdim);
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

        for(int i = 0;i < np;i++)
        {
            quad_u[i] = U_DG[i+eln*np];
            //std::cout << "quad_u[i] " << quad_u[i] << std::endl;
        }

        //ForwardTransform(np, P, z, w, Jac[eln], quad_e, coeff_e);
        //std::vector<double> coeff_e = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e, np);

        std::vector<double> coeff_e = ForwardTransform(P, np, basis, wquad, nq, J, quad_e);
        std::vector<double> coeff_u = ForwardTransform(P, np, basis, wquad, nq, J, quad_u);

        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff[i+eln*(P+1)] = coeff_e[i];
            UcoeffU[i+eln*(P+1)] = coeff_u[i];
            //std::cout << "coeff_u[i] " << coeff_u[i] << std::endl;
        }
        //std::cout << std::endl;
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



    std::map<int,std::vector<double> > Umap;
    GetAllFwdBwdMap(Nel, np, U_DG, Umap);
    std::map<int,std::vector<double> > UmapCoeff;
    GetAllFwdBwdMapCoeff(Nel, P, UcoeffU, UmapCoeff);
    // for(int i=0;i<UtL.size();i++)
    // {
    //     std::cout << "Ut " << UtL[i] << " " << UtR[i] << std::endl;

    // }

    std::map<int,std::vector<double> >::iterator itm;
    std::map<int,std::vector<double> > Fmap;
    std::vector<double> FLUX_new(Nel+1);

    int normalL =  -1;
    int normalR =  1;
    for(itm=Umap.begin();itm!=Umap.end();itm++)
    {
        double uLFwd = 0.0, uLBwd = 0.0, FLFwd = 0.0, FLBwd = 0.0;
        double uRFwd = 0.0, uRBwd = 0.0, FRFwd = 0.0, FRBwd = 0.0;

        double uLFwdCoeff = 0.0, uLBwdCoeff = 0.0, FLFwdCoeff = 0.0, FLBwdCoeff = 0.0;
        double uRFwdCoeff = 0.0, uRBwdCoeff = 0.0, FRFwdCoeff = 0.0, FRBwdCoeff = 0.0;

        int elid = itm->first;
        double J = Jac[elid];
        if(itm->first == 0)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[Nel-1][1];
            
            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];  

            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid][0];
            
            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid+1][0];   

        }
        else if(itm->first == Nel-1)
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[0][0];


            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid-1][1];

            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid][1];
        }
        else
        {
            uLFwd = Umap[elid][0];
            uLBwd = Umap[elid-1][1];

            uRFwd = Umap[elid][1];
            uRBwd = Umap[elid+1][0];

            uLFwdCoeff = UmapCoeff[elid][0];
            uLBwdCoeff = UmapCoeff[elid-1][1];

            uRFwdCoeff = UmapCoeff[elid][1];
            uRBwdCoeff = UmapCoeff[elid+1][0];
        }


        FLFwd = uLFwd*uLFwd*0.5;
        FRFwd = uRFwd*uRFwd*0.5;

        FLBwd = uLBwd*uLBwd*0.5;
        FRBwd = uRBwd*uRBwd*0.5;

        FLFwdCoeff = uLFwdCoeff*uLFwdCoeff*0.5;
        FRFwdCoeff = uRFwdCoeff*uRFwdCoeff*0.5;

        FLBwdCoeff = uLBwdCoeff*uLBwdCoeff*0.5;
        FRBwdCoeff = uRBwdCoeff*uRBwdCoeff*0.5;

        double Fl = LaxFriedrichsRiemann(uLFwd,uLBwd,1.0);
        double Fr = LaxFriedrichsRiemann(uRFwd,uRBwd,1.0);

        double FlCoeff = LaxFriedrichsRiemann(uLFwdCoeff,uLBwdCoeff,1.0);
        double FrCoeff = LaxFriedrichsRiemann(uRFwdCoeff,uRBwdCoeff,1.0);
        
        // if (uLFwd >= uLBwd)
        // {
        //     Fmap[elid].push_back((FLFwd));
        // }
        // if(uLFwd < uLBwd)
        // {
        //     Fmap[elid].push_back((FLBwd));
        // }


        // if (uRFwd >= uRBwd)
        // {
        //     Fmap[elid].push_back((FRFwd));
        // }
        // if(uRFwd < uRBwd)
        // {
        //     Fmap[elid].push_back((FRBwd));
        // }

        //std::cout << "ul -> " << uLFwd << "  " << uLBwd << " == uR -> "<< uRFwd << "  " << uRBwd <<std::endl;

        // std::cout << "uLFwdCoeff -- ("<< uLFwdCoeff << ", "<<  uRFwdCoeff << ") (" << uLFwd << ", " << uRFwd <<")" << std::endl;
        // Fmap[elid].push_back(normalL*(0));
        // Fmap[elid].push_back(normalR*(0));

        // Fmap[elid].push_back(normalL*(Fl-FLFwd));
        // Fmap[elid].push_back(normalR*(Fr-FRFwd));

        // Fmap[elid].push_back(normalL*((Fl)-((FLFwd+FLBwd))*0.5));
        // Fmap[elid].push_back(normalR*((Fr)-((FRFwd+FRBwd))*0.5));

        Fmap[elid].push_back(normalL*(-Fl+FLBwd));
        Fmap[elid].push_back(normalR*(-Fr+FRBwd));

        // Fmap[elid].push_back((FLFwd));
        // Fmap[elid].push_back((FRFwd));

        // Fmap[elid].push_back((FLBwd));
        // Fmap[elid].push_back((FRBwd));
        


        // Fmap[elid].push_back(normalL*(Fl));
        // Fmap[elid].push_back(normalR*(Fr));
        
        // Fmap[elid].push_back(FrCoeff-FRFwdCoeff);

        //std::cout << " Fl -> (" << "("<< Fl << ","<<FLFwd<<") " << " Fr -> (" << Fr << ","<<FRFwd<<") " << std::endl;
    }


    double *numcoeff = dvector(Mdim);
    int cnt = 0;
    
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff[i] = 0.0;
    }
    double *numcoeff2 = dvector(Mdim);
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff2[i] = 0.0;
    }
    
    std::map<int,std::vector<double> > element2flux;
    std::vector<double> flux_res(Nel);
    std::map<int,std::vector<double> > JumpLeftMap;

    std::map<int,std::vector<double> > JumpRightMap;
    for(int i = 0;i < Nel;i++)
    {
        
        if(i == 0)
        {   
            numcoeff[0]                 =       -Fmap[i][0];
            numcoeff[P]                 =       Fmap[i][1];

            numcoeff2[0]                 =      Umap[i][0];
            numcoeff2[P]                 =      Umap[i][1];
        }
        else if(i == Nel-1)
        {

            numcoeff[(Nel-1)*(P+1)]     =       -Fmap[i][0];
            numcoeff[Nel*(P+1)-1  ]     =       Fmap[i][1];

            numcoeff2[(Nel-1)*(P+1)]    =      Umap[i][0];
            numcoeff2[Nel*(P+1)-1]      =      Umap[i][1];

        }
        else
        {

            numcoeff[i*(P+1)]           =       -Fmap[i][0];
            numcoeff[i*(P+1)+P]         =       Fmap[i][1];


            numcoeff2[i*(P+1)]           =      Umap[i][0];
            numcoeff2[i*(P+1)+P]         =      Umap[i][1];


        }

                    //std::cout << "Fmap " << Fmap[i][0] << " " << Fmap[i][1] << std::endl;
        
    }

    // for(int i = 0;i<Mdim;i++)
    // {
    //     std::cout << "numcoeff " << numcoeff[i] << " " << numcoeff2[i]*numcoeff2[i]*0.5 << std::endl;
    // }


    GetGlobalStiffnessMatrixNew(Nel, P, wquad, D, Jac, map, Mdim, basis, StiffnessMatGlobal);

    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,Fcoeff,&ONE_INT,&ZERO_DOUBLE,tmp,&ONE_INT);
    
    for(int i = 0;i < Mdim; i++)
    {
        Ucoeff[i] = -tmp[i]+numcoeff[i];
        Ucoeff_o0[i] = -tmp[i];
        Ucoeff_o1[i] =  numcoeff[i];

        // std::cout << "numcoeff["<<i<<"]="<<numcoeff[i]<<std::endl;
    }
    //std::cout << std::endl;
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, basis, MassMatGlobal);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal[0], &Mdim, ipiv, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff, &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_o0, &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff_o1, &Mdim, &INFO);
    // Transform back onto quadrature points.
    std::vector<double> R_DG_tmp0(Nel*np,0.0);
    std::vector<double> R_DG_tmp1(Nel*np,0.0);
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        for(int i = 0;i<(P+1);i++)
        {
            coeff_e[i]   = Ucoeff[i+eln*(P+1)];
            coeff_eo0[i] = Ucoeff_o0[i+eln*(P+1)];
            coeff_eo1[i] = Ucoeff_o1[i+eln*(P+1)];
        }
        for(int i=0;i<np;i++)
        {
            quad_e[i]  =0.0;
            quad_eo0[i]=0.0;
            quad_eo1[i]=0.0;
        }

        std::vector<double> quad_e = BackwardTransform(P,  np,  basis,  coeff_e);
        std::vector<double> quad_eo0 = BackwardTransform(P,  np,  basis,  coeff_eo0);
        std::vector<double> quad_eo1 = BackwardTransform(P,  np,  basis,  coeff_eo1);
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
            R_DG_tmp0[i+np*eln] = quad_eo0[i];
            R_DG_tmp1[i+np*eln] = quad_eo1[i];
            // if(i==0)
            // {
            //     R_DG[i+np*eln]=R_DG[i+np*eln]-Fmap[eln][0];
            // }
            // if(i==np-1)
            // {
            //     R_DG[i+np*eln]=R_DG[i+np*eln]+Fmap[eln][1];
            // }
        }


    }


        ofstream solout;
    solout.open("dgRHSdata.out");
    for(int i = 0;i < (Nel*np);i++)
    {
        solout << R_DG[i] << " " << R_DG_tmp0[i] << " " << R_DG_tmp1[i]<< endl;
    }
    solout.close();



    // GetAllFwdBwd(Nel, np, bc, R_DG, UtL, UtR);
    // std::cout << std::endl;
    // for(int i = 0;i < (Nel+1);i++)
    // {
    //     FtLn[i]      =  nL*(UtL[i]*UtL[i])*0.5;
    //     FtRn[i]      =  nR*(UtR[i]*UtR[i])*0.5;

    //     alphaL[i]   = (UtL[i]);
    //     alphaR[i]   = (UtR[i]);

    //     FLUX[i]     = 0.5*(FtLn[i]+FtRn[i])-max(fabs(alphaL[i]),fabs(alphaR[i]))*(UtR[i]-UtL[i]);

    //     DeltaF_l[i] = FLUX[i]-FtLn[i];
    //     DeltaF_r[i] = FLUX[i]-FtRn[i];

    //     std::cout << FLUX[i] << " " << FtLn[i] << " " << FtRn[i] << std::endl;

    // }
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
            UtR[i]=  quad[Nel*np-1];
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

void GetAllFwdBwdMap(int Nel, int np, double *quad, std::map<int,std::vector<double> > &Umap)
{
    
    for(int i=0;i<Nel;i++)
    {
        std::vector<double> row(2,0.0);
        if(i == 0)
        {
            row[0] = quad[0];
            row[1] = quad[np-1];

        }
        else if(i == Nel)
        {
            row[0] = quad[Nel*np-1]; 
            row[1] = quad[Nel*np-1];
        }
        else
        {
            row[0] = quad[i*np];
            row[1] = quad[(i+1)*np-1];
        }
        Umap[i] = row;
    }
}


void GetAllFwdBwdMapCoeff(int Nel, int P, double *coeff, std::map<int,std::vector<double> > &Umap)
{
    
    for(int i=0;i<Nel;i++)
    {
        std::vector<double> row(2,0.0);
        if(i == 0)
        {
            row[0] = coeff[0];
            row[1] = coeff[P];

        }
        else if(i == Nel)
        {
            row[0] = coeff[(Nel-1)*(P+1)];
            row[1] = coeff[Nel*(P+1)-1  ];
        }
        else
        {
            row[0] = coeff[i*(P+1)];
            row[1] = coeff[i*(P+1)+P];
        }

        Umap[i] = row;
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



void GetGlobalStiffnessMatrixWeakNew(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal)
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
        GetElementStiffnessMatrixWeakNew(P, wquad, D, Jac[eln], basis, StiffnessElem);
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




void GetAllFwdBwdMapVec(int Nel, int np, std::vector<std::vector<double> > quad, std::map<int,std::vector<std::vector<double> > > &Umap, int nvar)
{
    
    for(int i=0;i<Nel;i++)
    {
        std::vector<std::vector<double> > rowrow(nvar);
        for(int n=0;n<nvar;n++)
        {
            std::vector<double> row(2,0.0);

            if(i == 0)
            {
                row[0] = quad[n][0];
                row[1] = quad[n][np-1];
            }
            else if(i == Nel)
            {
                row[0] = quad[n][(Nel-1)*np-1]; 
                row[1] = quad[n][Nel*np-1];
            }
            else
            {
                row[0] = quad[n][i*np];
                row[1] = quad[n][(i+1)*np-1];
            }

            rowrow[n] = row;
        }

        Umap[i] = rowrow;

    }
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



void GetElementStiffnessMatrixWeakNew(int P, std::vector<double> wquad, double **D, double J, std::vector<std::vector<double> > basis, double **StiffMatElem)
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
    double *dphi1 = dvector(np);
    double *dphi2 = dvector(np);
    //std::cout << "Stiffness Matrix " << std::endl;
    for(int i=0;i<P+1;i++)
    {
        //std::vector<double> phi1 = getLagrangeBasisFunction(i,zquad,nq,z,np,P);
        std::vector<double> phi1 = basis[i];
        diff( np, D, phi1.data(), dphi1, J);

        for(int j=0;j<P+1;j++)
        {
            // std::vector<double> phi2 = getLagrangeBasisFunction(j,zquad,nq,z,np,P);
            std::vector<double> phi2 = basis[j];
            diff( np, D, phi2.data(), dphi2, J);
            //lagrange_basis(np, P, j, z, phi2);
            // diff( np, D, phi2.data(), dphi2, J);
            
            //StiffMatElem[i][j] = J*integr(np, wquad.data(), phi2.data(), dphi1);
            StiffMatElem[i][j] = J*integr(np, wquad.data(), dphi1, phi2.data());

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
// double **dmatrix(int n)
// {
//   double **A;
//   A = (double **)malloc(n*sizeof(double *));
//   A[0] = (double *)malloc(n*n*sizeof(double));

//   for(int i=1;i<n;i++){
//     A[i] = A[i-1]+n;
//   }
//   return A;
// }

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
// double integr(int np, double *w, double *phi1, double *phi2)
// {
//   register double sum = 0.;

//   for(int i=0;i<np;i++){
//     sum = sum + phi1[i]*phi2[i]*w[i]; 
//   }
//   return sum;
// }


// This member function performs numerical differentiation.
// void *diff(int np, double **D, double *p, double *pd, double J)
// {
//     for(int i=0;i<np;i++){
//         pd[i] = 0;
//         for(int j=0;j<np;j++){
//             pd[i] = pd[i] + D[i][j]*p[j];
//         }
//         pd[i] = pd[i]/J;
        
//     }
//     return 0;
// }

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

