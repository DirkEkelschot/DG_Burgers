#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/Polylib.h"
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace polylib;
void CalculateRHS(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG);
void CalculateRHS_v2(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG);
void *negatednormals(int Nel, double *n);
int **iarray(int n,int m);
std::vector<std::vector<double> > getModalBasis(std::vector<double> zquad, int nq, int np, int P);
void GetFwdBwd(int eln, int Nel, int np, double *bc, double *quad, double *UtL, double *UtR);
void GetAllFwdBwd(int Nel, int np, double *bc, double *quad, std::vector<double> &UtL, std::vector<double> &UtR);
void TraceMap(int np, int Nel, int **trace);
std::vector<double> BackwardTransformLagrange(int P, std::vector<double> zquad, std::vector<double> wquad, int nq, double J, std::vector<double> coeff, std::vector<double> z, int np);
void InnerProductWRTDerivBasis(int np, int P, double J, double *w, double *z, double **D, double *F_DG, double *coeff);
void GetGlobalMassMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double *Jac, int **map, int Mdim, double **MassMatGlobal);
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
void GetGlobalStiffnessMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, int Mdim, double **StiffnessGlobal);
std::vector<double> modal_basis(int np, int P, int i, std::vector<double> z);



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






std::vector<double> GetElementMassMatrix_Lagrange(int P, 
                            std::vector<double> z,
                            int np,
                            std::vector<double> zquad, 
                            std::vector<double> wquad, 
                            int nq,
                            double J)
{
    std::vector<double> MassMatElem((P+1)*(P+1),0.0);
    //std::cout << "Mass matrix " << std::endl;
    for(int i=0;i<P+1;i++)
    {
        std::vector<double> phi1 = getLagrangeBasisFunction(i,zquad,nq,z,np,P);
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = getLagrangeBasisFunction(j,zquad,nq,z,np,P);
            MassMatElem[i*(P+1)+j] = J*integr(np, wquad.data(), phi1.data(), phi2.data());
            //std::cout << MassMatElem[i*(P+1)+j] << " ";
        }
        //std::cout << std::endl;
    }
    return MassMatElem;
}



std::vector<double> GetElementMassMatrix_Modal(int P, 
                            std::vector<double> z,
                            int np,
                            std::vector<double> zquad, 
                            std::vector<double> wquad, 
                            int nq,
                            double J)
{
    std::vector<double> MassMatElem((P+1)*(P+1),0.0);
 
    std::vector<std::vector<double> > modal = getModalBasis(zquad, nq, np, P);
    //std::cout << std::endl;
    for(int i=0;i<P+1;i++)
    {
        std::vector<double> phi1 = modal[i];
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = modal[j];
            MassMatElem[i*(P+1)+j] = J*integr(np, wquad.data(), phi1.data(), phi2.data());
            //std::cout << MassMatElem[i*(P+1)+j] << " ";
        }
        //std::cout << std::endl;
    }
    return MassMatElem;
}

std::vector<double> ForwardTransformLagrange(int P, 
                std::vector<double>zquad, std::vector<double>wquad, int nq,
                double J, 
                std::vector<double> z, int np)
{
    
    std::vector<double> coeff(P+1);
    int ncoeffs     = P+1;
    double *Icoeff  = dvector(ncoeffs);
    
    for(int j=0;j<P+1;j++)
    {
        //basis(np, P, j, z, phi1);
        std::vector<double> phi1 = getLagrangeBasisFunction(j,zquad,nq,zquad,np,P);

        Icoeff[j] = J*integr(nq, wquad.data(), phi1.data(), z.data());
        //std::cout << "Icoeff " << Icoeff[j] << std::endl;
    }
    
    std::vector<double> MassMatElem = GetElementMassMatrix_Lagrange(P,zquad,np,zquad,wquad,nq,J);

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

std::vector<double> BackwardTransformLagrange(int P, std::vector<double> zquad, std::vector<double> wquad, int nq,  
                        double J, std::vector<double> coeff, std::vector<double> z, int np)
{
    std::vector<double> quad(np,0.0);
    double sum = 0.0;
    for(int i = 0;i<P+1;i++)
    {
        std::vector<double> phi1 = getLagrangeBasisFunction(i,zquad,nq,z,np,P);
        for( int j=0;j<np;j++)
        {
            quad[j] = quad[j]+coeff[i]*phi1[j];
        }
    }

    return quad;
}





std::vector<double> BackwardTransformModal(int P, std::vector<double> zquad, std::vector<double> wquad, int nq,  
                        double J, std::vector<double> coeff, std::vector<double> z, int np)
{
    std::vector<std::vector<double> > modal = getModalBasis(zquad, nq, np, P);

    std::vector<double> quad(np,0.0);
    double sum = 0.0;
    for(int i = 0;i<P+1;i++)
    {
        std::vector<double> phi1 =modal[i];
        for( int j=0;j<np;j++)
        {
            quad[j] = quad[j]+coeff[i]*phi1[j];
        }
    }

    return quad;
}






std::vector<double> FilterNodalCoeffs(std::vector<double> zquad, 
                                            std::vector<double> wquad, 
                                            std::vector<double> z, 
                                            int np, int nq, 
                                            std::vector<double> coeffs_nodal, int P, int Pf, double J)
{

    std::vector<double> quad_e = BackwardTransformLagrange(P, zquad, wquad, nq, J, coeffs_nodal, z, np);

    std::vector<std::vector<double> > modal = getModalBasis(zquad, nq, np, P);

    std::vector<double> coeffs_update(P+1,0.0);
    std::vector<double> coeffs_modal(P+1,0.0);

    int ncoeffs     = P+1;
    double *Icoeff  = dvector(ncoeffs);
    
    for(int j=0;j<P+1;j++)
    {
        std::vector<double> phi1 = modal[j];

        Icoeff[j] = J*integr(nq, wquad.data(), phi1.data(), quad_e.data());
        //std::cout << "Icoeff[j] " << Icoeff[j]  << " jaccie " << J<< std::endl; 
    }
    
    std::vector<double> MassMatElemModal = GetElementMassMatrix_Modal(P,zquad,np,zquad,wquad,nq,J);

    int ONE_INT=1;
    double ONE_DOUBLE=1.0;
    double ZERO_DOUBLE=0.0;
    unsigned char TR = 'T';
    int INFO;
    int LWORK = ncoeffs*ncoeffs;
    double *WORK = new double[LWORK];
    int *ip = ivector(ncoeffs);
    // Create inverse Mass matrix.
    dgetrf_(&ncoeffs, &ncoeffs, MassMatElemModal.data(), &ncoeffs, ip, &INFO);
    dgetri_(&ncoeffs, MassMatElemModal.data(), &ncoeffs, ip, WORK, &LWORK, &INFO);
    // Apply InvMass to Icoeffs hence M^-1 Icoeff = uhat
    dgemv_(&TR,&ncoeffs,&ncoeffs,&ONE_DOUBLE,MassMatElemModal.data(),&ncoeffs,Icoeff,&ONE_INT,&ZERO_DOUBLE,coeffs_modal.data(),&ONE_INT);

    for(int n=0;n<(Pf+1);n++)
    {
        coeffs_update[n]=coeffs_modal[n];
        //std::cout << "coeffs_modal[n] " << coeffs_modal[n] << " " << coeffs_update.size() << " " << Pf+1 << std::endl;
    }



    //std::vector<double> quad_e_modal = BackwardTransformModal(P, zquad, wquad, nq, J, coeffs_modal, z, np);


    std::vector<double> quad_e_modal_filtered = BackwardTransformModal(P, zquad, wquad, nq, J, coeffs_update, z, np);


    //std::vector<double> coeff_e = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e_modal, np);

    std::vector<double> coeff_e_filtered = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e_modal_filtered, np);

    // for(int n=0;n<coeff_e.size();n++)
    // {
    //     std::cout << "coeffs_nodal["<<n<<"]=" << coeffs_nodal[n] << " = "<< coeff_e[n] << " :: " << coeff_e_filtered[n] << " " << coeff_e[n]-coeff_e_filtered[n]<< std::endl;
    // }

    return coeff_e_filtered;

}

// std::vector<double> modal_basis(int np, int P, int i, std::vector<double> z)
// {
//     std::vector<double> phi(np);
//     if(i == 0){
//         for(int k=0;k<np;k++){
//             phi[k] = (1 - z[k])/2;
//         }
//     }else if(i == P){
//         for(int k=0;k<np;k++){
//             phi[k] = (1 + z[k])/2;
//         }
//     }else{
//         jacobfd(np, z, phi, NULL, i-1, 1.0, 1.0);
//         for(int k=0;k<np;k++){
//             phi[k] = ((1-z[k])/2)*((1+z[k])/2)*phi[k];
//         }
//     }
//     return phi;
// }


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
        std::vector<double> phi1 = modal_basis(nq, P, n, zquad);
        
        basis.push_back(phi1);
    }
    return basis;
}



std::vector<std::vector<double> > getLagrangeBasis(std::vector<double> zquad, int nq, std::vector<double> z, int np, int P)
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
            phi1[q] = hglj(n, zquad[q], z.data(), numModes, 0.0, 0.0);
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
  std::cout << Nel << std::endl;
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
    std::vector<std::vector<double> > basis = getLagrangeBasis(zq,nq,z,np,P);
    double J = 1.0;
    std::vector<double> M = GetElementMassMatrix_Lagrange(P, z, np, zq, wq, nq, J);

    for(int i=0;i<nq;i++)
    {
        std::cout << "w[" << i << "]=" << wq[i] << std::endl;
    }

    for(int i=0;i<(P+1);i++)
    {
        for(int j=0;j<(P+1);j++)
        {
            std::cout << M[i*(P+1)+j] << " ";
        }
        std::cout << std::endl;
    }

    
    std::vector<double> coeffs = ForwardTransformLagrange(P, zq, wq, nq, J, z, np);
    
    for(int i=0;i<P+1;i++)
    {
        std::cout << "c["<<i<<"]=" << coeffs[i] << std::endl;
    }

    std::vector<double> quad = BackwardTransformLagrange(P, zq, wq, nq, J, coeffs, z, np);

    for(int i=0;i<Q+1;i++)
    {
        std::cout << "quad["<<i<<"]=" << quad[i] << std::endl;
    }

    int Pf = P - 1;
    std::vector<double> coeffs_filtered = FilterNodalCoeffs(zq, wq, z, np, nq, coeffs, P, Pf, J);




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
    std::cout << "start initialization" << std::endl;
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

    double* U_DG_new   = dvector(Nel*(np));
    double* R_DG0      = dvector(Nel*(np));
    double* R_DG1      = dvector(Nel*(np));
    double* R_DG2      = dvector(Nel*(np));
    double* R_DG3      = dvector(Nel*(np));
    double* k1         = dvector(Nel*(np));
    double* k2         = dvector(Nel*(np));
    double* k3         = dvector(Nel*(np));
    double* k4         = dvector(Nel*(np));
    int RKstages       = 4;
    double* a          = dvector(RKstages);
    a[0]=1.0/6.0;
    a[1]=1.0/3.0;
    a[2]=1.0/3.0;
    a[3]=1.0/6.0;
    double *bc=dvector(2);
    bc[0]=0.0;
    bc[1]=0.0;
    double time = 0.0;
    for(int t = 0; t < nt;t++)
    {
        

        // CalculateRHS_v2(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, U_DG, R_DG0);
        // for(int i=0;i<(Nel*np);i++)
        // {
        //     U_DG[i] = U_DG[i]+dt*R_DG0[i];
        //     //std::cout << "R_DG0 " << U_DG[i] << " " << R_DG0[i] << std::endl;

        // }

        
        // //Calculate Stage 1;
        CalculateRHS_v2(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, U_DG, R_DG0);
        for(int i=0;i<(Nel*np);i++)
        {
            k1[i] = U_DG[i]+a[0]*dt*R_DG0[i];
            //std::cout << "k1 " << k1[i] << " " << U_DG[i] << std::endl;   
        }
        //std::cout << std::endl;
        //Calculate Stage 2;
        CalculateRHS_v2(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, k1, R_DG1);
        for(int i=0;i<(Nel*np);i++)
        {
            k2[i] = U_DG[i]+a[1]*dt*R_DG1[i];
        }
        //std::cout << std::endl;
        //Calculate Stage 3;
        CalculateRHS_v2(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, k2, R_DG2);
        for(int i=0;i<(Nel*np);i++)
        {
            k3[i] = U_DG[i]+a[2]*dt*R_DG2[i];
        }
        //Calculate Stage 4;
        CalculateRHS_v2(np, nq, Nel, P, zq, wq, z, D, Jac, map, bc, X_DG, k3, R_DG3);
        for(int i=0;i<(Nel*np);i++)
        {
            k4[i] = U_DG[i]+a[3]*dt*R_DG3[i];
            U_DG[i] = k4[i];
        } 
        
        

        

        // double value = std::ceil(time * 100.0) / 100.0;
        // solout.open("dgdata"+std::to_string(value)+".out");
        // for(int i = 0;i < (Nel*np);i++)
        // {
        //     solout << X_DG[i] << " " << U_DG_new[i] << endl;
        // }
        // solout.close();

        

        

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
void InnerProductWRTDerivLagrangeBasis(int np, int P, double J, std::vector<double> wquad, std::vector<double> zquad, std::vector<double> z, double **D, std::vector<double> quad, std::vector<double> &coeff)
{
    int ONE_INT=1;
    double ONE_DOUBLE=1.0;
    double ZERO_DOUBLE=0.0;
    unsigned char TR = 'T';
    
    int nq = zquad.size();

    int ncoeff = P;
    double *tmp   = dvector(np);
    double *phi1  = dvector(np);
    double *dphi1 = dvector(np);
    
    for(int n=0;n<P+1;n++)
    {
        //basis(np, P, n, z, phi1);
        // lagrange_basis(np, P, n, z, z, phi1);
        std::vector<double> phi1 = getLagrangeBasisFunction(n,zquad,nq,zquad,np,P);
        diff(np, D, phi1.data(), dphi1, J);
        coeff[n] = J*integr(np, wquad.data(), dphi1, quad.data());
    }
}
// void CalculateRHS_v2(int np, int Nel, int P, double *z, double *w, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG)

void CalculateRHS_v2(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG)
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
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];
        for(int i = 0;i < np;i++)
        {
            quad_e[i] = F_DG[i+eln*np];
        }
        //ForwardTransform(np, P, z, w, Jac[eln], quad_e, coeff_e);
        std::vector<double> coeff_e = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e, np);
        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff[i+eln*(P+1)] = coeff_e[i];
        }
    }
    //==========================================================
    // Calculate the numerical flux;
    double *quads = dvector(np);
    
    double *n        = dvector(Nel*2);
    // double *UtL      = dvector(Nel*2);
    // double *UtR      = dvector(Nel*2);
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
    cnt = 0;
    int nL = 1;int nR=-1;
    for(int i = 0;i < (Nel+1);i++)
    {
        
        FtLn[i]      =  nL*(UtL[i]*UtL[i])*0.5;
        FtRn[i]      =  nR*(UtR[i]*UtR[i])*0.5;
        
        alphaL[i]   = 2.0*(UtL[i]);
        alphaR[i]   = 2.0*(UtR[i]);
        
        FLUX[i]     = 0.5*(FtLn[i]+FtRn[i])-0.5*max(fabs(alphaL[i]),fabs(alphaR[i]))*(UtR[i]-UtL[i]);
        /*if((UtR[i]-UtL[i])!=0.0)
        {
            alpha[i] = (FtLn[i]-FtRn[i])/(nR*UtR[i]-nL*UtL[i]);
        }
        else
        {
            alpha[i] = 0.0;
        }
        
        if(alpha[i]>=0.0)
        {
            FLUX[i] = FtLn[i];
        }
        else
        {
            FLUX[i] = FtRn[i];
        }
        cout << alpha[i] << endl;*/
    }
    double *numcoeff = dvector(Mdim);
    cnt = 0;
    
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff[i] = 0.0;
    }
    
    for(int i = 0;i < Nel;i++)
    {
        if(i == 0)
        {
            numcoeff[0] = -Jac[i]*FLUX[0];
            numcoeff[P] =  Jac[i]*FLUX[1];
        }
        else if(i == Nel-1)
        {
            numcoeff[(Nel-1)*(P+1)]   = -Jac[i]*FLUX[i];
            numcoeff[Nel*(P+1)-1    ] =  Jac[i]*FLUX[i+1];
        }
        else
        {
            numcoeff[i*(P+1)]     = -Jac[i]*FLUX[i];
            numcoeff[i*(P+1)+P]   =  Jac[i]*FLUX[i+1];
        }
    }

    
    //for(int i = 0;i<Mdim;i++)
    //{
    //   cout<< numcoeff[i] <<endl;
    //}
    //cout << endl;
    // HACK FOR 3 ELEMENTS;
    /*numcoeff[0] =     -Jac[0]*FLUX[0];
    numcoeff[P] =      Jac[0]*FLUX[1];
    
    //numcoeff[(Nel-3)*(P+1)]   =     numf[1];
    //numcoeff[(Nel-3)*(P+1)+P] =     numf[3];
    
    numcoeff[P+1] =    -Jac[1]*FLUX[1];
    numcoeff[(Nel-1)*P+1] =   Jac[1]*FLUX[2];
    
    numcoeff[(Nel-1)*P+2] =     -Jac[2]*FLUX[2];
    numcoeff[(Nel-1)*P-1] =    Jac[2]*FLUX[3];
    */
    
    
    
    // GetGlobalStiffnessMatrix(Nel, P, np, z, w, D, Jac, map, Mdim, StiffnessMatGlobal);
    GetGlobalStiffnessMatrix(Nel, P, np, nq, zquad, wquad, z, D, Jac, map, Mdim, StiffnessMatGlobal);


    dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,Fcoeff,&ONE_INT,&ZERO_DOUBLE,tmp,&ONE_INT);
    
    for(int i = 0;i < Mdim; i++)
    {
        Ucoeff[i] = -tmp[i]-numcoeff[i];
    }
    GetGlobalMassMatrix(Nel, P, np, nq, zquad, wquad, z, Jac, map, Mdim, MassMatGlobal);
    // GetGlobalMassMatrix(Nel, P, np, z, w, Jac, map, Mdim, MassMatGlobal);
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
        // BackwardTransform(np, P, z, w, Jac[eln], coeff_e, quad_e);
        std::vector<double> quad_e = BackwardTransformLagrange(P, zquad, wquad, nq, J, coeff_e, z, np);

        for(int i = 0;i < np;i++)
        {
            R_DG[i+np*eln] = quad_e[i];
        }
    }
}




void CalculateRHS(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;
    
    int Mdim                        = (P+1)*Nel;
    double *F_DG                    = dvector(Nel*np);
    std::vector<double> quad_e(np,0.0);
    ipiv                            = ivector(Mdim);
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
            //std::cout << "U_DG[i + eln*np] " << U_DG[i + eln*np] << std::endl;
        }
    }
    // Transform fluxes forward into coefficient space.
    //==========================================================
    double *Fwd = dvector(2);
    double *Bwd = dvector(2);
    std::vector<double> FtLn(2,0);
    std::vector<double> FtRn(2,0);

    std::vector<double> alphaL(2,0);
    std::vector<double> alphaR(2,0);
    int nL = 1;
    int nR = -1;
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];
        for(int i = 0;i < np;i++)
        {
            quad_e[i] = F_DG[i+eln*np];
        }

        std::vector<double> coeff_new(P+1,0.0);

        InnerProductWRTDerivLagrangeBasis(np, P, J, wquad, zquad, z, D, quad_e, coeff_new);

        GetFwdBwd(eln, Nel, np, bc, U_DG, Fwd, Bwd);

        double FluxFwd0         = Fwd[0]*Fwd[0]*0.5;
        double FluxFwd1         = Fwd[1]*Fwd[1]*0.5;

        double FluxBwd0         = Bwd[0]*Bwd[0]*0.5;
        double FluxBwd1         = Bwd[1]*Bwd[1]*0.5;

        FtLn[0]                 = nL*(FluxFwd0*FluxFwd0)*0.5;
        FtRn[0]                 = nR*(FluxBwd0*FluxBwd0)*0.5;

        FtLn[1]                 = nL*(FluxFwd1*FluxFwd1)*0.5;
        FtRn[1]                 = nR*(FluxBwd1*FluxBwd1)*0.5;
        
        alphaL[0]               = 2.0*(FluxFwd0);
        alphaR[0]               = 2.0*(FluxBwd0);

        alphaL[1]               = 2.0*(FluxFwd1);
        alphaR[1]               = 2.0*(FluxBwd1);
        //Lax Friedrichs
        double num_fluxLeft     = 0.5*(FtLn[0]+FtRn[0])
                                -0.5*max(fabs(alphaL[0]),fabs(alphaR[0]))*(Bwd[0]-Fwd[0]);

        double num_fluxRight    = 0.5*(FtLn[1]+FtRn[1])
                                -0.5*max(fabs(alphaL[1]),fabs(alphaR[1]))*(Bwd[1]-Fwd[1]);

        std::vector<double> coeff_e = ForwardTransformLagrange(P, zquad, wquad, nq, J, quad_e, np);
        int Pf = P-1;

        // std::vector<double> coeffs_filtered = FilterNodalCoeffs(zquad, wquad, z, np, nq, coeff_e, P, Pf, J);

        for(int i = 0;i < (P+1);i++)
        {
            // Fcoeff[i+eln*(P+1)] = coeff_e[i];
            Fcoeff[i+eln*(P+1)] = coeff_new[i];
            //std::cout << "coeff_new " << i << " " << coeff_new[i] << std::endl;
        }
        coeff_new.clear();
    }
    //==========================================================
    // Calculate the numerical flux;
    double *quads = dvector(np);
    
    double *n        = dvector(Nel*2);
    double *UtL      = dvector(Nel*2);
    double *UtR      = dvector(Nel*2);
    double *numfluxL = dvector(Nel*2);
    double *numfluxR = dvector(Nel*2);
    double *numf = dvector(Nel*2);
    
    negatednormals(Nel, n);
    int cnt = 0;
    
    double *alpha     = dvector(Nel+1);
    double *alphaLv2    = dvector(Nel+1);
    double *alphaRv2    = dvector(Nel+1);
    double *FtLnv2      = dvector(Nel+1);
    double *FtRnv2      = dvector(Nel+1);
    double *FLUX      = dvector(Nel+1);
    cnt = 0;



    std::vector<double> UL(Nel+1,0.0);
    std::vector<double> UR(Nel+1,0.0);

    GetAllFwdBwd(Nel,np,bc,U_DG, UL, UR);

    for(int i = 0;i < (Nel+1);i++)
    {
        //GetFwdBwd(i, Nel, np, bc, U_DG, Fwd, Bwd);
        //std::cout << "UtL[i]*UtL[i] " << UtL[i]*UtL[i] << std::endl;
        FtLnv2[i]      =  nL*(UL[i]*UL[i])*0.5;
        FtRnv2[i]      =  nR*(UR[i]*UR[i])*0.5;
        
        alphaLv2[i]    = 2.0*(UL[i]);
        alphaRv2[i]    = 2.0*(UR[i]);
        //Lax Friedrichs
        FLUX[i]      = 0.5*(FtLnv2[i]+FtRnv2[i])-0.5*max(fabs(alphaLv2[i]),fabs(alphaRv2[i]))*(UR[i]-UL[i]);
    }

    double *numcoeff = dvector(Mdim);
    cnt = 0;
    
    for(int i = 0;i<Mdim;i++)
    {
        numcoeff[i] = 0.0;
    }
    
    for(int i = 0;i < Nel;i++)
    {
        if(i == 0)
        {
            numcoeff[0] = -Jac[i]*FLUX[0];
            numcoeff[P] =  Jac[i]*FLUX[1];
        }
        else if(i == Nel-1)
        {
            numcoeff[(Nel-1)*(P+1)]   = -Jac[i]*FLUX[i];
            numcoeff[Nel*(P+1)-1    ] =  Jac[i]*FLUX[i+1];
        }
        else
        {
            numcoeff[i*(P+1)]     = -Jac[i]*FLUX[i];
            numcoeff[i*(P+1)+P]   =  Jac[i]*FLUX[i+1];
        }
    }

    //GetGlobalStiffnessMatrix(Nel, P, np, nq, zquad, wquad, z, D, Jac, map, Mdim, StiffnessMatGlobal);
    
    //dgemv_(&TRANS,&Mdim,&Mdim,&ONE_DOUBLE,StiffnessMatGlobal[0],&Mdim,Fcoeff,&ONE_INT,&ZERO_DOUBLE,tmp,&ONE_INT);
    
    for(int i = 0;i < Mdim; i++)
    {
        // Ucoeff[i] = -tmp[i]-numcoeff[i];
        Ucoeff[i] = -Fcoeff[i]+numcoeff[i];
    }
    
    GetGlobalMassMatrix(Nel, P, np, nq, zquad, wquad, z, Jac, map, Mdim, MassMatGlobal);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal[0], &Mdim, ipiv, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal[0], &Mdim, ipiv, Ucoeff, &Mdim, &INFO);
    
    // Transform back onto quadrature points.
    for(int eln=0;eln<Nel;eln++)
    {
        double J  = Jac[eln];

        for(int i = 0;i<(P+1);i++)
        {
            coeff_e[i] = Ucoeff[i+eln*(P+1)];
        }

        std::vector<double> quad_e = BackwardTransformLagrange(P, zquad, wquad, nq, J, coeff_e, z, np);

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
    
    for(int eln=0;eln<Nel;eln++)
    {
        // Determine elemental mass matrix;
        GetElementStiffnessMatrix(np, nq, P, zquad, wquad, z, D, Jac[eln], StiffnessElem);
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



void GetElementStiffnessMatrix(int np, int nq, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double J, double **StiffMatElem)
{
    for(int i=0;i<P+1;i++)
    {
        for(int j=0;j<P+1;j++){
            StiffMatElem[i][j] = 0;
        }
    }
    
    //double *phi1  = dvector(np);
    // double *dphi1 = dvector(np);
    // double *phi2  = dvector(np);
    double *dphi2 = dvector(np);

    //std::cout << "Stiffness Matrix " << std::endl;
    for(int i=0;i<P+1;i++)
    {
        std::vector<double> phi1 = getLagrangeBasisFunction(i,zquad,nq,z,np,P);
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = getLagrangeBasisFunction(j,zquad,nq,z,np,P);
            //lagrange_basis(np, P, j, z, phi2);
            diff( np, D, phi2.data(), dphi2, J);
            
            StiffMatElem[i][j] = J*integr(np, wquad.data(), phi1.data(), dphi2);

            //std::cout << StiffMatElem[i][j] << " ";
        }
        //std::cout << std::endl;
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

// This member function assembles the global mass matrix.
void GetGlobalMassMatrix(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad,  std::vector<double> z, double *Jac, int **map, int Mdim, double **MassMatGlobal)
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
        std::vector<double> MassMatElem = GetElementMassMatrix_Lagrange(P,zquad,np,zquad,wquad,nq,J);

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

