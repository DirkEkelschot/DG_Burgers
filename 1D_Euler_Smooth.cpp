#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/Polylib.h"
#include "src/basis_functions.h"
#include "src/basis.h"
#include "src/io.h"
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
#include "src/tinyxml.h"
#include <sstream>

using namespace std;
using namespace polylib;
std::vector<double> RoeRiemannVec(std::vector<double> Ul, std::vector<double> Ur, double normal);
std::vector<double> LaxFriedrichsRiemannVec(std::vector<double> Ul, std::vector<double> Ur, double normal);
void GetAllFwdBwdMapVec(int Nel, int np, std::vector<std::vector<double> > quad, std::map<int,std::vector<std::vector<double> > > &Umap, int nvar);
void CalculateRHS_Modal(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt);
void CalculateRHS(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis);
void CalculateRHSWeakFR(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP);
void CalculateRHSStrongFR(int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, double *bc, double *X_DG, double *U_DG, double *R_DG, double dt, std::vector<std::vector<double> > basis, std::vector<std::vector<double> > basisRadauM, std::vector<std::vector<double> > basisRadauP);
void CalculateRHSStrongFREuler(Basis* bkey, int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, std::vector<std::vector<double> > bc, std::vector<double> X_DG, std::vector<std::vector<double> > U_DG, std::vector<std::vector<double> > &R_DG, double dt, std::vector<std::vector<double> > basis);
void CalculateRHSWeakFREuler(Basis* bkey, int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, std::string btype, std::string ptype, double **D, double *Jac, int **map, std::vector<std::vector<double> > bc, std::vector<double> X_DG, std::vector<std::vector<double> > U_DG, std::vector<std::vector<double> > &R_DG, double dt, std::vector<std::vector<double> > basis);

void *negatednormals(int Nel, double *n);
int **iarray(int n,int m);
//std::vector<std::vector<double> > getNodalBasis(std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getModalBasis(std::vector<double> zquad, int nq, int np, inCalculateRHSStrongFREulert P);
//std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);

//std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
//std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
void GetAllFwdBwd(int Nel, int nq, double *bc, double *quad, std::vector<double> &UtL, std::vector<double> &UtR);
void GetAllFwdBwdMap(int Nel, int nq, double *quad, std::map<int,std::vector<double> > &Umap);
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
void GetElementStiffnessMatrixWeakNewBasis(int P, std::vector<double> wquad, double **D, double J, Basis* bkey, double **StiffMatElem);
void GetGlobalStiffnessMatrixWeakNewBasis(Basis* bkey, int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal);

void GetElementStiffnessMatrixNewBasis(int P, std::vector<double> wquad, double **D, double J, Basis* bkey, double **StiffMatElem);
void GetGlobalStiffnessMatrixNewBasis(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, Basis* bkey, double **StiffnessGlobal);


std::vector<double> modal_basis(int np, int P, int i, std::vector<double> z);
void GetGlobalStiffnessMatrix_Modal(int Nel, int P, int np, int nq, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, int Mdim, double **StiffnessGlobal);
void GetElementStiffnessMatrix_Modal(int np, int nq, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double J, double **StiffMatElem);
void GetGlobalMassMatrixNew(int Nel, int P, std::vector<double> wquad, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **MassMatGlobal);

void GetGlobalMassMatrixNewBasis(int Nel, int P, std::vector<double> wquad, double *Jac, int **map, int Mdim, Basis* bkey, double **MassMatGlobal);

std::vector<double> ForwardTransform(int P, int np, std::vector<std::vector<double> > basis,  std::vector<double>wquad, int nq, double J, std::vector<double> input_quad);


extern "C" {extern void dgetrf_(int *, int *, double (*), int *, int [], int*);}
extern "C" {extern void dgetrs_(unsigned char *, int *, int *, double (*), int *, int [], double [], int *, int *);}
extern "C" {extern void dgemv_(unsigned char *, int *, int *, double *, double (*), int *, double [], int *, double *, double [], int *);}
extern "C" {extern void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);}


void printProgressBar(int current, int total, int width = 50) {
    float progress = static_cast<float>(current) / total;
    int barWidth = static_cast<int>(width * progress);

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < barWidth) std::cout << "=";
        else if (i == barWidth) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%\r";
    std::cout.flush();
}

// void *lagrange_basis(int np, int P, int i, double *z, double *zwgll, double *phi)
// {
//     for (int q = 0; q < np; ++q)
//     {
//         phi[q] = hglj(i, z[q], zwgll, P, 0.0, 0.0);
//     }

//     return 0;
// }


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





/*
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
*/





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









// std::vector<std::vector<double> > getNodalBasis(std::vector<double> zquad, int nq, int np, int P)
// {

//     if(nq != zquad.size())
//     {
//         std::cout << "error: nq != zquad.size() " << std::endl;
//     }
//     int numModes = P + 1;


//     std::vector<std::vector<double> > basis;
    
//     for(int n=0;n<numModes;n++)
//     {
//         std::vector<double> phi1(nq);
//         for (int q = 0; q < nq; ++q)
//         {
//             phi1[q] = hglj(n, zquad[q], zquad.data(), numModes, 0.0, 0.0);
//         }
//         basis.push_back(phi1);
//     }
//     return basis;
// }



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

bool readRestart(const std::string& filename, 
                        std::vector<double>& col1, 
                        std::vector<double>& col2, 
                        std::vector<double>& col3, 
                        std::vector<double>& col4) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    double val1, val2, val3, val4;

    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        if (iss >> val1 >> val2 >> val3 >> val4) {
            col1.push_back(val1);
            col2.push_back(val2);
            col3.push_back(val3);
            col4.push_back(val4);
        }
    }

    inFile.close();
    return true;
}



int main(int argc, char* argv[])
{
    Inputs* inputs = ReadXmlFile("inputs.xml");

    int P             = inputs->porder;
    int Nel           = inputs->nelem;
    double dt         = inputs->dt;
    int nt            = inputs->nt;
    std::string btype = inputs->btype;
    std::string ptype = inputs->ptype;
    int nquad         = inputs->nquad;
    int restart       = inputs->restart;
    int modal = 1;
    int Q = P;
    int np = P + 1;
    int nq = 2*Q + 1;
    
    // int nt = 1500;
    // double dt = 0.001;

    double** D;//          = dmatrix(np);
    double** Dt;//         = dmatrix(np);

    std::vector<double> z;
    std::vector<double> w;
    std::vector<double> zq;
    std::vector<double> wq;

    std::vector<std::vector<double> > basis_m;
    std::vector<std::vector<double> > basis_rm;
    std::vector<std::vector<double> > basis_rp;
    double* bound           = dvector(Nel+1);
    double* Jac             = dvector(Nel);
    elbound(Nel, bound, 0.0, 1.0);
    
    np = P + 1;
    nq = nquad;

    z.resize(np,0.0);
    w.resize(np,0.0);
    zq.resize(nq,0.0);
    wq.resize(nq,0.0); 
    D  = dmatrix(nq);
    Dt = dmatrix(nq);
    
    Basis* bkey = new Basis(inputs->ptype,
                                 inputs->btype,
                                 zq,wq,P,nq);

    run_new_basis_test(zq,wq,nq,P);

    //=====================================================================

    
    double time = 0.0;
    int timeScheme = 0;
    // timeScheme = 0 -> Forward Euler
    // timeScheme = 1 -> Runge Kutta 4
    if(inputs->timescheme.compare("RK4") == 0)
    {
        timeScheme = 1;
    }
    


    std::vector<double> X_DG_e(Nel*nq,0.0);
    std::vector<std::vector<double> > U_DG(3);
    std::vector<double> U_DG_row0(Nel*nq,0.0);
    std::vector<double> U_DG_row1(Nel*nq,0.0);
    std::vector<double> U_DG_row2(Nel*nq,0.0);

    int** map  = imatrix(Nel, P+1);
    mapping(Nel, P, map);
    
    double* x                   = dvector(nq);
    double gammaMone = 1.4 - 1.0;
    double pL   = 1.0;
    double uL   = 0.0;
    double rhoL = 1.0;

    double pR   = 0.1;
    double uR   = 0.0;
    double rhoR = 0.125;
    if(restart==0)
    {


        for(int eln=0;eln<Nel;eln++)
        {
            // Determine the coordinates in each element x.
            chi(nq, eln, x, zq.data(), Jac, bound);

            for(int i=0;i<nq;i++)
            {
                X_DG_e[eln*nq+i] = x[i];
                U_DG_row0[eln*nq+i]    = 1.0+0.2*sin(2.0*M_PI*x[i]);
                U_DG_row1[eln*nq+i]    = 1.0;
                U_DG_row2[eln*nq+i]    = 1.0;
            }
                
            //chi(np, eln, x2, zq.data(), Jac, bound);
            
            // Places the element coordinates x into the right place in
            // the global coordinate vector.
            // for(int i=0;i<nq;i++)
            // {
            //     X_DG_e[eln*nq+i] = x[i];

            //     if(x[i]<0.5)
            //     {
            //         double pressure        = pL;
            //         U_DG_row0[eln*nq+i]    = rhoL;
            //         U_DG_row1[eln*nq+i]    = rhoL*uL;
            //         U_DG_row2[eln*nq+i]    = (pL/gammaMone);

            //     }
            //     else if(x[i]>0.5)
            //     {
            //         double pressure        = pR;
            //         U_DG_row0[eln*nq+i]    = rhoR;
            //         U_DG_row1[eln*nq+i]    = rhoR*uR;
            //         U_DG_row2[eln*nq+i]    = (pR/gammaMone);
            //     }
            // }
        }
    }
    if(restart==1)
    {   
        readRestart("dgdataEuler.out",X_DG_e,U_DG_row0,U_DG_row1,U_DG_row2);
    }


    // Run test:

    Basis* bModalkey = new Basis(inputs->ptype,
                                 "Modal",
                                 zq,wq,P,nq);

    std::vector<std::vector<double> > basis_test = bModalkey->GetB();

    for(int el = 0;el < Nel;el++)
    {   
        double J = Jac[el];
        std::vector<double> quad_eq0(nq);
        for(int q = 0; q < nq;q++)
        {
            quad_eq0[q] = U_DG_row0[el*nq+q];
        }
        std::vector<double> coeff_eq0     = ForwardTransform(P, basis_test, wq, nq, J, quad_eq0);
        std::vector<double> quad_e0_recov = BackwardTransform(P,  nq,  basis_test,  coeff_eq0);

        for(int q = 0; q < nq;q++)
        {
            std::cout << "test 1 " << quad_eq0[q] << " " << quad_e0_recov[q] << " " << J << std::endl;
        }
    
    }

    std::vector<std::vector<double> > basis_plot = getModalBasisEval(zq, zq, nq, P, inputs->ptype);

    for(int el = 0;el < Nel;el++)
    {   
        double J = Jac[el];
        std::vector<double> quad_eq0(nq);
        for(int q = 0; q < nq;q++)
        {
            quad_eq0[q] = U_DG_row0[el*nq+q];
        }
        std::vector<double> coeff_eq0     = ForwardTransform(P, basis_plot, wq, nq, J, quad_eq0);
        std::vector<double> quad_e0_recov = BackwardTransform(P,  nq,  basis_plot,  coeff_eq0);

        for(int q = 0; q < nq;q++)
        {
            std::cout << "test 2 " << quad_eq0[q] << " " << quad_e0_recov[q] << " " << J << std::endl;
        }
    
    }
    

    U_DG[0] = U_DG_row0;
    U_DG[1] = U_DG_row1;
    U_DG[2] = U_DG_row2;


    std::vector<std::vector<double> > U_DG_prev(3);
    std::vector<double> U_DG_row0_prev(Nel*nq,0.0);
    std::vector<double> U_DG_row1_prev(Nel*nq,0.0);
    std::vector<double> U_DG_row2_prev(Nel*nq,0.0);
    U_DG_prev[0] = U_DG_row0_prev;
    U_DG_prev[1] = U_DG_row1_prev;
    U_DG_prev[2] = U_DG_row2_prev;

    ofstream sol_e;
    sol_e.open("dgdataEuler.in");
    for(int i = 0;i < (Nel*nq);i++)
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


    std::vector<double> X_DG_test(nq,0.0);
    std::vector<double> U_DG_test(nq,0.0);


    


    std::vector<std::vector<double> > k1_input(3);
    std::vector<std::vector<double> > k2_input(3);
    std::vector<std::vector<double> > k3_input(3);
    std::vector<std::vector<double> > k4_input(3);

    std::vector<std::vector<double> > k1(3);
    std::vector<std::vector<double> > k2(3);
    std::vector<std::vector<double> > k3(3);
    std::vector<std::vector<double> > k4(3);

    std::vector<std::vector<double> > R_DG1(3);
    std::vector<std::vector<double> > R_DG2(3);
    std::vector<std::vector<double> > R_DG3(3);
    std::vector<std::vector<double> > R_DG4(3);


    for(int i=0;i<3;i++)
    {
        std::vector<double> k1_row(Nel*nq,0.0);
        k1_input[i] = k1_row;
        std::vector<double> k1_r(Nel*nq,0.0);
        k1[i] = k1_r;

        std::vector<double> k2_row(Nel*nq,0.0);
        k2_input[i] = k2_row;
        std::vector<double> k2_r(Nel*nq,0.0);
        k2[i] = k2_r;

        std::vector<double> k3_row(Nel*nq,0.0);
        k3_input[i] = k3_row;
        std::vector<double> k3_r(Nel*nq,0.0);
        k3[i] = k3_r;

        std::vector<double> k4_row(Nel*nq,0.0);
        k4_input[i] = k4_row;
        std::vector<double> k4_r(Nel*nq,0.0);
        k4[i] = k4_r;
    }



    int it = 0;

    int nanfound = 0;
    for(int t = 0; t < nt; t++)
    {
        
         for(int i = 0;i < (Nel*nq);i++)
        {

            U_DG_prev[0][i] = U_DG[0][i];
            U_DG_prev[1][i] = U_DG[1][i];
            U_DG_prev[2][i] = U_DG[2][i];
        }
        //================================================================================
        //Forward Euler time integration
        if(timeScheme==0)
        {
            //CalculateRHSStrongFREuler(bModalkey, np, nq, Nel, P, zq, wq, zq, D, Jac, map, bc_e, X_DG_e, U_DG, R_DG0, dt, basis_m);
            CalculateRHSWeakFREuler(bkey, np, nq, Nel, P, zq, wq, zq, inputs->btype, inputs->ptype, D, Jac, map, bc_e, X_DG_e, U_DG, R_DG0, dt, basis_m);

            for(int i=0;i<(Nel*nq);i++)
            {
                //std::cout << "R_DG0[0][i] " << R_DG0[0][i] << " " << R_DG0[1][i] << " " << R_DG0[2][i] << std::endl; 
                U_DG[0][i]  = U_DG[0][i]+dt*R_DG0[0][i];
                U_DG[1][i]  = U_DG[1][i]+dt*R_DG0[1][i];
                U_DG[2][i]  = U_DG[2][i]+dt*R_DG0[2][i];

                k1[0][i] = dt*R_DG0[0][i];
                k1[1][i] = dt*R_DG0[1][i];
                k1[2][i] = dt*R_DG0[2][i];

                 if(std::isnan(U_DG[0][i]) || std::isnan(U_DG[1][i]) || std::isnan(U_DG[2][i]))
                {
                    //std::cout << "NaN " <<std::endl;
                    nanfound = 1;
                    t = nt;
                }

            }
        }

        if(timeScheme==1)
        {
            //CalculateRHSStrongFREuler(np, nq, Nel, P, zq, wq, zq, D, Jac, map, bc_e, X_DG_e, U_DG, R_DG0, dt, basis_m);
            CalculateRHSWeakFREuler(bkey, np, nq, Nel, P, zq, wq, zq, inputs->btype, inputs->ptype, D, Jac, map, bc_e, X_DG_e, U_DG, R_DG0, dt, basis_m);
            for(int i=0;i<(Nel*nq);i++)
            {
                // std::cout << "R_DG0[0][i] " << R_DG0[0][i] << " " << R_DG0[1][i] << " " << R_DG0[2][i] << std::endl; 
                k1_input[0][i]  = U_DG[0][i]+dt*R_DG0[0][i];
                k1_input[1][i]  = U_DG[1][i]+dt*R_DG0[1][i];
                k1_input[2][i]  = U_DG[2][i]+dt*R_DG0[2][i];


                k1[0][i] = dt*R_DG0[0][i];
                k1[1][i] = dt*R_DG0[1][i];
                k1[2][i] = dt*R_DG0[2][i];

                 if(std::isnan(U_DG[0][i]) || std::isnan(U_DG[1][i]) || std::isnan(U_DG[2][i]))
                {
                    //std::cout << "NaN " <<std::endl;
                    t = nt;
                }
            }

            //CalculateRHSStrongFREuler(np, nq, Nel, P, zq, wq, zq, D, Jac, map, bc_e, X_DG_e, k1_input, R_DG1, dt, basis_m);
            CalculateRHSWeakFREuler(bkey, np, nq, Nel, P, zq, wq, zq, inputs->btype, inputs->ptype, D, Jac, map, bc_e, X_DG_e, k1_input, R_DG1, dt, basis_m);
            for(int i=0;i<(Nel*nq);i++)
            {
                // std::cout << "R_DG0[0][i] " << R_DG0[0][i] << " " << R_DG0[1][i] << " " << R_DG0[2][i] << std::endl; 
                k2_input[0][i]  = U_DG[0][i]+dt*R_DG1[0][i];
                k2_input[1][i]  = U_DG[1][i]+dt*R_DG1[1][i];
                k2_input[2][i]  = U_DG[2][i]+dt*R_DG1[2][i];


                k2[0][i] = dt*R_DG1[0][i];
                k2[1][i] = dt*R_DG1[1][i];
                k2[2][i] = dt*R_DG1[2][i];

                 if(std::isnan(U_DG[0][i]) || std::isnan(U_DG[1][i]) || std::isnan(U_DG[2][i]))
                {
                    //std::cout << "NaN " <<std::endl;
                    t = nt;
                }
            }

            //CalculateRHSStrongFREuler(np, nq, Nel, P, zq, wq, zq, D, Jac, map, bc_e, X_DG_e, k2_input, R_DG2, dt, basis_m);
            CalculateRHSWeakFREuler(bkey, np, nq, Nel, P, zq, wq, zq, inputs->btype, inputs->ptype, D, Jac, map, bc_e, X_DG_e, k2_input, R_DG2, dt, basis_m);

            for(int i=0;i<(Nel*nq);i++)
            {
                // std::cout << "R_DG0[0][i] " << R_DG0[0][i] << " " << R_DG0[1][i] << " " << R_DG0[2][i] << std::endl; 
                k3_input[0][i]  = U_DG[0][i]+dt*R_DG2[0][i];
                k3_input[1][i]  = U_DG[1][i]+dt*R_DG2[1][i];
                k3_input[2][i]  = U_DG[2][i]+dt*R_DG2[2][i];


                k3[0][i] = dt*R_DG2[0][i];
                k3[1][i] = dt*R_DG2[1][i];
                k3[2][i] = dt*R_DG2[2][i];

                 if(std::isnan(U_DG[0][i]) || std::isnan(U_DG[1][i]) || std::isnan(U_DG[2][i]))
                {
                    //std::cout << "NaN " <<std::endl;
                    t = nt;
                }
            }

            //CalculateRHSStrongFREuler(np, nq, Nel, P, zq, wq, zq, D, Jac, map, bc_e, X_DG_e, k3_input, R_DG3, dt, basis_m);
            CalculateRHSWeakFREuler(bkey, np, nq, Nel, P, zq, wq, zq, inputs->btype, inputs->ptype, D, Jac, map, bc_e, X_DG_e, k3_input, R_DG3, dt, basis_m);


            for(int i=0;i<(Nel*nq);i++)
            {
                // std::cout << "R_DG0[0][i] " << R_DG0[0][i] << " " << R_DG0[1][i] << " " << R_DG0[2][i] << std::endl; 
                k4[0][i]  = dt*R_DG3[0][i];
                k4[1][i]  = dt*R_DG3[1][i];
                k4[2][i]  = dt*R_DG3[2][i];

                U_DG[0][i] = U_DG[0][i]+1.0/6.0*(k1[0][i]+2.0*k2[0][i]+2.0*k3[0][i]+k4[0][i]);
                U_DG[1][i] = U_DG[1][i]+1.0/6.0*(k1[1][i]+2.0*k2[1][i]+2.0*k3[1][i]+k4[1][i]);
                U_DG[2][i] = U_DG[2][i]+1.0/6.0*(k1[2][i]+2.0*k2[2][i]+2.0*k3[2][i]+k4[2][i]);

                if(std::isnan(U_DG[0][i]) || std::isnan(U_DG[1][i]) || std::isnan(U_DG[2][i]))
                {
                    //std::cout << "NaN " <<std::endl;
                    nanfound = 1;
                    t = nt;
                }
            }


            
        
        }

        printProgressBar(t, nt);
       
         
        //std::cout << "time = " << time << " iteration = "<< it <<  std::endl;
        
        time = time + dt; 
        it = it + 1;      
    }

    

    std::cout << std::endl;
    std::cout << "time final = " << time << " iteration = "<< it <<  std::endl;
    ofstream solout;
    solout.open("dgdataEuler.out");

    if(nanfound == 1)
    {
        std::cout << "NaNs found " << std::endl;
        for(int i = 0;i < (Nel*nq);i++)
        {
                solout << X_DG_e[i] << " " << U_DG_prev[0][i] << " " << U_DG_prev[1][i] << " " << U_DG_prev[2][i] << endl;
        }
        
    }

    if(nanfound == 0)
    {
        for(int i = 0;i < (Nel*nq);i++)
        {
                solout << X_DG_e[i] << " " << U_DG[0][i] << " " << U_DG[1][i] << " " << U_DG[2][i] << endl;
        }
        
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








void CalculateRHSWeakFREuler(Basis* bkey, int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, std::string btype, std::string ptype, double **D, double *Jac, int **map, std::vector<std::vector<double> > bc, std::vector<double> X_DG, std::vector<std::vector<double> > U_DG, std::vector<std::vector<double> > &R_DG, double dt, std::vector<std::vector<double> > basis)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;

    // np = basis[0].size();
    // std::cout << " nq = " << nq << " np = " << np << std::endl;

    int Mdim                        = (P+1)*Nel;
    std::vector<std::vector<double> > F_DG(3);
    ipiv                            = ivector(Mdim);

    std::vector<double> F_DG_row0(Nel*nq,0.0);
    std::vector<double> F_DG_row1(Nel*nq,0.0);
    std::vector<double> F_DG_row2(Nel*nq,0.0);


    std::vector<std::vector<double> > Dmat = bkey->GetD();
    std::vector<std::vector<double> > Bmat = bkey->GetB();


    double gammaMone = 1.4-1.0;
    for(int eln=0;eln<Nel;eln++)
    {
        for(int i=0;i<nq;i++)
        {

            double rho  = U_DG[0][i + eln*nq];
            double rhou = U_DG[1][i + eln*nq];
            double E    = U_DG[2][i + eln*nq];
            double u    = rhou/rho;
            double p    = (E-0.5*rho*u*u)*gammaMone;
            // Evaluate the flux at each quadrature point.
            F_DG_row0[i + eln*nq] = rho*u;
            F_DG_row1[i + eln*nq] = p+rho*u*u;
            F_DG_row2[i + eln*nq] = (E+p)*u;

        }
    }

    F_DG[0] = F_DG_row0;
    F_DG[1] = F_DG_row1;
    F_DG[2] = F_DG_row2;
    // Transform fluxes forward into coefficient space.
    //==========================================================
    
    std::vector<std::vector<double> > F_DG_coeff(3);
    std::vector<double> quad_eq0(nq,0.0);
    std::vector<double> quad_eq1(nq,0.0);
    std::vector<double> quad_eq2(nq,0.0);
    std::vector<double> dx(Nel);

    std::vector<double> Fcoeff_eq0(Nel*(P+1),0.0);
    std::vector<double> Fcoeff_eq1(Nel*(P+1),0.0);
    std::vector<double> Fcoeff_eq2(Nel*(P+1),0.0);

    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        double xstart = X_DG[eln*nq];
        double xend   = X_DG[eln*nq+nq-1];

        dx[eln] = xend-xstart; 
        //std::cout << eln << " " << dx[eln] << std::endl;
        for(int i = 0;i < nq;i++)
        {
            quad_eq0[i] = F_DG[0][i+eln*nq];
            quad_eq1[i] = F_DG[1][i+eln*nq];
            quad_eq2[i] = F_DG[2][i+eln*nq];
        }

        std::vector<double> coeff_eq0 = ForwardTransform(P, Bmat, wquad, nq, J, quad_eq0);
        std::vector<double> coeff_eq1 = ForwardTransform(P, Bmat, wquad, nq, J, quad_eq1);
        std::vector<double> coeff_eq2 = ForwardTransform(P, Bmat, wquad, nq, J, quad_eq2);

        // std::vector<double> quad_e0_recov = BackwardTransform(P,  nq,  Bmat,  coeff_eq0);
        // std::vector<double> quad_e1_recov = BackwardTransform(P,  nq,  Bmat,  coeff_eq1);
        // std::vector<double> quad_e2_recov = BackwardTransform(P,  nq,  Bmat,  coeff_eq2);

        // for(int i = 0;i < nq;i++)
        // {
        //     std::cout << quad_eq0[i] << " -- " << quad_e0_recov[i]/J << std::endl;
        //     std::cout << quad_eq1[i] << " -- " << quad_e1_recov[i]/J << std::endl;
        //     std::cout << quad_eq2[i] << " -- " << quad_e2_recov[i]/J << std::endl;
        // }

        // if(btype == "Nodal")
        // {
        //     double value_left  = EvaluateFromNodalBasis(P, -1, coeff_eq0, zquad, ptype);
        //     double value_right = EvaluateFromNodalBasis(P,  1, coeff_eq0, zquad, ptype);

        //     double value_left_int  = EvaluateFromNodalBasis(P, zquad[0], coeff_eq0, zquad, ptype);
        //     double value_right_int = EvaluateFromNodalBasis(P,  zquad[nq-1], coeff_eq0, zquad, ptype);

        //     std::cout << -1 << " " << zquad[0] << " -- " <<  zquad[nq-1] << " " << 1 << std::endl;
        //     std::cout << J << std::endl;
        //     std::cout << eln << " coeff_eq0 " << coeff_eq0.size() << " (" << value_left << " - " << value_right << ") (" << value_left_int << " - " << value_right_int << ")" << std::endl;
        // }

        // if(btype == "Modal")
        // {
        //     double value_left  = EvaluateFromModalBasis(P, -1, nq, coeff_eq0, ptype);
        //     double value_right = EvaluateFromModalBasis(P,  1, nq, coeff_eq0, ptype);

        //     double value_left_int  = EvaluateFromModalBasis(P, zquad[0], nq, coeff_eq0, ptype);
        //     double value_right_int = EvaluateFromModalBasis(P, zquad[nq-1], nq, coeff_eq0, ptype);

        //     std::cout << -1 << " " << zquad[0] << " -- " <<  zquad[nq-1] << " " << 1 << std::endl;
        //     std::cout << eln << " coeff_eq0 " << coeff_eq0.size() << " (" << value_left << " - " << value_right << ") (" << value_left_int << " - " << value_right_int << ")" << std::endl;
        // }
        
        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff_eq0[i+eln*(P+1)] = coeff_eq0[i];
            Fcoeff_eq1[i+eln*(P+1)] = coeff_eq1[i];
            Fcoeff_eq2[i+eln*(P+1)] = coeff_eq2[i];

            //std::cout << Fcoeff_eq0[i+eln*(P+1)]  << " " << Fcoeff_eq1[i+eln*(P+1)]  << " " << Fcoeff_eq2[i+eln*(P+1)]  << std::endl;
        }
        //std::cout << std::endl;
    }

    F_DG_coeff[0] = Fcoeff_eq0;
    F_DG_coeff[1] = Fcoeff_eq1;
    F_DG_coeff[2] = Fcoeff_eq2;
   

    std::map<int,std::vector<std::vector<double> > > Umap_v;
    GetAllFwdBwdMapVec(Nel, nq, U_DG, Umap_v, 3);
    
    int normalL =  -1;
    int normalR =   1;

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

        // for(int n=0;n<nvar;n++)
        // {
            if(elid == 0)
            {
                uLFwd[0] = Umap_v[elid][0][0];
                uLFwd[1] = Umap_v[elid][1][0];
                uLFwd[2] = Umap_v[elid][2][0];

                
                double rhoFwd  =  uLFwd[0];
                double rhouFwd =  uLFwd[1];
                double EFwd    =  uLFwd[2];
                double pFwd    =  (EFwd-0.5*rhouFwd*rhouFwd/rhoFwd)*gammaMone;

                double rhoBwd  =  Umap_v[Nel-1][0][1];
                double rhouBwd =  Umap_v[Nel-1][1][1];//-rhouFwd;
                double EBwd    =  Umap_v[Nel-1][2][1];

                // std::cout << "bcs 0 left " << bc[0][0] << " " << rhoFwd << " " << rhoBwd << std::endl; 
                // std::cout << "bcs 1 left " << bc[0][1] << " " << rhouFwd << " " << rhouBwd << std::endl; 
                // std::cout << "bcs 2 left " << bc[0][2] << " " << EFwd << " " << EBwd << std::endl; 

                uLBwd[0] =  rhoBwd;
                uLBwd[1] =  rhouBwd;
                uLBwd[2] =  EBwd;
                
                uRFwd[0] = Umap_v[elid][0][1];
                uRFwd[1] = Umap_v[elid][1][1];
                uRFwd[2] = Umap_v[elid][2][1];

                uRBwd[0] = Umap_v[elid+1][0][0];  
                uRBwd[1] = Umap_v[elid+1][1][0];  
                uRBwd[2] = Umap_v[elid+1][2][0];  

            }
            else if(elid == Nel-1)
            {
                uLFwd[0] = Umap_v[elid][0][0];
                uLFwd[1] = Umap_v[elid][1][0];
                uLFwd[2] = Umap_v[elid][2][0];

                uLBwd[0] = Umap_v[elid-1][0][1];
                uLBwd[1] = Umap_v[elid-1][1][1];
                uLBwd[2] = Umap_v[elid-1][2][1];

                uRFwd[0] = Umap_v[elid][0][1];
                uRFwd[1] = Umap_v[elid][1][1];
                uRFwd[2] = Umap_v[elid][2][1];

                //uRBwd[n] = Umap_v[0][n][0];
                // uRBwd[n] = 2.0*bc[1][n]-uRFwd[n];

                double rhoFwd  =  uRFwd[0];
                double rhouFwd =  uRFwd[1];
                double EFwd    =  uRFwd[2];
                double pFwd    =  (EFwd-0.5*rhouFwd*rhouFwd/rhoFwd)*gammaMone;

                double rhoBwd  =   Umap_v[0][0][0];
                double rhouBwd =   Umap_v[0][1][0];//-rhouFwd;
                double EBwd    =   Umap_v[0][2][0];

                // std::cout << "bcs 0 right " << bc[1][0] << " " << rhoFwd << " " << rhoBwd << std::endl; 
                // std::cout << "bcs 1 right " << bc[1][1] << " " << rhouFwd << " " << rhouBwd << std::endl; 
                // std::cout << "bcs 2 right " << bc[1][2] << " " << EFwd << " " << EBwd << std::endl; 

                uRBwd[0] = rhoBwd; 
                uRBwd[1] = rhouBwd;
                uRBwd[2] = EBwd;
            }
            else
            {
                uLFwd[0] = Umap_v[elid][0][0];
                uLFwd[1] = Umap_v[elid][1][0];
                uLFwd[2] = Umap_v[elid][2][0];
                
                uLBwd[0] = Umap_v[elid-1][0][1];
                uLBwd[1] = Umap_v[elid-1][1][1];
                uLBwd[2] = Umap_v[elid-1][2][1];

                uRFwd[0] = Umap_v[elid][0][1];
                uRFwd[1] = Umap_v[elid][1][1];
                uRFwd[2] = Umap_v[elid][2][1];
                
                uRBwd[0] = Umap_v[elid+1][0][0];
                uRBwd[1] = Umap_v[elid+1][1][0];
                uRBwd[2] = Umap_v[elid+1][2][0];
            }
       // }

        
        double rhoLFwd  = uLFwd[0];
        double rhouLFwd = uLFwd[1];
        double ELFwd    = uLFwd[2];
        double uLFwd_s    = rhouLFwd/rhoLFwd;
        double pLFwd    = (ELFwd-0.5*rhoLFwd*uLFwd_s*uLFwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FLFwd[0] = rhouLFwd;
        FLFwd[1] = pLFwd+rhoLFwd*uLFwd_s*uLFwd_s;
        FLFwd[2] = (ELFwd+pLFwd)*uLFwd_s;

        double rhoRFwd  = uRFwd[0];
        double rhouRFwd = uRFwd[1];
        double ERFwd    = uRFwd[2];
        double uRFwd_s    = rhouRFwd/rhoRFwd;
        double pRFwd    = (ERFwd-0.5*rhoRFwd*uRFwd_s*uRFwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FRFwd[0] = rhouRFwd;
        FRFwd[1] = pRFwd+rhoRFwd*uRFwd_s*uRFwd_s;
        FRFwd[2] = (ERFwd+pRFwd)*uRFwd_s;


        double rhoLBwd  = uLBwd[0];
        double rhouLBwd = uLBwd[1];
        double ELBwd    = uLBwd[2];
        double uLBwd_s    = rhouLBwd/rhoLBwd;
        double pLBwd    = (ELBwd-0.5*rhoLBwd*uLBwd_s*uLBwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FLBwd[0] = rhouLBwd;
        FLBwd[1] = pLBwd+rhoLBwd*uLBwd_s*uLBwd_s;
        FLBwd[2] = (ELBwd+pLBwd)*uLBwd_s;

        
        double rhoRBwd  = uRBwd[0];
        double rhouRBwd = uRBwd[1];
        double ERBwd    = uRBwd[2];
        double uRBwd_s    = rhouRBwd/rhoRBwd;
        double pRBwd    = (ERBwd-0.5*rhoRBwd*uRBwd_s*uRBwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FRBwd[0] = rhouRBwd;
        FRBwd[1] = pRBwd+rhoRBwd*uRBwd_s*uRBwd_s;
        FRBwd[2] = (ERBwd+pRBwd)*uRBwd_s;
        

        //std::cout << uLBwd_s << " " << uRBwd_s <<  " "  << uLFwd_s << " " << uRFwd_s << std::endl;

        
        std::vector<double> Fl = LaxFriedrichsRiemannVec(uLBwd,uLFwd,1.0);
        std::vector<double> Fr = LaxFriedrichsRiemannVec(uRFwd,uRBwd,1.0);

        // std::vector<double> Fl = RoeRiemannVec(uLBwd,uLFwd,1.0);
        // std::vector<double> Fr = RoeRiemannVec(uRFwd,uRBwd,1.0);

        std::vector<double> Fleft(nvar,0.0);
        std::vector<double> Fright(nvar,0.0);

        for(int n=0;n<nvar;n++)
        {
            // Fleft[n]  = normalL*(-Fl[n]+FLBwd[n]);
            // Fright[n] = normalR*(-Fr[n]+FRBwd[n]);
            // Fleft[n]  = (Fl[n]);
            // Fright[n] = (Fr[n]);
            Fleft[n]  = (Fl[n]);
            Fright[n] = (Fr[n]);
            
            // std::cout << Fleft[n] << " " << Fright[n] << std::endl;

            // Fleft[n]  = ((FLFwd[n]+FLBwd[n])*0.5);
            // Fright[n] = ((FRFwd[n]+FRBwd[n])*0.5);

            // Fleft[n]  = (FLBwd[n]);
            // Fright[n] = (FRBwd[n]);
           
            // std::cout << "Fleft["<<n<<"] "  << Fl[n] << " " << Fr[n] << std::endl; 
        }


        //std::cout << Fl[1] << " " << Fr[1] << std::endl;
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
            numcoeff_eq0[P]                 =        Fmap_v[i][1][0];

            numcoeff_eq1[0]                 =       -Fmap_v[i][0][1];
            numcoeff_eq1[P]                 =        Fmap_v[i][1][1];

            numcoeff_eq2[0]                 =       -Fmap_v[i][0][2];
            numcoeff_eq2[P]                 =        Fmap_v[i][1][2];
        }
        else if(i == Nel-1)
        {

            numcoeff_eq0[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][0];
            numcoeff_eq0[Nel*(P+1)-1  ]     =        Fmap_v[i][1][0];

            numcoeff_eq1[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][1];
            numcoeff_eq1[Nel*(P+1)-1  ]     =        Fmap_v[i][1][1];

            numcoeff_eq2[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][2];
            numcoeff_eq2[Nel*(P+1)-1  ]     =        Fmap_v[i][1][2];


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
    // GetGlobalStiffnessMatrixNew(Nel, P, wquad, D, Jac, map, Mdim, basis, StiffnessMatGlobal);
    // GetGlobalStiffnessMatrixWeakNew(Nel, P, wquad, D, Jac, map, Mdim, basis_update, StiffnessMatGlobal);
    GetGlobalStiffnessMatrixWeakNewBasis(bkey, Nel, P, wquad, D, Jac, map, Mdim, Bmat, StiffnessMatGlobal);

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
        Ucoeff_eq0[i] = tmp0[i]-numcoeff_eq0[i];
        Ucoeff_eq1[i] = tmp1[i]-numcoeff_eq1[i];
        Ucoeff_eq2[i] = tmp2[i]-numcoeff_eq2[i];

        //std::cout << tmp0[i] << " " << tmp1[i] << " " << tmp2[i] << std::endl;
    }
    double **MassMatGlobal0          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, Bmat, MassMatGlobal0);
    // GetGlobalMassMatrixNewBasis(Nel, P, wquad, Jac, map, Mdim, bkey, MassMatGlobal0);

    double **MassMatGlobal1          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, Bmat, MassMatGlobal1);
    // GetGlobalMassMatrixNewBasis(Nel, P, wquad, Jac, map, Mdim, bkey, MassMatGlobal0);

    double **MassMatGlobal2          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, Bmat, MassMatGlobal2);
    // GetGlobalMassMatrixNewBasis(Nel, P, wquad, Jac, map, Mdim, bkey, MassMatGlobal0);


    dgetrf_(&Mdim, &Mdim, MassMatGlobal0[0], &Mdim, ipiv, &INFO);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal1[0], &Mdim, ipiv, &INFO);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal2[0], &Mdim, ipiv, &INFO);

    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal0[0], &Mdim, ipiv, Ucoeff_eq0.data(), &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal1[0], &Mdim, ipiv, Ucoeff_eq1.data(), &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal2[0], &Mdim, ipiv, Ucoeff_eq2.data(), &Mdim, &INFO);
    
    // Transform back onto quadrature points.
    std::vector<double> R_DG_row0(Nel*nq,0.0);
    std::vector<double> R_DG_row1(Nel*nq,0.0);
    std::vector<double> R_DG_row2(Nel*nq,0.0);

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
        
        std::vector<double> quad_e0 = BackwardTransform(P,  nq,  Bmat,  coeff_eq0_tmp);
        std::vector<double> quad_e1 = BackwardTransform(P,  nq,  Bmat,  coeff_eq1_tmp);
        std::vector<double> quad_e2 = BackwardTransform(P,  nq,  Bmat,  coeff_eq2_tmp);
        int Pf = P - 1;

        for(int i = 0;i < nq;i++)
        {
            R_DG_row0[i+nq*eln] = quad_e0[i];
            R_DG_row1[i+nq*eln] = quad_e1[i];
            R_DG_row2[i+nq*eln] = quad_e2[i];
        }


        R_DG[0] = R_DG_row0;
        R_DG[1] = R_DG_row1;
        R_DG[2] = R_DG_row2;

    }
    
}





void CalculateRHSStrongFREuler(Basis* bkey, int np, int nq, int Nel, int P, std::vector<double> zquad, std::vector<double> wquad, std::vector<double> z, double **D, double *Jac, int **map, std::vector<std::vector<double> > bc, std::vector<double> X_DG, std::vector<std::vector<double> > U_DG, std::vector<std::vector<double> > &R_DG, double dt, std::vector<std::vector<double> > basis)
{
    unsigned char TRANS = 'T';
    int NRHS=1,INFO,*ipiv,ONE_INT=1;
    double ZERO_DOUBLE=0.0,ONE_DOUBLE=1.0;

    // np = basis[0].size();
    //std::cout << " nq = " << nq << " np = " << np << std::endl;
    int Mdim                        = (P+1)*Nel;
    std::vector<std::vector<double> > F_DG(3);
    ipiv                            = ivector(Mdim);

    std::vector<double> F_DG_row0(Nel*nq,0.0);
    std::vector<double> F_DG_row1(Nel*nq,0.0);
    std::vector<double> F_DG_row2(Nel*nq,0.0);

    std::vector<std::vector<double> > Dmat = bkey->GetD();
    std::vector<std::vector<double> > Bmat = bkey->GetB();


    double gammaMone = 1.4-1.0;
    for(int eln=0;eln<Nel;eln++)
    {
        for(int i=0;i<nq;i++)
        {

            double rho  = U_DG[0][i + eln*nq];
            double rhou = U_DG[1][i + eln*nq];
            double E    = U_DG[2][i + eln*nq];
            double u    = rhou/rho;
            double p    = (E-0.5*rho*u*u)*gammaMone;
            // Evaluate the flux at each quadrature point.
            F_DG_row0[i + eln*nq] = rho*u;
            F_DG_row1[i + eln*nq] = p+rho*u*u;
            F_DG_row2[i + eln*nq] = (E+p)*u;

        }
    }

    F_DG[0] = F_DG_row0;
    F_DG[1] = F_DG_row1;
    F_DG[2] = F_DG_row2;
    // Transform fluxes forward into coefficient space.
    //==========================================================
    
    std::vector<std::vector<double> > F_DG_coeff(3);
    std::vector<double> quad_eq0(nq,0.0);
    std::vector<double> quad_eq1(nq,0.0);
    std::vector<double> quad_eq2(nq,0.0);
    std::vector<double> dx(Nel);

    std::vector<double> Fcoeff_eq0(Nel*(P+1),0.0);
    std::vector<double> Fcoeff_eq1(Nel*(P+1),0.0);
    std::vector<double> Fcoeff_eq2(Nel*(P+1),0.0);
    for(int eln=0;eln<Nel;eln++)
    {
        double J = Jac[eln];

        double xstart = X_DG[eln*nq];
        double xend   = X_DG[eln*nq+nq-1];

        dx[eln] = xend-xstart; 
        //std::cout << eln << " " << dx[eln] << std::endl;
        for(int i = 0;i < nq;i++)
        {
            quad_eq0[i] = F_DG[0][i+eln*nq];
            quad_eq1[i] = F_DG[1][i+eln*nq];
            quad_eq2[i] = F_DG[2][i+eln*nq];
        }

        
        std::vector<double> coeff_eq0 = ForwardTransform(P, Bmat, wquad, nq, J, quad_eq0);
        std::vector<double> coeff_eq1 = ForwardTransform(P, Bmat, wquad, nq, J, quad_eq1);
        std::vector<double> coeff_eq2 = ForwardTransform(P, Bmat, wquad, nq, J, quad_eq2);


        for(int i = 0;i < (P+1);i++)
        {
            Fcoeff_eq0[i+eln*(P+1)] = coeff_eq0[i];
            Fcoeff_eq1[i+eln*(P+1)] = coeff_eq1[i];
            Fcoeff_eq2[i+eln*(P+1)] = coeff_eq2[i];

            //std::cout << Fcoeff_eq0[i+eln*(P+1)]  << " " << Fcoeff_eq1[i+eln*(P+1)]  << " " << Fcoeff_eq2[i+eln*(P+1)]  << std::endl;
        }
        //std::cout << std::endl;
    }

    F_DG_coeff[0] = Fcoeff_eq0;
    F_DG_coeff[1] = Fcoeff_eq1;
    F_DG_coeff[2] = Fcoeff_eq2;
   

    std::map<int,std::vector<std::vector<double> > > Umap_v;
    GetAllFwdBwdMapVec(Nel, nq, U_DG, Umap_v, 3);
    
    int normalL =  -1;
    int normalR =   1;

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

        // for(int n=0;n<nvar;n++)
        // {
            if(elid == 0)
            {
                uLFwd[0] = Umap_v[elid][0][0];
                uLFwd[1] = Umap_v[elid][1][0];
                uLFwd[2] = Umap_v[elid][2][0];

                
                double rhoFwd  =  uLFwd[0];
                double rhouFwd =  uLFwd[1];
                double EFwd    =  uLFwd[2];
                double pFwd    =  (EFwd-0.5*rhouFwd*rhouFwd/rhoFwd)*gammaMone;

                double rhoBwd  =  2.0*bc[0][0]-rhoFwd;
                double rhouBwd =  2.0*bc[0][1]-rhouFwd;//-rhouFwd;
                double pBwd    =  (2.0*bc[0][2]*gammaMone)-pFwd;
                double EBwd    =  pBwd/(gammaMone)+0.5*rhouBwd*rhouBwd/rhoBwd;

                // std::cout << "bcs 0 left " << bc[0][0] << " " << rhoFwd << " " << rhoBwd << std::endl; 
                // std::cout << "bcs 1 left " << bc[0][1] << " " << rhouFwd << " " << rhouBwd << std::endl; 
                // std::cout << "bcs 2 left " << bc[0][2] << " " << EFwd << " " << EBwd << std::endl; 

                uLBwd[0] =  rhoBwd;
                uLBwd[1] =  rhouBwd;
                uLBwd[2] =  EBwd;
                
                uRFwd[0] = Umap_v[elid][0][1];
                uRFwd[1] = Umap_v[elid][1][1];
                uRFwd[2] = Umap_v[elid][2][1];

                uRBwd[0] = Umap_v[elid+1][0][0];  
                uRBwd[1] = Umap_v[elid+1][1][0];  
                uRBwd[2] = Umap_v[elid+1][2][0];  

            }
            else if(elid == Nel-1)
            {
                uLFwd[0] = Umap_v[elid][0][0];
                uLFwd[1] = Umap_v[elid][1][0];
                uLFwd[2] = Umap_v[elid][2][0];

                uLBwd[0] = Umap_v[elid-1][0][1];
                uLBwd[1] = Umap_v[elid-1][1][1];
                uLBwd[2] = Umap_v[elid-1][2][1];

                uRFwd[0] = Umap_v[elid][0][1];
                uRFwd[1] = Umap_v[elid][1][1];
                uRFwd[2] = Umap_v[elid][2][1];

                //uRBwd[n] = Umap_v[0][n][0];
                // uRBwd[n] = 2.0*bc[1][n]-uRFwd[n];

                double rhoFwd  =  uRFwd[0];
                double rhouFwd =  uRFwd[1];
                double EFwd    =  uRFwd[2];
                double pFwd    =  (EFwd-0.5*rhouFwd*rhouFwd/rhoFwd)*gammaMone;


                double rhoBwd  =   2.0*bc[1][0]-rhoFwd;
                double rhouBwd =   2.0*bc[1][1]-rhouFwd;//-rhouFwd;
                double pBwd    =   (2.0*bc[1][2]*gammaMone)-pFwd;
                double EBwd    =  pBwd/(gammaMone)+0.5*rhouBwd*rhouBwd/rhoBwd;

                // std::cout << "bcs 0 right " << bc[1][0] << " " << rhoFwd << " " << rhoBwd << std::endl; 
                // std::cout << "bcs 1 right " << bc[1][1] << " " << rhouFwd << " " << rhouBwd << std::endl; 
                // std::cout << "bcs 2 right " << bc[1][2] << " " << EFwd << " " << EBwd << std::endl; 

                uRBwd[0] = rhoBwd; 
                uRBwd[1] = rhouBwd;
                uRBwd[2] = EBwd;
            }
            else
            {
                uLFwd[0] = Umap_v[elid][0][0];
                uLFwd[1] = Umap_v[elid][1][0];
                uLFwd[2] = Umap_v[elid][2][0];
                
                uLBwd[0] = Umap_v[elid-1][0][1];
                uLBwd[1] = Umap_v[elid-1][1][1];
                uLBwd[2] = Umap_v[elid-1][2][1];

                uRFwd[0] = Umap_v[elid][0][1];
                uRFwd[1] = Umap_v[elid][1][1];
                uRFwd[2] = Umap_v[elid][2][1];
                
                uRBwd[0] = Umap_v[elid+1][0][0];
                uRBwd[1] = Umap_v[elid+1][1][0];
                uRBwd[2] = Umap_v[elid+1][2][0];
            }
       // }

        
        double rhoLFwd  = uLFwd[0];
        double rhouLFwd = uLFwd[1];
        double ELFwd    = uLFwd[2];
        double uLFwd_s    = rhouLFwd/rhoLFwd;
        double pLFwd    = (ELFwd-0.5*rhoLFwd*uLFwd_s*uLFwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FLFwd[0] = rhouLFwd;
        FLFwd[1] = pLFwd+rhoLFwd*uLFwd_s*uLFwd_s;
        FLFwd[2] = (ELFwd+pLFwd)*uLFwd_s;

        double rhoRFwd  = uRFwd[0];
        double rhouRFwd = uRFwd[1];
        double ERFwd    = uRFwd[2];
        double uRFwd_s  = rhouRFwd/rhoRFwd;
        double pRFwd    = (ERFwd-0.5*rhoRFwd*uRFwd_s*uRFwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FRFwd[0] = rhouRFwd;
        FRFwd[1] = pRFwd+rhoRFwd*uRFwd_s*uRFwd_s;
        FRFwd[2] = (ERFwd+pRFwd)*uRFwd_s;


        double rhoLBwd  = uLBwd[0];
        double rhouLBwd = uLBwd[1];
        double ELBwd    = uLBwd[2];
        double uLBwd_s  = rhouLBwd/rhoLBwd;
        double pLBwd    = (ELBwd-0.5*rhoLBwd*uLBwd_s*uLBwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FLBwd[0] = rhouLBwd;
        FLBwd[1] = pLBwd+rhoLBwd*uLBwd_s*uLBwd_s;
        FLBwd[2] = (ELBwd+pLBwd)*uLBwd_s;

        
        double rhoRBwd  = uRBwd[0];
        double rhouRBwd = uRBwd[1];
        double ERBwd    = uRBwd[2];
        double uRBwd_s  = rhouRBwd/rhoRBwd;
        double pRBwd    = (ERBwd-0.5*rhoRBwd*uRBwd_s*uRBwd_s)*gammaMone;
        // Evaluate the flux at each quadrature point.

        FRBwd[0] = rhouRBwd;
        FRBwd[1] = pRBwd+rhoRBwd*uRBwd_s*uRBwd_s;
        FRBwd[2] = (ERBwd+pRBwd)*uRBwd_s;
        

        //std::cout << uLBwd_s << " " << uRBwd_s <<  " "  << uLFwd_s << " " << uRFwd_s << std::endl;

        
        std::vector<double> Fl = LaxFriedrichsRiemannVec(uLBwd,uLFwd,1.0);
        std::vector<double> Fr = LaxFriedrichsRiemannVec(uRFwd,uRBwd,1.0);

        // std::vector<double> Fl = RoeRiemannVec(uLBwd,uLFwd,1.0);
        // std::vector<double> Fr = RoeRiemannVec(uRFwd,uRBwd,1.0);

        std::vector<double> Fleft(nvar,0.0);
        std::vector<double> Fright(nvar,0.0);

        for(int n=0;n<nvar;n++)
        {
            // Fleft[n]  = normalL*(-Fl[n]+FLBwd[n]);
            // Fright[n] = normalR*(-Fr[n]+FRBwd[n]);
            Fleft[n]  = (-Fl[n]+FLFwd[n]);
            Fright[n] = (-Fr[n]+FRFwd[n]);

            // std::cout << Fleft[n] << " " << Fright[n] << std::endl;

            // Fleft[n]  = ((FLFwd[n]+FLBwd[n])*0.5);
            // Fright[n] = ((FRFwd[n]+FRBwd[n])*0.5);

            // Fleft[n]  = (FLBwd[n]);
            // Fright[n] = (FRBwd[n]);
           
            // std::cout << "Fleft["<<n<<"] "  << Fl[n] << " " << Fr[n] << std::endl; 
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
            numcoeff_eq0[P]                 =        Fmap_v[i][1][0];

            numcoeff_eq1[0]                 =       -Fmap_v[i][0][1];
            numcoeff_eq1[P]                 =        Fmap_v[i][1][1];

            numcoeff_eq2[0]                 =       -Fmap_v[i][0][2];
            numcoeff_eq2[P]                 =        Fmap_v[i][1][2];
        }
        else if(i == Nel-1)
        {

            numcoeff_eq0[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][0];
            numcoeff_eq0[Nel*(P+1)-1  ]     =        Fmap_v[i][1][0];

            numcoeff_eq1[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][1];
            numcoeff_eq1[Nel*(P+1)-1  ]     =        Fmap_v[i][1][1];

            numcoeff_eq2[(Nel-1)*(P+1)]     =       -Fmap_v[i][0][2];
            numcoeff_eq2[Nel*(P+1)-1  ]     =        Fmap_v[i][1][2];


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
    GetGlobalStiffnessMatrixNewBasis(Nel, P, wquad, D, Jac, map, Mdim, bkey, StiffnessMatGlobal);
    // GetGlobalStiffnessMatrixWeakNew(Nel, P, wquad, D, Jac, map, Mdim, basis, StiffnessMatGlobal);
    
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

        //std::cout << tmp0[i] << " " << tmp1[i] << " " << tmp2[i] << std::endl;
    }
    double **MassMatGlobal0          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, Bmat, MassMatGlobal0);

    double **MassMatGlobal1          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, Bmat, MassMatGlobal1);

    double **MassMatGlobal2          = dmatrix(Mdim);
    GetGlobalMassMatrixNew(Nel, P, wquad, Jac, map, Mdim, Bmat, MassMatGlobal2);

    dgetrf_(&Mdim, &Mdim, MassMatGlobal0[0], &Mdim, ipiv, &INFO);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal1[0], &Mdim, ipiv, &INFO);
    dgetrf_(&Mdim, &Mdim, MassMatGlobal2[0], &Mdim, ipiv, &INFO);

    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal0[0], &Mdim, ipiv, Ucoeff_eq0.data(), &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal1[0], &Mdim, ipiv, Ucoeff_eq1.data(), &Mdim, &INFO);
    dgetrs_(&TRANS, &Mdim, &NRHS, MassMatGlobal2[0], &Mdim, ipiv, Ucoeff_eq2.data(), &Mdim, &INFO);
    
    // Transform back onto quadrature points.
    std::vector<double> R_DG_row0(Nel*nq,0.0);
    std::vector<double> R_DG_row1(Nel*nq,0.0);
    std::vector<double> R_DG_row2(Nel*nq,0.0);

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
        
        std::vector<double> quad_e0 = BackwardTransform(P,  nq,  Bmat,  coeff_eq0_tmp);
        std::vector<double> quad_e1 = BackwardTransform(P,  nq,  Bmat,  coeff_eq1_tmp);
        std::vector<double> quad_e2 = BackwardTransform(P,  nq,  Bmat,  coeff_eq2_tmp);
        int Pf = P - 1;

        for(int i = 0;i < nq;i++)
        {
            R_DG_row0[i+nq*eln] = quad_e0[i];
            R_DG_row1[i+nq*eln] = quad_e1[i];
            R_DG_row2[i+nq*eln] = quad_e2[i];
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


    double gammaMone = 1.4-1.0;
    double rhoL  = Ul[0];
    double rhouL = Ul[1];
    double EL    = Ul[2];
    double uL    = rhouL/rhoL;
    double pL    = (EL-0.5*rhoL*uL*uL)*gammaMone;
    std::vector<double> Flvec(3,0.0);
    Flvec[0] = rhouL;
    Flvec[1] = pL+rhoL*uL*uL;
    Flvec[2] = (EL+pL)*uL;

    double rhoR  = Ur[0];
    double rhouR = Ur[1];
    double ER    = Ur[2];
    double uR    = rhouR/rhoR;
    double pR    = (ER-0.5*rhoR*uR*uR)*gammaMone;
    std::vector<double> Frvec(3,0.0);
    Frvec[0] = rhouR;
    Frvec[1] = pR+rhoR*uR*uR;
    Frvec[2] = (ER+pR)*uR;


    double aL = sqrt(1.4*pL/rhoL);
    double aR = sqrt(1.4*pR/rhoR);

    double srL  = sqrt(rhoL);
    double srR  = sqrt(rhoR);
    double srLR = srL + srR;
    double HL = (EL + pL) / rhoL;
    double HR = (ER + pR) / rhoR;
    double uRoe  = (srL * uL + srR * uR) / srLR;
    double HRoe  = (srL * HL + srR * HR) / srLR;
    double URoe2 = uRoe * uRoe;
    double cRoe = sqrt((1.4 - 1.0) * (HRoe - 0.5 * URoe2));
    // std::cout << "fabs(uRoe)+cRoe " << fabs(uRoe)+cRoe << std::endl;
    for(int n=0;n<nvar;n++)
    { 
        double Fl = Flvec[n];
        double Fr = Frvec[n];

        // double alphaL   = fabs(Ul[n])+aL;
        // double alphaR   = fabs(Ur[n])+aR;

        double alphaL   = fabs(uRoe)+cRoe;
        double alphaR   = fabs(uRoe)+cRoe;
        // Fn[n] = 0.5*((Fl+Fr)*normal-(fabs(uRoe)+cRoe)*(Ur[n]-Ul[n]));
        Fn[n] = 0.5*((Fl+Fr)*normal)-0.5*max(fabs(alphaL),fabs(alphaR))*(Ur[n]-Ul[n]);
        //Fn[n] = 0.5*((Fl+Fr)*normal-alphaR*(Ur[n]-Ul[n]));
    }


    return Fn;


}




std::vector<double> RoeRiemannVec(std::vector<double> Ul, std::vector<double> Ur, double normal)
{

    int nvar = Ul.size();
    std::vector<double> Fn(nvar,0.0);

    double gamma = 1.4;

    double gammaMone = 1.4-1.0;
    double rhoL  = Ul[0];
    double rhouL = Ul[1];
    double rhovL = 0.0;
    double rhowL = 0.0;
    double EL    = Ul[2];
    double uL    = rhouL/rhoL;
    double vL    = rhovL/rhoL;
    double wL    = rhowL/rhoL;
    double pL    = (EL-0.5*rhoL*uL*uL)*gammaMone;


    double rhoR  = Ur[0];
    double rhouR = Ur[1];
    double rhovR = 0.0;
    double rhowR = 0.0;
    double ER    = Ur[2];
    double uR    = rhouR/rhoR;
    double vR    = rhovR/rhoR;
    double wR    = rhowR/rhoR;
    double pR    = (ER-0.5*rhoR*uR*uR)*gammaMone;

    double aL = sqrt(1.4*pL/rhoL);
    double aR = sqrt(1.4*pR/rhoR);

    double srL  = sqrt(rhoL);
    double srR  = sqrt(rhoR);
    double srLR = srL + srR;
    double HL = (EL + pL) / rhoL;
    double HR = (ER + pR) / rhoR;
    double uRoe  = (srL * uL + srR * uR) / srLR;
    double vRoe  = (srL * vL + srR * vR) / srLR;
    double wRoe  = (srL * wL + srR * wR) / srLR;
    double HRoe  = (srL * HL + srR * HR) / srLR;
    double URoe2 = uRoe * uRoe;
    double cRoe = sqrt((1.4 - 1.0) * (HRoe - 0.5 * URoe2));
    double hRoe = HRoe;

    // Left and right velocities


    // Compute eigenvectors (equation 11.59).
    double k[5][5] = {{1., uRoe - cRoe, vRoe, wRoe, hRoe - uRoe * cRoe},
                 {1., uRoe, vRoe, wRoe, 0.5 * URoe2},
                 {0., 0., 1., 0., vRoe},
                 {0., 0., 0., 1., wRoe},
                 {1., uRoe + cRoe, vRoe, wRoe, hRoe + uRoe * cRoe}};

    // Calculate jumps \Delta u_i (defined preceding equation 11.67).
    double jump[5] = {rhoR - rhoL, rhouR - rhouL, rhovR - rhovL, rhowR - rhowL,
                 ER - EL};

    // Define \Delta u_5 (equation 11.70).
    double jumpbar = jump[4] - (jump[2] - vRoe * jump[0]) * vRoe -
                (jump[3] - wRoe * jump[0]) * wRoe;

    // Compute wave amplitudes (equations 11.68, 11.69).
    double alpha[5];
    alpha[1] = (gamma - 1.0) *
               (jump[0] * (hRoe - uRoe * uRoe) + uRoe * jump[1] - jumpbar) /
               (cRoe * cRoe);
    alpha[0] =
        (jump[0] * (uRoe + cRoe) - jump[1] - cRoe * alpha[1]) / (2.0 * cRoe);
    alpha[4] = jump[0] - (alpha[0] + alpha[1]);
    alpha[2] = jump[2] - vRoe * jump[0];
    alpha[3] = jump[3] - wRoe * jump[0];

    // Compute average of left and right fluxes needed for equation 11.29.
    double rhof  = 0.5 * (rhoL * uL + rhoR * uR);
    double rhouf = 0.5 * (pL + rhoL * uL * uL + pR + rhoR * uR * uR);
    double rhovf = 0.5 * (rhoL * uL * vL + rhoR * uR * vR);
    double rhowf = 0.5 * (rhoL * uL * wL + rhoR * uR * wR);
    double Ef    = 0.5 * (uL * (EL + pL) + uR * (ER + pR));

    // Needed to get right overload resolution for std::abs
    using std::abs;

    // Compute eigenvalues \lambda_i (equation 11.58).
    double  uRoeAbs   = abs(uRoe);
    double  lambda[5] = {abs(uRoe - cRoe), uRoeAbs, uRoeAbs, uRoeAbs,
                   abs(uRoe + cRoe)};

    // Finally perform summation (11.29).
    for (size_t i = 0; i < 5; ++i)
    {
        uRoeAbs = 0.5 * alpha[i] * lambda[i];

        rhof -= uRoeAbs * k[i][0];
        rhouf -= uRoeAbs * k[i][1];
        rhovf -= uRoeAbs * k[i][2];
        rhowf -= uRoeAbs * k[i][3];
        Ef -= uRoeAbs * k[i][4];
    }

    Fn[0] = rhof;
    Fn[1] = rhouf;
    Fn[2] = Ef;

    // for(int n=0;n<nvar;n++)
    // { 
    //     double Fl = Flvec[n];
    //     double Fr = Frvec[n];

    //     double alphaL   = fabs(Ul[n]);
    //     double alphaR   = fabs(Ur[n]);
    //     // Fn[n] = 0.5*((Fl+Fr)*normal-(fabs(uRoe)+cRoe)*(Ur[n]-Ul[n]));
    //     Fn[n] = 0.5*((Fl+Fr)*normal-max(fabs(alphaL),fabs(alphaR))*(Ur[n]-Ul[n]));
    //     //Fn[n] = 0.5*((Fl+Fr)*normal-alphaR*(Ur[n]-Ul[n]));
    // }


    return Fn;


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


void GetAllFwdBwd(int Nel, int nq, double *bc, double *quad, std::vector<double> &UtL, std::vector<double> &UtR)
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
            UtL[i] = quad[Nel*nq-1]; 
            UtR[i]=  quad[Nel*nq-1];
        }
        else
        {
            UtL[i] = quad[i*nq-1];
            UtR[i] = quad[i*nq];
        }
    }
}

void GetAllFwdBwdMap(int Nel, int nq, double *quad, std::map<int,std::vector<double> > &Umap)
{
    
    for(int i=0;i<Nel;i++)
    {
        std::vector<double> row(2,0.0);
        if(i == 0)
        {
            row[0] = quad[0];
            row[1] = quad[nq-1];

        }
        else if(i == Nel)
        {
            row[0] = quad[Nel*nq-1]; 
            row[1] = quad[Nel*nq-1];
        }
        else
        {
            row[0] = quad[i*nq];
            row[1] = quad[(i+1)*nq-1];
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





void GetGlobalStiffnessMatrixNewBasis(int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, Basis* bkey, double **StiffnessGlobal)
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
        // GetElementStiffnessMatrixNew(P, wquad, D, Jac[eln], basis, StiffnessElem);

        GetElementStiffnessMatrixNewBasis(P, wquad, D, Jac[eln], bkey, StiffnessElem);

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





void GetGlobalStiffnessMatrixWeakNewBasis(Basis* bkey, int Nel, int P, std::vector<double> wquad, double **D, double *Jac, int **map, int Mdim, std::vector<std::vector<double> > basis, double **StiffnessGlobal)
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
        GetElementStiffnessMatrixWeakNewBasis(P, wquad, D, Jac[eln], bkey, StiffnessElem);


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


/*
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
*/

void GetAllFwdBwdMapVecNodal(int Nel, int nq, std::vector<std::vector<double> > quad, std::map<int,std::vector<std::vector<double> > > &Umap, int nvar)
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
                row[1] = quad[n][nq-1];
            }
            else if(i == Nel)
            {
                row[0] = quad[n][(Nel-1)*nq-1]; 
                row[1] = quad[n][Nel*nq-1];
            }
            else
            {
                row[0] = quad[n][i*nq];
                row[1] = quad[n][(i+1)*nq-1];
            }

            rowrow[n] = row;
        }

        Umap[i] = rowrow;

    }
}

void GetAllFwdBwdMapVec(int Nel, int nq, std::vector<std::vector<double> > quad, std::map<int,std::vector<std::vector<double> > > &Umap, int nvar)
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
                row[1] = quad[n][nq-1];
            }
            else if(i == Nel)
            {
                row[0] = quad[n][(Nel-1)*nq-1]; 
                row[1] = quad[n][Nel*nq-1];
            }
            else
            {
                row[0] = quad[n][i*nq];
                row[1] = quad[n][(i+1)*nq-1];
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




void GetElementStiffnessMatrixNewBasis(int P, std::vector<double> wquad, double **D, double J, Basis* bkey, double **StiffMatElem)
{
    for(int i=0;i<P+1;i++)
    {
        for(int j=0;j<P+1;j++){
            StiffMatElem[i][j] = 0;
        }
    }

    std::vector<std::vector<double> > basis_update        = bkey->GetB();
    std::vector<std::vector<double> > basis_diff_update   = bkey->GetD();

    int nq = basis_update[0].size();

    for(int i=0;i<P+1;i++)
    {   
        std::vector<double> phi1 = basis_update[i];
        std::vector<double> dphi1 = basis_diff_update[i];
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = basis_update[j];
            std::vector<double> dphi2 = basis_diff_update[j];
            StiffMatElem[i][j] = J*integr(nq, wquad.data(), phi1.data(), dphi2.data());
            // StiffMatElem[i][j] = J*integr(nq, wquad.data(), basis_diff_update[i].data(), basis_update[j].data());
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






void GetElementStiffnessMatrixWeakNewBasis(int P, 
                                            std::vector<double> wquad, 
                                            double **D, double J, 
                                            Basis* bkey, double **StiffMatElem)
{
    for(int i=0;i<P+1;i++)
    {
        for(int j=0;j<P+1;j++){
            StiffMatElem[i][j] = 0;
        }
    }

    std::vector<std::vector<double> > basis_update        = bkey->GetB();
    std::vector<std::vector<double> > basis_diff_update   = bkey->GetD();

    int nq = basis_update[0].size();

    for(int i=0;i<P+1;i++)
    {   
        std::vector<double> phi1 = basis_update[i];
        std::vector<double> dphi1 = basis_diff_update[i];

        for(int s=0;s<nq;s++)
        {
            dphi1[s] = dphi1[s]/J;
        }
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = basis_update[j];
            std::vector<double> dphi2 = basis_diff_update[j];

            for(int s=0;s<nq;s++)
            {
                dphi2[s] = dphi2[s]/J;
            }

            StiffMatElem[i][j] = J*integr(nq, wquad.data(), dphi1.data(), phi2.data());
            // StiffMatElem[i][j] = J*integr(nq, wquad.data(), basis_diff_update[i].data(), basis_update[j].data());
        }
    }
}


















// // This member function calculates the inner product of the flux with the derivative of the basis functions.
// void InnerProductWRTDerivBasis(int np, int P, double J, double *w, double *z, double **D, double *quad, double *coeff)
// {
//     int ONE_INT=1;
//     double ONE_DOUBLE=1.0;
//     double ZERO_DOUBLE=0.0;
//     unsigned char TR = 'T';
    
//     int ncoeff = P;
//     double *tmp   = dvector(np);
//     double *phi1  = dvector(np);
//     double *dphi1 = dvector(np);
    
//     for(int n=0;n<P+1;n++)
//     {
        
//         //basis(np, P, n, z, phi1);
//         lagrange_basis(np, P, n, z, z, phi1);
//         diff(np, D, phi1, dphi1, J);
//         coeff[n] = J*integr(np, w, dphi1, quad);
//     }
// }




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






// void getbasis(int np, int P, int ncoeff, double *z, double **basis)
// {
//     //cout << "i="<< i << endl;
//     for(int i=0;i<ncoeff;i++)
//     {
//         if(i == 0){
//             for(int k=0;k<np;k++){
//                 basis[k][i] = (1 - z[k])/2;
//             }
//         }else if(i == P-1){
//             for(int k=0;k<np;k++){
//                 basis[k][i] = (1 + z[k])/2;
//             }
//         }else{
//             double *tmp = dvector(np);
//             jacobfd(np, z, tmp, NULL, i-1, 1.0, 1.0);
//             for(int k=0;k<np;k++){
//                 basis[k][i] = ((1-z[k])/2)*((1+z[k])/2)*tmp[k];
//             }
//         }
//     }
// }



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

