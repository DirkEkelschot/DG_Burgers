#include <vector>
#include <iostream>
#include "Polylib.h"
#include <fstream>
#ifndef BASIS_FUNCTIONS_H
#define BASIS_FUNCTIONS_H

using namespace std;
using namespace polylib;


double **dmatrix(int Mdim);

double integr(int np, double *w, double *phi1, double *phi2);

void *diff(int np, double **D, double *p, double *pd, double J);

std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);




std::vector<double> GetElementMassMatrix(int P, 
                            std::vector<std::vector<double> > basis,
                            std::vector<double> wquad,
                            double J);





std::vector<double> BackwardTransform(int P, int np, std::vector<std::vector<double> > basis, std::vector<double> input_coeff);
std::vector<double> ForwardTransform(int P, int np, std::vector<std::vector<double> > basis, std::vector<double>wquad, int nq, double J, std::vector<double> input_quad);



void run_nodal_test(double* x, 
                    std::vector<double> z, 
                    std::vector<double> w, 
                    double* bound, 
                    double* Jac, 
                    int np, int nq, int P);

void run_modal_test(double* x, 
                    std::vector<double> z, 
                    std::vector<double> w, 
                    double* bound, 
                    double* Jac, 
                    int np, int nq, int P);

#endif