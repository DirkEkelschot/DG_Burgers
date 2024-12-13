#include <vector>
#include <iostream>
#include "Polylib.h"
#include "basis.h"
#include <fstream>
#include <cmath>
#ifndef BASIS_FUNCTIONS_H
#define BASIS_FUNCTIONS_H

using namespace std;
using namespace polylib;


double **dmatrix(int Mdim);

double integr(int np, double *w, double *phi1, double *phi2);

void *diff(int np, double **D, double *p, double *pd, double J);

void run_new_basis_test(std::vector<double> zq, std::vector<double>  wq, int nq, int P);

std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int P,std::string ptype);
std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int P,std::string ptype);
std::vector<std::vector<double> > getNodalBasisEvalNew(std::vector<double> zquad_eval,std::vector<double> zquad, int P);
std::vector<std::vector<double> > getModalBasisEval2(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);



std::vector<double> GetElementMassMatrix(int P, 
                            std::vector<std::vector<double> > basis,
                            std::vector<double> wquad,
                            double J);





std::vector<double> BackwardTransform(int P, int nq, std::vector<std::vector<double> > basis, std::vector<double> input_coeff);
std::vector<double> ForwardTransform(int P, std::vector<std::vector<double> > basis, std::vector<double>wquad, int nq, double J, std::vector<double> input_quad);


double EvaluateFromNodalBasis(int P, 
                              double xref,
                              std::vector<double> coeff,
                              std::vector<double> zquad,
                              std::string ptype);

double EvaluateFromModalBasis(int P, 
                              double zref, 
                              int nq, 
                              std::vector<double> coeff, 
                              std::string ptype);


void run_nodal_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    int nq, int P, int O, std::string ptype);

void run_nodal_test_new(std::vector<double> zq,
                    std::vector<double> z, 
                    std::vector<double> w, int P, int O);
                    

void run_modal_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    std::vector<double> w, 
                    double* bound, 
                    double* Jac, 
                    int np, int nq, int P, int O, std::string ptype);


void run_left_radau_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    int nq, int P, int O, std::string ptype);


void run_right_radau_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    int nq, int P, int O, std::string ptype);

#endif