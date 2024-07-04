#include <vector>
#include <iostream>
#include "Polylib.h"
#ifndef BASIS_FUNCTIONS_H
#define BASIS_FUNCTIONS_H

using namespace std;
using namespace polylib;

std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);
std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P);


#endif