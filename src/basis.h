#include "Polylib.h"
#include <cblas.h>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>


#ifndef BASIS_H
#define BASIS_H

class Basis {

    public:
        // Constructor
             Basis(){};
             Basis(std::string pt, 
                    std::string bt, 
                    std::vector<double>& z, 
                    std::vector<double>& w,
                    int P, int nq);
            ~Basis();

            std::vector<std::vector<double> > GetB();
            std::vector<std::vector<double> > GetD();
            std::vector<double> GetZn();
            std::vector<double> GetWn();
            std::string GetBtype();

            double GetNodalBasisValue(int P, 
                              double xref,
                              int i,
                              std::vector<double> zquad,
                              std::string ptype);

            double  GetModalBasisValue(int P, 
                              double zref, 
                              int n,
                              int nq, 
                              std::string ptype);
            
            std::vector<double> GetBasisLeftValues();
            std::vector<double> GetBasisRightValues();

            std::vector<double> BackwardTransformValModal(int P, 
                                      double xq, 
                                      std::vector<double> input_coeff,
                                      std::string ptype);

            std::vector<double> BackwardTransformValNodal(int P, 
                                      double xq, 
                                      std::vector<double> input_coeff,
                                      std::string ptype);

    private:
        std::vector<std::vector<double> > m_bdata_out;
        std::vector<std::vector<double> > m_dbdata_out;

        std::vector<double> m_bdata;
        std::vector<double> m_dbdata;

        std::vector<double> zn;
        std::vector<double> wn;

        std::vector<double> bl;
        std::vector<double> br;

        std::string ptype;
        std::string btype;

};

#endif