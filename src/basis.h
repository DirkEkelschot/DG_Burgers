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

    private:
        std::vector<std::vector<double> > m_bdata_out;
        std::vector<std::vector<double> > m_dbdata_out;

        std::vector<double> m_bdata;
        std::vector<double> m_dbdata;

        std::string ptype;
        std::string btype;

};

#endif