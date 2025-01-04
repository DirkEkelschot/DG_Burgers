#include "Polylib.h"
#include <cblas.h>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>


#ifndef BASIS_POLY_H
#define BASIS_POLY_H


using namespace std;

class BasisPoly {
public:
    virtual ~BasisPoly() = default;
    virtual void ConstructBasis(){
    // Default implementation or throw an error
    throw std::runtime_error("ConstructBasis not implemented in base class");
};
    
    // virtual std::vector<double> EvaluatePoints();

    static std::unique_ptr<BasisPoly> Create(const std::string& bt, int P, std::string pt, std::vector<double>& z, std::vector<double>& w);

protected:
    BasisPoly(int P, std::string pt, std::vector<double>& z, std::vector<double>& w) : m_P(P),m_pt(pt),m_z(z),m_w(w) {}
    
    // inputs
    int m_P;
    std::string m_pt;
    std::vector<double>& m_z;
    std::vector<double>& m_w;

     // additional data
    std::vector<double> m_zn;
    std::vector<double> m_wn;
    std::vector<double> m_bl;
    std::vector<double> m_br;
    std::vector<double> m_bdata;
    std::vector<double> m_dbdata;

};


//=================================================================

class NodalBasis : public BasisPoly {
public:
    NodalBasis(int P, std::string pt, std::vector<double>& z, std::vector<double>& w) : BasisPoly(P, pt, z, w) {}
    
    void ConstructBasis() override;
    // std::vector<double> EvaluatePoints() override;
};

class ModalBasis : public BasisPoly {
public:
    ModalBasis(int P, std::string pt, std::vector<double>& z, std::vector<double>& w) : BasisPoly(P, pt, z, w) {}

    void ConstructBasis() override;
    // std::vector<double> EvaluatePoints() override;
};









//=================================================================


#endif