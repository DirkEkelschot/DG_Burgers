#include "Polylib.h"
#include "basis.h"
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <fstream>




#include "basis_poly.h"

// GLBasis implementation
// NodalBasis::NodalBasis(int P, std::string pt, std::vector<double>& z, std::vector<double>& w) {
//     // Initialize GLData
// }

// In basis_poly.cpp
// std::vector<double> BasisPoly::ConstructBasis() {
//     // Default implementation or throw an error
//     throw std::runtime_error("ConstructBasis not implemented in base class");
// }

// BasisPoly::~BasisPoly() = default;

// std::unique_ptr<BasisPoly> BasisPoly::Create(const std::string& bt, int P, std::string pt, std::vector<double>& z, std::vector<double>& w) {
//     if (bt == "Nodal") return std::make_unique<NodalBasis>(P, pt, z, w);
//     if (bt == "Modal") return std::make_unique<ModalBasis>(P, pt, z, w);
//     throw std::invalid_argument("Unknown basis type");
// }


std::vector<double> NodalBasis::ConstructBasis()  
{
    double *mode;

    int numModes    = m_P + 1;
    int nq          = m_z.size();
    int np          = m_P + 1;
    int numPoints   = nq;

    m_zn.resize(np);
    m_wn.resize(np);
    m_bl.resize(numModes);
    m_br.resize(numModes);

    double** D  = (double **)malloc(nq*sizeof(double *));
    double** Dt = (double **)malloc(nq*sizeof(double *));
    D[0]        = (double *)malloc(nq*nq*sizeof(double));
    Dt[0]       = (double *)malloc(nq*nq*sizeof(double));

    if(m_pt.compare("GaussLegendreLobatto") == 0)
    {
        polylib::zwgll(m_z.data(), m_w.data(), nq);
        polylib::zwgll(m_zn.data(), m_wn.data(), np);
        // polylib::Dgll(D, Dt, m_z.data(), nq);
    }
    if(m_pt.compare("GaussLegendre") == 0)
    {
        polylib::zwgl(m_z.data(), m_w.data(), nq);
        polylib::zwgl(m_zn.data(), m_wn.data(), np);
        // polylib::Dgl(D, Dt, m_z.data(), nq);
    }

    m_bdata.resize(numModes*nq);
    m_dbdata.resize(numModes*nq);

    mode = m_bdata.data();

    if(m_pt.compare("GaussLegendreLobatto") == 0)
    {
        for (int p = 0; p < numModes; ++p, mode += numPoints)
        {
            std::vector<double> phi1(nq);
            for (int q = 0; q < numPoints; ++q)
            {
                phi1[q] = polylib::hglj(p, m_z[q], m_zn.data(), numModes, 0.0, 0.0);
                m_bdata[p*nq+q] = phi1[q];
            }
            
            std::vector<double> diff_mode(nq);
            for(int i=0;i<nq;i++)
            {
                diff_mode[i] = 0;
                for(int j=0;j<nq;j++)
                {
                    diff_mode[i] = diff_mode[i] + D[i][j]*phi1[j];
                }
                diff_mode[i] = diff_mode[i]/1.0;
                m_dbdata[p*nq+i] = diff_mode[i]/1.0;
            }

            // m_dbdata[p] = diff_mode;
            // m_bl[p]   = GetNodalBasisValue(m_P, -1.0, p, m_zn, m_pt);
            // m_br[p]   = GetNodalBasisValue(m_P,  1.0, p, m_zn, m_pt);
            
        }



    }
    if(m_pt.compare("GaussLegendre") == 0)
    {
        for (int p = 0; p < numModes; ++p, mode += numPoints)
        {
            std::vector<double> phi1(nq);
            for (int q = 0; q < numPoints; ++q)
            {
                phi1[q] = polylib::hgj(p, m_z[q], m_zn.data(), numModes, 0.0, 0.0);
                m_bdata[p*nq+q] = phi1[q];
            }
            
            std::vector<double> diff_mode(nq);
            for(int i=0;i<nq;i++)
            {
                diff_mode[i] = 0;
                for(int j=0;j<nq;j++)
                {
                    diff_mode[i] = diff_mode[i] + D[i][j]*phi1[j];
                }
                diff_mode[i] = diff_mode[i]/1.0;
                m_dbdata[p*nq+i] = diff_mode[i]/1.0;
            }
            // m_dbdata[p] = diff_mode;

            // m_bl[p]   = GetNodalBasisValue(m_P, -1.0, p, m_zn, m_pt);
            // m_br[p]   = GetNodalBasisValue(m_P,  1.0, p, m_zn, m_pt);
        }
    }
    // Implementation using m_data
}

// // GLLBasis implementation
// ModalBasis::ModalBasis(int P, std::string pt, std::vector<double>& z, std::vector<double>& w) {
//     // Initialize GLLData
// }

std::vector<double> ModalBasis::ConstructBasis()  
{
    
    double *mode;
    int numModes    = m_P + 1;
    int nq          = m_z.size();
    int np          = m_P + 1;
    int numPoints   = nq;

    double** D  = (double **)malloc(nq*sizeof(double *));
    double** Dt = (double **)malloc(nq*sizeof(double *));
    D[0]        = (double *)malloc(nq*nq*sizeof(double));
    Dt[0]       = (double *)malloc(nq*nq*sizeof(double));

    m_zn.resize(np);
    m_wn.resize(np);
    m_bl.resize(numModes);
    m_br.resize(numModes);
    
    m_bdata.resize(numModes*nq);
    m_dbdata.resize(numModes*nq);

    if(m_pt.compare("GaussLegendreLobatto") == 0)
    {
        polylib::zwgll(m_z.data(), m_w.data(), nq);
        polylib::zwgll(m_zn.data(), m_wn.data(), np);
        // polylib::Dgll(D, Dt, z.data(), nq);
    }
    if(m_pt.compare("GaussLegendre") == 0)
    {
        polylib::zwgl(m_z.data(), m_w.data(), nq);
        polylib::zwgl(m_zn.data(), m_wn.data(), np);
        // polylib::Dgl(D, Dt, z.data(), nq);
    }
    for (int n = 0; n < numModes; ++n)
    {
        
        std::vector<double> phi1(nq,0.0);

        if(n == 0)
        {
            for(int k=0;k<nq;k++)
            {
                phi1[k] = (1 - m_z[k])/2;
                m_bdata[n*nq+k] = phi1[k];
            }
        }
        else if(n == m_P)
        {
            for(int k=0;k<nq;k++)
            {
                phi1[k] = (1 + m_z[k])/2;
                m_bdata[n*nq+k] = phi1[k];
            }
        }
        else
        {
            polylib::jacobfd(nq, m_z.data(), phi1.data(), NULL, n-1, 1.0, 1.0);
            //jacobd(nq, zquad.data(), phi1.data(), n-1, 1.0, 1.0);
            for(int k=0;k<nq;k++)
            {
                phi1[k] = ((1-m_z[k])/2)*((1+m_z[k])/2)*phi1[k];
                m_bdata[n*nq+k] = phi1[k];
            }
        }

        

        std::vector<double> diff_mode(nq);

        for(int i=0;i<nq;i++)
        {
            diff_mode[i] = 0;

            for(int j=0;j<nq;j++)
            {
                diff_mode[i] = diff_mode[i] + D[i][j]*phi1[j];
                m_dbdata[n*nq+i] = diff_mode[i];
            }
            
        }
    }

    for(int p = 0; p < numModes; p++)
    {
        // m_bl[p]   = GetModalBasisValue(m_P,  -1.0, p, m_z.size(), m_pt);
        // m_br[p]   = GetModalBasisValue(m_P,   1.0, p, m_z.size(), m_pt);
    }
    // Implementation using m_data
}