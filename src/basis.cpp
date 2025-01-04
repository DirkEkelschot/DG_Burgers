#include "Polylib.h"
#include "basis.h"
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;


Basis::Basis(std::string pt, 
            std::string bt, 
            std::vector<double>& z, 
            std::vector<double>& w,
            int P, int nq)
{
    
    double *mode;



    double** D  = (double **)malloc(nq*sizeof(double *));
    double** Dt = (double **)malloc(nq*sizeof(double *));
    D[0]        = (double *)malloc(nq*nq*sizeof(double));
    Dt[0]       = (double *)malloc(nq*nq*sizeof(double));

    for(int i=1;i<nq;i++){
        D[i]  = D[i-1]+nq;
        Dt[i] = Dt[i-1]+nq;
    }

    const double *Din;
    Din = &(*D)[0];

    int numModes  = P + 1;
    int numPoints = nq;

    m_bdata.resize(numModes*nq);
    m_dbdata.resize(numModes*nq);

    ptype = pt;
    btype = bt;

    int np = P + 1;
    zn.resize(np);
    wn.resize(np);
    if(ptype.compare("GaussLegendreLobatto") == 0)
    {
        polylib::zwgll(z.data(), w.data(), nq);
        polylib::zwgll(zn.data(), wn.data(), np);
        polylib::Dgll(D, Dt, z.data(), nq);
        
    }
    if(ptype.compare("GaussLegendre") == 0)
    {
        polylib::zwgl(z.data(), w.data(), nq);
        polylib::zwgl(zn.data(), wn.data(), np);
        polylib::Dgl(D, Dt, z.data(), nq);
    }

    std::vector<double> DinNew(nq*nq);

    for(int i=0;i<nq;i++)
    {
        for(int j=0;j<nq;j++)
        {
             DinNew[i*nq+j] = D[i][j];
        }
       
    }

    bl.resize(numModes,0.0);
    br.resize(numModes,0.0);

    if(btype == "Modal")
    {
        for (int n = 0; n < numModes; ++n)
        {
           
            std::vector<double> phi1(nq,0.0);

            if(n == 0)
            {
                for(int k=0;k<nq;k++)
                {
                    phi1[k] = (1 - z[k])/2;
                    m_bdata[n*nq+k] = phi1[k];
                }
            }
            else if(n == P)
            {
                for(int k=0;k<nq;k++)
                {
                    phi1[k] = (1 + z[k])/2;
                    m_bdata[n*nq+k] = phi1[k];
                }
            }
            else
            {
                polylib::jacobfd(nq, z.data(), phi1.data(), NULL, n-1, 1.0, 1.0);
                //jacobd(nq, zquad.data(), phi1.data(), n-1, 1.0, 1.0);
                for(int k=0;k<nq;k++)
                {
                    phi1[k] = ((1-z[k])/2)*((1+z[k])/2)*phi1[k];
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
            bl[p]   = GetModalBasisValue(P,  -1.0, p, z.size(), ptype);
            br[p]   = GetModalBasisValue(P,   1.0, p, z.size(), ptype);
        }
        
        
        // define derivative basis
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //             numPoints, numModes, numPoints, 1.0, DinNew.data(), 
        //             numPoints, m_bdata.data(), numPoints, 0.0, 
        //             m_dbdata.data(), numPoints);

    }

    if(btype == "Nodal")
    {

        mode = m_bdata.data();

        if(ptype.compare("GaussLegendreLobatto") == 0)
        {
            for (int p = 0; p < numModes; ++p, mode += numPoints)
            {
                std::vector<double> phi1(nq);
                for (int q = 0; q < numPoints; ++q)
                {
                    phi1[q] = polylib::hglj(p, z[q], zn.data(), numModes, 0.0, 0.0);
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
                bl[p]   = GetNodalBasisValue(P, -1.0, p, zn, ptype);
                br[p]   = GetNodalBasisValue(P,  1.0, p, zn, ptype);
                
            }



        }
        if(ptype.compare("GaussLegendre") == 0)
        {
            for (int p = 0; p < numModes; ++p, mode += numPoints)
            {
                std::vector<double> phi1(nq);
                for (int q = 0; q < numPoints; ++q)
                {
                    phi1[q] = polylib::hgj(p, z[q], zn.data(), numModes, 0.0, 0.0);
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

                bl[p]   = GetNodalBasisValue(P, -1.0, p, zn, ptype);
                br[p]   = GetNodalBasisValue(P,  1.0, p, zn, ptype);
            }
        }



        // define derivative basis
        // diff( O*nq, Dplot, basis_plot[i].data(), basis_plot_diff.data(), 1.0);

        

        

        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //             numPoints, numModes, numPoints, 1.0, DinNew.data(), 
        //             numPoints, m_bdata.data(), numPoints, 0.0, 
        //             m_dbdata.data(), numPoints);

    }

    //std::cout << "Basis print out " << btype << std::endl;
    for(int p=0;p<numModes;p++)
    {
        std::vector<double> basis_row(numPoints);
        std::vector<double> diff_basis_row(numPoints);

        for(int s=0;s<numPoints;s++)
        {
            basis_row[s]         = m_bdata[p*numPoints+s];
            diff_basis_row[s]    = m_dbdata[p*numPoints+s];
            //std::cout << basis_row[s] << " "; 
        }

        //std::cout << std::endl;
        m_bdata_out.push_back(basis_row);
        m_dbdata_out.push_back(diff_basis_row);
    }
}


double Basis::GetNodalBasisValue(int P, 
                              double xref,
                              int i,
                              std::vector<double> zquad,
                              std::string ptype)
{

    double res = 0.0;
    int numModes = P + 1;
    
    // phi1[q] = hglj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);

    if(ptype == "GaussLegendre")
    {
        // res = hgj(i, zquad.data(), xref, numModes, 0.0, 0.0);
        res = polylib::hgj(i, xref, zquad.data(), numModes, 0.0, 0.0);
    }
    if(ptype == "GaussLegendreLobatto")
    {
        // res = hglj(i, zquad.data(), xref, numModes, 0.0, 0.0);
        res = polylib::hglj(i, xref, zquad.data(), numModes, 0.0, 0.0);
    }


    return res;
}

double Basis::GetModalBasisValue(int P, 
                              double zref, 
                              int n,
                              int nq, 
                              std::string ptype)
{

    double res = 0.0;
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    std::vector<double> zref_tmp(1,0.0);
    zref_tmp[0] = zref;

    std::vector<double> phi1(1,0.0);

    if(n == 0)
    {
        phi1[0] = (1 - zref)/2;
    }
    else if(n == P)
    {
        
        phi1[0] = (1 + zref)/2;
        
    }
    else
    {
        polylib::jacobfd(1, zref_tmp.data(), phi1.data(), NULL, n-1, 1.0, 1.0);

        phi1[0] = ((1-zref)/2)*((1+zref)/2)*phi1[0];
    }
    
    res = phi1[0];
    // }
    return res;
}




std::vector<double> Basis::BackwardTransformValNodal(int P, 
                                      double xq, 
                                      std::vector<double> input_coeff,
                                      std::string ptype)
{
    int numModes = P + 1;

    std::vector<double> quad(1,0.0);
    double sum = 0.0;
    for(int i = 0;i<P+1;i++)
    {
        double phi1 = polylib::hgj(i, xq, zn.data(), numModes, 0.0, 0.0);

        quad[0] = quad[0]+input_coeff[i]*phi1;
        
    }

    return quad;
}




std::vector<double> Basis::BackwardTransformValModal(int P, 
                                      double xq, 
                                      std::vector<double> input_coeff,
                                      std::string ptype)
{
    std::vector<double> quad(1,0.0);
    int numModes = P + 1;
    for (int n = 0; n < numModes; ++n)
    {
        std::vector<double> phi1(1,0.0);
        std::vector<double> xqtmp(1,0.0);
        xqtmp[0] = xq;
        if(n == 0)
        {
            phi1[0] = (1 - xq)/2;
        }
        else if(n == P)
        {
            phi1[0] = (1 + xq)/2;
        }
        else
        {
            polylib::jacobfd(1, xqtmp.data(), phi1.data(), NULL, n-1, 1.0, 1.0);
            phi1[0] = ((1-xq)/2)*((1+xq)/2)*phi1[0];
        }
        quad[0] = quad[0]+input_coeff[n]*phi1[0];
    }

    return quad;
}


std::vector<double> Basis::GetBasisLeftValues()
{
    return bl;
}

std::vector<double> Basis::GetBasisRightValues()
{
    return br;
}

std::vector<std::vector<double> > Basis::GetB()
{
    return m_bdata_out;
}

std::vector<std::vector<double> > Basis::GetD()
{
    return m_dbdata_out;
}

std::vector<double> Basis::GetZn()
{
    return zn;
}

std::vector<double> Basis::GetWn()
{
    return wn;
}

std::string Basis::GetBtype()
{
    return btype;
}

Basis::~Basis()
{

}