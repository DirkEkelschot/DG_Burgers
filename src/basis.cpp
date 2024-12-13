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
    std::vector<double> zn(np);
    std::vector<double> wn(np);
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

    // std::cout << "komtie hoor " << std::endl;
    for(int i=0;i<nq;i++)
    {
        for(int j=0;j<nq;j++)
        {
             DinNew[i*nq+j] = D[i][j];
            //  std::cout <<  D[i][j] << " ";
        }
        // std::cout << std::endl;
    }



    // if(btype == "Modal")
    // {
    //     for (int i = 0; i < numPoints; ++i)
    //     {
    //         m_bdata[i]             = 0.5 * (1 - z[i]);
    //         m_bdata[numPoints + i] = 0.5 * (1 + z[i]);

    //         m_dbdata[i]             = -0.5;
    //         m_dbdata[numPoints + i] = 0.5;
    //     }

    //     mode = m_bdata.data() + 2 * numPoints;

    //     for (int p = 2; p < numModes; ++p, mode += numPoints)
    //     {
    //         // polylib::jacobfd(numPoints, z.data(), mode, NULL, p - 1, 1.0,
    //                         // 1.0);
    //         std::vector<double> phi1(nq);
    //         polylib::jacobfd(nq, z.data(), phi1.data(), NULL, p-1, 1.0, 1.0);
    //         for (int i = 0; i < numPoints; ++i)
    //         {
    //             // mode[i] *= m_bdata[i] * m_bdata[numPoints + i];
    //             phi1[i] = ((1-z[i])/2)*((1+z[i])/2)*phi1[i];
    //             m_dbdata[p*nq+i] = phi1[i];
    //         }

           

    //         std::vector<double> diff_mode(nq);

    //         for(int i=0;i<nq;i++)
    //         {
    //             diff_mode[i] = 0;

    //             for(int j=0;j<nq;j++)
    //             {
    //                 diff_mode[i] = diff_mode[i] + D[i][j]*phi1[j];
    //             }

    //             diff_mode[i]     = diff_mode[i] / 1.0;
    //             m_dbdata[p*nq+i] = diff_mode[i] / 1.0;
    //         }
    //     }

        
    //     // define derivative basis
    //     // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //     //             numPoints, numModes, numPoints, 1.0, DinNew.data(), 
    //     //             numPoints, m_bdata.data(), numPoints, 0.0, 
    //     //             m_dbdata.data(), numPoints);

    // }

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
            }
        }

        // define derivative basis
        // diff( O*nq, Dplot, basis_plot[i].data(), basis_plot_diff.data(), 1.0);

        

        

        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //             numPoints, numModes, numPoints, 1.0, DinNew.data(), 
        //             numPoints, m_bdata.data(), numPoints, 0.0, 
        //             m_dbdata.data(), numPoints);

    }


    for(int p=0;p<numModes;p++)
    {
        std::vector<double> basis_row(numPoints);
        std::vector<double> diff_basis_row(numPoints);

        for(int s=0;s<numPoints;s++)
        {
            basis_row[s]         = m_bdata[p*numPoints+s];
            diff_basis_row[s]    = m_dbdata[p*numPoints+s];
        }

        m_bdata_out.push_back(basis_row);
        m_dbdata_out.push_back(diff_basis_row);
    }

}

std::vector<std::vector<double> > Basis::GetB()
{
    return m_bdata_out;
}

std::vector<std::vector<double> > Basis::GetD()
{
    return m_dbdata_out;
}

Basis::~Basis()
{

}