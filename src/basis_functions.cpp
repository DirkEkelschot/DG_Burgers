#include "basis_functions.h"


std::vector<std::vector<double> > getLegendreBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

   
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(np,0.0);

        if(n == 0)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = 1.0;
            }
        }
        else if(n == 1)
        {
            for(int k=0;k<np;k++)
            {
                phi1[k] = zquad[k];
            }
        }
        else
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);

            for(int k=0;k<np;k++)
            {
                phi1[k] = phi1[k];
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}











std::vector<std::vector<double> > getRadauPlusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(zquad_eval.size(),0.0);

        if(n == 0)
        {
            jacobfd(np, zquad_eval.data(), phi1.data(), NULL, 1, 0.0, 0.0);
            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = 0.0;
            }
        }
        else
        {
            std::vector<double> phi2(zquad_eval.size(),0.0);

            jacobfd(np, zquad_eval.data(), phi1.data(), NULL, n, 0.0, 0.0);
            jacobfd(np, zquad_eval.data(), phi2.data(), NULL, n-1, 0.0, 0.0);

            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = 0.5*(phi1[k] + phi2[k]);
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}




std::vector<std::vector<double> > getRadauMinusBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

   
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(zquad_eval.size(),0.0);

        if(n == 0)
        {
            jacobfd(np, zquad.data(), phi1.data(), NULL, 1, 0.0, 0.0);
            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = 0.0;
            }
        }
        else
        {
            std::vector<double> phi2(zquad_eval.size(),0.0);

            jacobfd(np, zquad.data(), phi1.data(), NULL, n, 0.0, 0.0);
            jacobfd(np, zquad.data(), phi2.data(), NULL, n-1, 0.0, 0.0);

            for(int k=0;k<zquad_eval.size();k++)
            {
                phi1[k] = pow(-1, n)*0.5*(phi1[k] - phi2[k]);
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}




std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int nq, int np, int P)
{

    if(nq != zquad.size())
    {
        std::cout << "error: nq != zquad.size() " << std::endl;
    }
    int numModes = P + 1;


    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {
        std::vector<double> phi1(zquad_eval.size());
        for (int q = 0; q < zquad_eval.size(); ++q)
        {
            phi1[q] = hglj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
        }
        basis.push_back(phi1);
    }
    return basis;
}




std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int np, int P)
{

   
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




// void run_LegendreModalBasis_test()
// {

// }

// void run_ModifiedModalBasis_test()
// {
    
// }

// void run_LagrangeNodalBasis_test()
// {
    
// }

// void run_RadauMinusBasis_test()
// {
    
// }


// void run_RadauPlusBasis_test()
// {
    
// }

