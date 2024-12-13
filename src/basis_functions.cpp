#include "basis_functions.h"
#include "basis.h"

extern "C" {extern void dgetrf_(int *, int *, double (*), int *, int [], int*);}
extern "C" {extern void dgetrs_(unsigned char *, int *, int *, double (*), int *, int [], double [], int *, int *);}
extern "C" {extern void dgemv_(unsigned char *, int *, int *, double *, double (*), int *, double [], int *, double *, double [], int *);}
extern "C" {extern void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);}

void *diff(int np, double **D, double *p, double *pd, double J)
{
    for(int i=0;i<np;i++){
        pd[i] = 0;
        for(int j=0;j<np;j++){
            pd[i] = pd[i] + D[i][j]*p[j];
        }
        pd[i] = pd[i]/J;
        
    }
    return 0;
}

// This member function perform numerical quadrature.
double integr(int np, double *w, double *phi1, double *phi2)
{
  register double sum = 0.;

  for(int i=0;i<np;i++){
    sum = sum + w[i]*phi1[i]*phi2[i]; 
  }
  return sum;
}


double **dmatrix(int n)
{
  double **A;
  A = (double **)malloc(n*sizeof(double *));
  A[0] = (double *)malloc(n*n*sizeof(double));

  for(int i=1;i<n;i++){
    A[i] = A[i-1]+n;
  }
  return A;
}



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




std::vector<std::vector<double> > getNodalBasisEval(std::vector<double> zquad_eval,std::vector<double> zquad, int P, std::string ptype)
{

    // if(nq != zquad_eval.size())
    // {
    //     std::cout << "error: nq != zquad.size() " << nq << " " << zquad_eval.size() << std::endl;
    // }

    int numModes = P + 1;


    std::vector<std::vector<double> > basis;
    if (ptype.compare("GLL") == 0)
    {
        //std::cout << "with " << "GLL " << ptype << std::endl;
        for(int n=0;n<numModes;n++)
        {
            std::vector<double> phi1(zquad_eval.size());
            for (int q = 0; q < zquad_eval.size(); ++q)
            {
                phi1[q] = hglj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
                //phi1[q] = hgj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
            }
            basis.push_back(phi1);
        }

    }

     if (ptype.compare("GL") == 0)
    {
        //std::cout << "with " << ptype << std::endl;
        for(int n=0;n<numModes;n++)
        {
            std::vector<double> phi1(zquad_eval.size());
            for (int q = 0; q < zquad_eval.size(); ++q)
            {
                //phi1[q] = hglj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
                phi1[q] = hgj(n, zquad_eval[q], zquad.data(), numModes, 0.0, 0.0);
            }
            basis.push_back(phi1);
        }

    }
    
    return basis;
}


std::vector<std::vector<double> > getNodalBasisEvalNew(std::vector<double> zquad_eval,std::vector<double> zquad, int P)
{

    // if(nq != zquad_eval.size())
    // {
    //     std::cout << "error: nq != zquad.size() " << nq << " " << zquad_eval.size() << std::endl;
    // }

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




double EvaluateFromModalBasis(int P, 
                              double zref, 
                              int nq, 
                              std::vector<double> coeff, 
                              std::string ptype)
{

    double res = 0.0;
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    std::vector<double> zref_tmp(1,0.0);
    zref_tmp[0] = zref;
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(1,0.0);

        if(n == 0)
        {
            phi1[0] = (1 - zref)/2;
        }
        else if(n == 1)
        {
            
            phi1[0] = (1 + zref)/2;
            
        }
        else
        {
            jacobfd(1, zref_tmp.data(), phi1.data(), NULL, n-1, 1.0, 1.0);

            phi1[0] = ((1-zref)/2)*((1+zref)/2)*phi1[0];
        }
        
        res = res + coeff[n]*phi1[0];
    }
    return res;
}




std::vector<std::vector<double> > getModalBasisEval(std::vector<double> zquad_eval, std::vector<double> zquad, int nq, int P, std::string ptype)
{

   
    int numModes = P + 1;

    std::vector<std::vector<double> > basis;
    
    for(int n=0;n<numModes;n++)
    {   
        std::vector<double> phi1(nq,0.0);

        if(n == 0)
        {
            for(int k=0;k<nq;k++)
            {
                phi1[k] = (1 - zquad[k])/2;
            }
        }
        else if(n == P)
        {
            for(int k=0;k<nq;k++)
            {
                phi1[k] = (1 + zquad[k])/2;
            }
        }
        else
        {
            jacobfd(nq, zquad.data(), phi1.data(), NULL, n-1, 1.0, 1.0);
            //jacobd(nq, zquad.data(), phi1.data(), n-1, 1.0, 1.0);
            for(int k=0;k<nq;k++)
            {
                phi1[k] = ((1-zquad[k])/2)*((1+zquad[k])/2)*phi1[k];
            }
        }
        
        basis.push_back(phi1);
    }
    return basis;
}














std::vector<double> GetElementMassMatrix(int P, 
                            std::vector<std::vector<double> > basis,
                            std::vector<double> wquad,
                            double J)
{
    std::vector<double> MassMatElem((P+1)*(P+1),0.0);
    int np = wquad.size();
    for(int i=0;i<P+1;i++)
    {
        std::vector<double> phi1 = basis[i];
        for(int j=0;j<P+1;j++)
        {
            std::vector<double> phi2 = basis[j];
            MassMatElem[i*(P+1)+j] = J*integr(np, wquad.data(), phi1.data(), phi2.data());
        }
    }
    return MassMatElem;
}



void run_new_basis_test(std::vector<double> zq, std::vector<double>  wq, int nq, int P)
{

    // NODAL BASIS TEST USING THE NEW IMPLEMENTATION
    Basis* bNodalkey = new Basis("GaussLegendre",
                       "Nodal",
                       zq,wq,P,nq);

    std::vector<std::vector<double> > Dmat = bNodalkey->GetD();
    std::vector<std::vector<double> > Bmat = bNodalkey->GetB();

    ofstream soloutnew;
    soloutnew.open("new_nodal_basis.out");
    for(int j=0;j<nq;j++)
    {
        soloutnew << zq[j] << " ";
    }
    soloutnew << std::endl;

    for(int i=0;i<(P+1);i++)
    {
        for(int j=0;j<nq;j++)
        {
            soloutnew << Bmat[i][j] << " ";
        }
        soloutnew << std::endl;
    }

    soloutnew.close();


    ofstream diffoutnew;
    diffoutnew.open("new_diff_nodal_basis.out");
    for(int i=0;i<(P+1);i++)
    {
        for(int j=0;j<nq;j++)
        {
            diffoutnew << Dmat[i][j] << " ";
        }
        diffoutnew << std::endl;
    }
    diffoutnew.close();


    // MODAL BASIS TEST USING THE NEW IMPLEMENTATION

    Basis* bModalkey = new Basis("GaussLegendre",
                       "Modal",
                       zq,wq,P,nq);

    std::vector<std::vector<double> > DmatModal = bModalkey->GetD();
    std::vector<std::vector<double> > BmatModal = bModalkey->GetB();

    ofstream soloutnewmodal;
    soloutnewmodal.open("new_modal_basis.out");
    for(int j=0;j<nq;j++)
    {
        soloutnewmodal << zq[j] << " ";
    }
    soloutnewmodal << std::endl;

    for(int i=0;i<(P+1);i++)
    {
        for(int j=0;j<nq;j++)
        {
            soloutnewmodal << BmatModal[i][j] << " ";
        }
        soloutnewmodal << std::endl;
    }

    soloutnewmodal.close();


    ofstream diffoutnewmodal;
    diffoutnewmodal.open("new_diff_modal_basis.out");
    for(int i=0;i<(P+1);i++)
    {
        for(int j=0;j<nq;j++)
        {
            diffoutnewmodal << DmatModal[i][j] << " ";
        }
        diffoutnewmodal << std::endl;
    }
    diffoutnewmodal.close();

}









std::vector<double> ForwardTransform(int P, 
                                     std::vector<std::vector<double> > basis, 
                                     std::vector<double>wquad, int nq,
                                     double J, 
                                     std::vector<double> input_quad)
{
    
    std::vector<double> coeff(P+1);
    int ncoeffs     = P+1;
    // double *Icoeff  = dvector(ncoeffs);
    std::vector<double> Icoeff(ncoeffs);
    for(int j=0;j<P+1;j++)
    {
        std::vector<double> phi1 = basis[j];

        Icoeff[j] = J*integr(nq, wquad.data(), phi1.data(), input_quad.data());
    }

    std::vector<double> MassMatElem = GetElementMassMatrix(P,basis,wquad,J);

    int ONE_INT=1;
    double ONE_DOUBLE=1.0;
    double ZERO_DOUBLE=0.0;
    unsigned char TR = 'T';
    int INFO;
    int LWORK = ncoeffs*ncoeffs;
    double *WORK = new double[LWORK];
    // int *ip = ivector(ncoeffs);
    std::vector<int> ip(ncoeffs);
    // Create inverse Mass matrix.
    dgetrf_(&ncoeffs, &ncoeffs, MassMatElem.data(), &ncoeffs, ip.data(), &INFO);
    dgetri_(&ncoeffs, MassMatElem.data(), &ncoeffs, ip.data(), WORK, &LWORK, &INFO);
    // Apply InvMass to Icoeffs hence M^-1 Icoeff = uhat
    dgemv_(&TR,&ncoeffs,&ncoeffs,&ONE_DOUBLE,MassMatElem.data(),&ncoeffs,Icoeff.data(),&ONE_INT,&ZERO_DOUBLE,coeff.data(),&ONE_INT);
    return coeff;
}







std::vector<double> BackwardTransform(int P, 
                                      int nq, 
                                      std::vector<std::vector<double> > basis,  
                                      std::vector<double> input_coeff)
{

    std::vector<double> quad(nq,0.0);
    double sum = 0.0;
    for(int i = 0;i<P+1;i++)
    {
        std::vector<double> phi1 =basis[i];
        for( int j=0;j<nq;j++)
        {
            quad[j] = quad[j]+input_coeff[i]*phi1[j];
        }
    }

    return quad;
}




double EvaluateFromNodalBasis(int P, 
                              double xref,
                              std::vector<double> coeff,
                              std::vector<double> zquad,
                              std::string ptype)
{

    double res = 0.0;
    int numModes = P + 1;

    if(ptype == "GL")
    {
        for(int i=0;i<(P+1);i++)
        {
            res = res + coeff[i]*hgj(i, xref, zquad.data(), numModes, 0.0, 0.0);
        }
    }
    if(ptype == "GLL")
    {
        for(int i=0;i<(P+1);i++)
        {
            res = res + coeff[i]*hglj(i, xref, zquad.data(), numModes, 0.0, 0.0);
        }
    }


    return res;
}















void run_nodal_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    int nq, int P, int O, std::string ptype)
{

    // std::vector<double> X_DG_test(np,0.0);
    // std::vector<double> U_DG_test(np,0.0);

    std::cout << "Running nodal test" << std::endl;

    ofstream solout;
    solout.open("nodal_basis.out");
    std::vector<double> zplot(O*nq,0.0);
    std::vector<double> wplot(O*nq,0.0);
    
    double** Dplot          = dmatrix(O*nq);
    double** Dplott         = dmatrix(O*nq);

    if(ptype == "GL")
    {
        zwgl(zplot.data(), wplot.data(), O*nq);
        Dgl(Dplot, Dplott, zplot.data(), O*nq);

    }
    if(ptype == "GLL")
    {
        zwgll(zplot.data(), wplot.data(), O*nq);
        Dgll(Dplot, Dplott, zplot.data(), O*nq);
    }
    
    std::vector<std::vector<double> > basis_plot = getNodalBasisEval(zplot, z, P, ptype);
    
    for(int i=0;i<basis_plot.size();i++)
    {


        std::vector<double> basis_plot_diff(basis_plot[i].size(),0.0);
        diff( O*nq, Dplot, basis_plot[i].data(), basis_plot_diff.data(), 1.0);


        for(int j=0;j<basis_plot[i].size();j++)
        {
                solout << zplot[j] << " " << basis_plot[i][j] << " " << basis_plot_diff[j] << endl;
                // /std::cout << zplot[j] << std::endl;
        }
    }

    solout.close();

    std::vector<double> quad_eq0(nq,0.5);

    std::vector<double> coeff_eq0 = ForwardTransform(P, basis_plot, wplot, nq, 1.0, quad_eq0);

    for(int i=0;i<coeff_eq0.size();i++)
    {
        std::cout << "nodal coeffs " << coeff_eq0[i] << std::endl;
    }

    ofstream solout2;
    solout2.open("nodal_points.out");
    for(int j=0;j<z.size();j++)
    {
            solout2 << z[j] << endl;
    }
    solout2.close();

    // std::vector<std::vector<double> > basis_m = getNodalBasis(z, nq, np, P);
    // std::vector<std::vector<double> > basis_m = getNodalBasisEvalNew(z, z, P);

    // chi(np, 0, x, z.data(), Jac, bound);
    /*
    for(int i=0;i<np;i++)
    {
        // construct global coordinates for each quadrature point.
        X_DG_test [i]    = x[i];
        // construct initial solution at each quadrature point.
        U_DG_test [i]    = 2.0+sin(2.0*M_PI*x[i]);
    }

    std::vector<double> coeff_e = ForwardTransform(P, np, basis_m, w, nq, Jac[0], U_DG_test);
    
    std::vector<double> quad_e  = BackwardTransform(P, np,  basis_m,  coeff_e);

    double L2norm = 0.0;
    for(int i=0;i<quad_e.size();i++)
    {
        L2norm = L2norm+(quad_e[i]-U_DG_test[i])*(quad_e[i]-U_DG_test[i]);
    }
    if(abs(L2norm)<1.0e-12)
    {
        std::cout << "Nodal basis Fwd and Bwd test PASSED :: L2norm = " << L2norm << std::endl; 
    }
    else
    {
        std::cout << "Nodal basis Fwd and Bwd test FAILED :: L2norm =" << L2norm << std::endl; 
    }
    */
}








void run_left_radau_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    int nq, int P, int O, std::string ptype)
{
    std::cout << "Running Gauss-Radau-Legendre Minus " << std::endl;

    ofstream solout2;
    solout2.open("radaum_basis.out");

    std::vector<double> zradaum(nq,0.0);
    std::vector<double> wradaum(nq,0.0);
    zwgrjm(zradaum.data(), wradaum.data(), nq, 0.0, 0.0);

    std::vector<double> zradaumplot(O*nq,0.0);
    std::vector<double> wradaumplot(O*nq,0.0);
    zwgrjm(zradaumplot.data(), wradaumplot.data(), O*nq, 0.0, 0.0);
    double** Drmplot          = dmatrix(O*nq);
    double** Drmplott         = dmatrix(O*nq);
    Dgrlm(Drmplot, Drmplott, zradaumplot.data(), O*nq);

    std::vector<std::vector<double> > basisradaum_plot = getRadauMinusBasisEval(zradaumplot, zradaumplot, nq, zradaumplot.size(), P);
    //std::vector<std::vector<double> > basisradau_plot = getRadauMinusBasisEvalV2(zradaumplot, zradaum, nq, 10*np, P);
    
    for(int i=0;i<basisradaum_plot.size();i++)
    {   
        //std::cout << "basisradaum_plot[i].size() " << basisradaum_plot[i].size() << "  " << zradaumplot.size() << std::endl;
        std::vector<double> basisradaum_plot_diff(basisradaum_plot[i].size(),0.0);
        diff( O*nq, Drmplot, basisradaum_plot[i].data(), basisradaum_plot_diff.data(), 1.0);

        for(int j=0;j<basisradaum_plot[i].size();j++)
        {
                solout2 << zradaumplot[j] << " " << basisradaum_plot[i][j] << " " << basisradaum_plot_diff[j] << endl;
        }
    }

    solout2.close();
}




void run_right_radau_test(std::vector<double> zq, 
                    std::vector<double> z, 
                    int nq, int P, int O, std::string ptype)
{
    std::cout << "Running Gauss-Radau-Legendre Plus " << std::endl;

    ofstream solout2;
    solout2.open("radaup_basis.out");

    std::vector<double> zradaup(nq,0.0);
    std::vector<double> wradaup(nq,0.0);
    zwgrjp(zradaup.data(), wradaup.data(), nq, 0.0, 0.0);

    std::vector<double> zradaupplot(O*nq,0.0);
    std::vector<double> wradaupplot(O*nq,0.0);
    zwgrjp(zradaupplot.data(), wradaupplot.data(), O*nq, 0.0, 0.0);
    double** Drpplot          = dmatrix(O*nq);
    double** Drpplott         = dmatrix(O*nq);
    Dgrlm(Drpplot, Drpplott, zradaupplot.data(), O*nq);

    std::vector<std::vector<double> > basisradaup_plot = getRadauPlusBasisEval(zradaupplot, zradaupplot, nq, zradaupplot.size(), P);
    //std::vector<std::vector<double> > basisradau_plot = getRadauMinusBasisEvalV2(zradaumplot, zradaum, nq, 10*np, P);
    
    for(int i=0;i<basisradaup_plot.size();i++)
    {   
        //std::cout << "basisradaup_plot[i].size() " << basisradaup_plot[i].size() << "  " << zradaupplot.size() << std::endl;
        std::vector<double> basisradaup_plot_diff(basisradaup_plot[i].size(),0.0);
        diff( O*nq, Drpplot, basisradaup_plot[i].data(), basisradaup_plot_diff.data(), 1.0);

        for(int j=0;j<basisradaup_plot[i].size();j++)
        {
                solout2 << zradaupplot[j] << " " << basisradaup_plot[i][j] << " " << basisradaup_plot_diff[j] << endl;
        }
    }

    solout2.close();
}



void run_nodal_test_new(std::vector<double> zq,
                    std::vector<double> z, 
                    std::vector<double> w, int P, int O)
{

    // std::vector<double> X_DG_test(np,0.0);
    // std::vector<double> U_DG_test(np,0.0);

    std::cout << "Running nodal " << std::endl;
    int np = z.size();
    ofstream solout;
    solout.open("nodal_basis.out");

    std::vector<double> zplot(O*np,0.0);
    std::vector<double> wplot(O*np,0.0);
    zwgll(zplot.data(), wplot.data(), O*np);
    std::vector<std::vector<double> > basis_plot = getNodalBasisEvalNew(zplot, z, P);

    
    
    for(int i=0;i<basis_plot.size();i++)
    {
        // std::vector<double> basis_plot_diff(basis_plot[i].size(),0.0);
        // diff( O*nq, Drmplot, basis_plot[i].data(), basis_plot_diff.data(), 1.0);

        for(int j=0;j<basis_plot[i].size();j++)
        {
                solout << zplot[j] << " " << basis_plot[i][j] << endl;
        }
    }

    solout.close();

    // std::vector<std::vector<double> > basis_m = getNodalBasis(z, nq, np, P);
    // std::vector<std::vector<double> > basis_m = getNodalBasisEvalNew(zq, z, P);

    std::vector<double> quad_eq0(np,1.0);

    std::vector<double> coeff_eq0 = ForwardTransform(P, basis_plot, wplot, np, 1.0, quad_eq0);

    for(int i=0;i<coeff_eq0.size();i++)
    {
        std::cout << "modal coeffs " << coeff_eq0[i] << std::endl;
    }


    // chi(np, 0, x, z.data(), Jac, bound);
    /*
    for(int i=0;i<np;i++)
    {
        // construct global coordinates for each quadrature point.
        X_DG_test [i]    = x[i];
        // construct initial solution at each quadrature point.
        U_DG_test [i]    = 2.0+sin(2.0*M_PI*x[i]);
    }

    std::vector<double> coeff_e = ForwardTransform(P, np, basis_m, w, nq, Jac[0], U_DG_test);
    
    std::vector<double> quad_e  = BackwardTransform(P, np,  basis_m,  coeff_e);

    double L2norm = 0.0;
    for(int i=0;i<quad_e.size();i++)
    {
        L2norm = L2norm+(quad_e[i]-U_DG_test[i])*(quad_e[i]-U_DG_test[i]);
    }
    if(abs(L2norm)<1.0e-12)
    {
        std::cout << "Nodal basis Fwd and Bwd test PASSED :: L2norm = " << L2norm << std::endl; 
    }
    else
    {
        std::cout << "Nodal basis Fwd and Bwd test FAILED :: L2norm =" << L2norm << std::endl; 
    }
    */
}





void run_modal_test(std::vector<double> zq,
                    std::vector<double> z,  
                    std::vector<double> w, 
                    double* bound, 
                    double* Jac, 
                    int np, int nq, int P, int O, std::string ptype)
{

    // std::vector<double> X_DG_test(np,0.0);
    // std::vector<double> U_DG_test(np,0.0);
    std::cout << "Running modal " << std::endl;

    ofstream solout;
    solout.open("modal_basis.out");

    std::vector<double> zplot(O*nq,0.0);
    std::vector<double> wplot(O*nq,0.0);

    double** Dplot          = dmatrix(O*nq);
    double** Dplott         = dmatrix(O*nq);

    if(ptype == "GL")
    {
        zwgl(zplot.data(), wplot.data(), O*nq);
        Dgl(Dplot, Dplott, zplot.data(), O*nq);
    }
    if(ptype == "GLL")
    {
        zwgll(zplot.data(), wplot.data(), O*nq);
        Dgll(Dplot, Dplott, zplot.data(), O*nq);
    }
    
    // std::vector<std::vector<double> > basis_plot = getLegendreBasisEval(zplot, zplot, nq, zplot.size(), P);
    std::vector<std::vector<double> > basis_plot = getModalBasisEval(zplot, zplot, O*nq, P, ptype);
    //std::cout << O*nq << " " << zplot.size() << std::endl;
    for(int i=0;i<basis_plot.size();i++)
    {
        std::vector<double> basis_plot_diff(basis_plot[i].size(),0.0);
        diff(O*np, Dplot, basis_plot[i].data(), basis_plot_diff.data(), 1.0);

        for(int j=0;j<basis_plot[i].size();j++)
        {
                solout << zplot[j] << " " << basis_plot[i][j] << " " << basis_plot_diff[j] << endl;
        }
    }
    
    solout.close();

    std::vector<std::vector<double> > basis_m = getModalBasisEval(z, z, nq, P, ptype);

    std::vector<double> MassMatElem = GetElementMassMatrix(P,basis_m,w,Jac[0]);


     std::vector<double> quad_eq0(nq,0.5);

    std::vector<double> coeff_eq0 = ForwardTransform(P, basis_plot, wplot, nq, 1.0, quad_eq0);

    for(int i=0;i<coeff_eq0.size();i++)
    {
        std::cout << "modal coeffs " << coeff_eq0[i] << std::endl;
    }


    ofstream solout2;
    solout2.open("modal_points.out");
    for(int j=0;j<z.size();j++)
    {
            solout2 << z[j] << endl;
    }
    solout2.close();

    // std::cout << "M=[";
    // for(int i=0;i<P+1;i++)
    // {
    //     std::cout << "[";
    //     for(int j=0;j<P+1;j++)
    //     {
    //         if(j<P)
    //         {
    //             std::cout << MassMatElem[i*(P+1)+j] << ", ";
    //         }
    //         else
    //         {
    //             std::cout << MassMatElem[i*(P+1)+j];
    //         }
            
    //     }
    //     if(i<P)
    //     {
    //         std::cout <<"],"<< std::endl;
    //     }
    //     else
    //     {
    //         std::cout <<"]]"<< std::endl;
    //     }
        
    // }

    /*
    std::cout << "Running Gauss-Radau-Legendre Minus " << std::endl;

    ofstream solout2;
    solout2.open("radaum_basis.out");

    std::vector<double> zradaum(np,0.0);
    std::vector<double> wradaum(np,0.0);
    zwgrjm(zradaum.data(), wradaum.data(), np, 0.0, 0.0);

    std::vector<double> zradaumplot(10*np,0.0);
    std::vector<double> wradaumplot(10*np,0.0);
    zwgrjm(zradaumplot.data(), wradaumplot.data(), 10*np, 0.0, 0.0);
    double** Drmplot          = dmatrix(10*np);
    double** Drmplott         = dmatrix(10*np);
    Dgrlm(Drmplot, Drmplott, zradaumplot.data(), 10*np);

    std::vector<std::vector<double> > basisradaum_plot = getRadauMinusBasisEval(zradaumplot, zradaumplot, nq, zradaumplot.size(), P);
    //std::vector<std::vector<double> > basisradau_plot = getRadauMinusBasisEvalV2(zradaumplot, zradaum, nq, 10*np, P);
    
    for(int i=0;i<basisradaum_plot.size();i++)
    {   
        std::cout << "basisradaum_plot[i].size() " << basisradaum_plot[i].size() << "  " << zradaumplot.size() << std::endl;
        std::vector<double> basisradaum_plot_diff(basisradaum_plot[i].size(),0.0);
        diff( 10*np, Drmplot, basisradaum_plot[i].data(), basisradaum_plot_diff.data(), 1.0);

        for(int j=0;j<basisradaum_plot[i].size();j++)
        {
                solout2 << zradaumplot[j] << " " << basisradaum_plot[i][j] << " " << basisradaum_plot_diff[j] << endl;
        }
    }

    solout2.close();


    ofstream solout30;
    solout30.open("radaum_points.out");
    for(int i=0;i<zradaum.size();i++)
    { 
        solout30 << zradaum[i] << std::endl;
    }
    solout30.close();


   




    ofstream solout3;
    solout3.open("radaup_basis.out");

    std::vector<double> zradaup(np,0.0);
    std::vector<double> wradaup(np,0.0);
    zwgrjp(zradaup.data(), wradaup.data(), np, 0.0, 0.0);

    std::vector<double> zradaupplot(10*np,0.0);
    std::vector<double> wradaupplot(10*np,0.0);
    zwgrjp(zradaupplot.data(), wradaupplot.data(), 10*np, 0.0, 0.0);
    std::vector<std::vector<double> > basisradaup_plot = getRadauPlusBasisEval(zradaupplot, zradaupplot, nq, zplot.size(), P);

    //std::vector<std::vector<double> > basisradau_plot = getRadauMinusBasisEvalV2(zradaumplot, zradaum, nq, 10*np, P);
    
    for(int i=0;i<basisradaup_plot.size();i++)
    {   
        std::cout << "basisradaup_plot[i].size() " << basisradaup_plot[i].size() << "  " << zradaumplot.size() << std::endl;
        for(int j=0;j<basisradaup_plot[i].size();j++)
        {
                solout3 << zradaumplot[j] << " " << basisradaup_plot[i][j] << endl;
        }
    }

    solout3.close();
    
     ofstream solout40;
    solout40.open("radaup_points.out");
    for(int i=0;i<zradaup.size();i++)
    { 
        solout40 << zradaup[i] << std::endl;
    }
    solout40.close();
    */
    
    /*
    for(int i=0;i<np;i++)
    {
        // construct global coordinates for each quadrature point.
        X_DG_test [i]    = x[i];
        // construct initial solution at each quadrature point.
        U_DG_test [i]    = 2.0+sin(2.0*M_PI*x[i]);
    }

    std::vector<double> coeff_e = ForwardTransform(P, np, basis_m, w, nq, Jac[0], U_DG_test);
    
    std::vector<double> quad_e  = BackwardTransform(P, np,  basis_m,  coeff_e);
    double L2norm = 0.0;
    for(int i=0;i<quad_e.size();i++)
    {
        L2norm = L2norm+(quad_e[i]-U_DG_test[i])*(quad_e[i]-U_DG_test[i]);
    }

    if(abs(L2norm)<1.0e-12)
    {
        std::cout << "Modal basis Fwd and Bwd test PASSED :: L2norm = " << L2norm << std::endl; 
    }
    else
    {
        std::cout << "Modal basis Fwd and Bwd test FAILED :: L2norm =" << L2norm << std::endl; 
    }
    */
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

