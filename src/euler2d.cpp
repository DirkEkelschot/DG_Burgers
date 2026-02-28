#include "euler2d.h"
#include "geom2d.h"
#include "basis_functions.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

extern "C" {
    extern void dgetrf_(int*, int*, double*, int*, int[], int*);
    extern void dgetrs_(unsigned char*, int*, int*, double*, int*, int[], double[], int*, int*);
}

// Reference-space coordinates on each local face, parameterised by s in [-1,1]
static void faceRefCoords(int localFace, double s, double& xi, double& eta)
{
    switch (localFace)
    {
        case 0: xi =  s;    eta = -1.0; break;
        case 1: xi =  1.0;  eta =  s;   break;
        case 2: xi = -s;    eta =  1.0; break;
        case 3: xi = -1.0;  eta = -s;   break;
        default: xi = 0; eta = 0; break;
    }
}

void eulerFluxX(double rho, double rhou, double rhov, double rhoE,
                double F[NVAR2D])
{
    double u = rhou / rho;
    double v = rhov / rho;
    double p = pressure2D(rho, rhou, rhov, rhoE);
    F[0] = rhou;
    F[1] = rhou * u + p;
    F[2] = rhou * v;
    F[3] = (rhoE + p) * u;
}

void eulerFluxY(double rho, double rhou, double rhov, double rhoE,
                double G[NVAR2D])
{
    double u = rhou / rho;
    double v = rhov / rho;
    double p = pressure2D(rho, rhou, rhov, rhoE);
    G[0] = rhov;
    G[1] = rhov * u;
    G[2] = rhov * v + p;
    G[3] = (rhoE + p) * v;
}

void laxFriedrichsFlux2D(const double UL[NVAR2D], const double UR[NVAR2D],
                         double nx, double ny,
                         double Fnum[NVAR2D])
{
    double FL[NVAR2D], GL[NVAR2D], FR[NVAR2D], GR[NVAR2D];
    eulerFluxX(UL[0], UL[1], UL[2], UL[3], FL);
    eulerFluxY(UL[0], UL[1], UL[2], UL[3], GL);
    eulerFluxX(UR[0], UR[1], UR[2], UR[3], FR);
    eulerFluxY(UR[0], UR[1], UR[2], UR[3], GR);

    // Normal fluxes
    double HnL[NVAR2D], HnR[NVAR2D];
    for (int n = 0; n < NVAR2D; ++n)
    {
        HnL[n] = FL[n] * nx + GL[n] * ny;
        HnR[n] = FR[n] * nx + GR[n] * ny;
    }

    // Roe-averaged wavespeed for the dissipation coefficient
    double rhoL = UL[0], uL = UL[1]/UL[0], vL = UL[2]/UL[0];
    double pL   = pressure2D(UL[0], UL[1], UL[2], UL[3]);
    double HL   = (UL[3] + pL) / rhoL;

    double rhoR = UR[0], uR = UR[1]/UR[0], vR = UR[2]/UR[0];
    double pR   = pressure2D(UR[0], UR[1], UR[2], UR[3]);
    double HR   = (UR[3] + pR) / rhoR;

    double srL  = std::sqrt(rhoL);
    double srR  = std::sqrt(rhoR);
    double srLR = srL + srR;

    double uRoe = (srL * uL + srR * uR) / srLR;
    double vRoe = (srL * vL + srR * vR) / srLR;
    double HRoe = (srL * HL + srR * HR) / srLR;
    double VnRoe = uRoe * nx + vRoe * ny;
    double qRoe2 = uRoe * uRoe + vRoe * vRoe;
    double cRoe  = std::sqrt(std::max((GAMMA - 1.0) * (HRoe - 0.5 * qRoe2), 1e-14));

    double alpha = std::abs(VnRoe) + cRoe;

    for (int n = 0; n < NVAR2D; ++n)
        Fnum[n] = 0.5 * (HnL[n] + HnR[n]) - 0.5 * alpha * (UR[n] - UL[n]);
}

// ============================================================================
// Mass matrix assembly and factorisation
// ============================================================================

void assembleAndFactorMassMatrices(
    const Mesh2D& mesh,
    const GeomData2D& geom,
    const std::vector<std::vector<double>>& Bxi,
    const std::vector<std::vector<double>>& Beta,
    const std::vector<double>& wxi,
    const std::vector<double>& weta,
    int P, int nq1d,
    std::vector<double>& massLU,
    std::vector<int>& massPiv)
{
    int nmodes  = (P + 1) * (P + 1);
    int nqVol   = nq1d * nq1d;
    int nE      = mesh.nElements;
    int blockSz = nmodes * nmodes;

    massLU.assign(nE * blockSz, 0.0);
    massPiv.assign(nE * nmodes, 0);

    for (int e = 0; e < nE; ++e)
    {
        double* M = &massLU[e * blockSz];

        // M_{(i1*neta+j1), (i2*neta+j2)} =
        //   sum_q w[q] * phi_i1(xi_q)*phi_j1(eta_q) * phi_i2(xi_q)*phi_j2(eta_q) * detJ
        for (int qx = 0; qx < nq1d; ++qx)
        {
            for (int qe = 0; qe < nq1d; ++qe)
            {
                int qIdx = e * nqVol + qx * nq1d + qe;
                double w = wxi[qx] * weta[qe] * geom.detJ[qIdx];

                for (int i1 = 0; i1 < P + 1; ++i1)
                {
                    for (int j1 = 0; j1 < P + 1; ++j1)
                    {
                        int row = i1 * (P + 1) + j1;
                        double phiRow = Bxi[i1][qx] * Beta[j1][qe];

                        for (int i2 = 0; i2 < P + 1; ++i2)
                        {
                            for (int j2 = 0; j2 < P + 1; ++j2)
                            {
                                int col = i2 * (P + 1) + j2;
                                double phiCol = Bxi[i2][qx] * Beta[j2][qe];
                                M[row * nmodes + col] += w * phiRow * phiCol;
                            }
                        }
                    }
                }
            }
        }

        // LU-factorise the block
        int INFO;
        dgetrf_(&nmodes, &nmodes, M, &nmodes, &massPiv[e * nmodes], &INFO);
        if (INFO != 0)
            std::cerr << "Mass matrix LU failed for element " << e
                      << " INFO=" << INFO << std::endl;
    }
}

// ============================================================================
// Mass matrix inverse (for GPU solver)
// ============================================================================

void computeMassInverse(
    const std::vector<double>& massLU,
    const std::vector<int>& massPiv,
    int nE, int nmodes,
    std::vector<double>& Minv)
{
    int blockSz = nmodes * nmodes;
    Minv.assign(nE * blockSz, 0.0);

    for (int e = 0; e < nE; ++e)
    {
        double* Mblock = &Minv[e * blockSz];
        for (int i = 0; i < nmodes; ++i)
            Mblock[i * nmodes + i] = 1.0;

        std::vector<double> LUcopy(blockSz);
        std::vector<int> pivCopy(nmodes);
        std::memcpy(LUcopy.data(), &massLU[e * blockSz], blockSz * sizeof(double));
        std::memcpy(pivCopy.data(), &massPiv[e * nmodes], nmodes * sizeof(int));

        unsigned char TRANS = 'N';
        int N = nmodes, NRHS = nmodes, LDA = nmodes, LDB = nmodes, INFO;
        dgetrs_(&TRANS, &N, &NRHS, LUcopy.data(), &LDA,
                pivCopy.data(), Mblock, &LDB, &INFO);

        if (INFO != 0)
            std::cerr << "Mass matrix inverse failed for element " << e
                      << " INFO=" << INFO << std::endl;
    }
}

// ============================================================================
// Full DG RHS
// ============================================================================

void computeDGRHS2D(const Mesh2D& mesh,
                    const GeomData2D& geom,
                    BasisPoly* basis1D,
                    int P, int nq1d, int nqFace,
                    const std::vector<double>& wq1d,
                    const std::vector<double>& wqFace,
                    const std::vector<double>& zq1d,
                    const std::vector<double>& zqFace,
                    const std::vector<std::vector<double>>& U,
                    std::vector<std::vector<double>>& R,
                    const std::vector<double>& massLU,
                    const std::vector<int>& massPiv,
                    int nModesPerElem,
                    double time,
                    const std::map<int, BoundaryStateFunc>& bcMap)
{
    int nE    = mesh.nElements;
    int nqVol = nq1d * nq1d;
    int nmodes = nModesPerElem;  // (P+1)^2

    std::vector<std::vector<double>> Bmat = basis1D->GetB();
    std::vector<std::vector<double>> Dmat = basis1D->GetD();
    std::vector<std::vector<double>> blr  = basis1D->GetLeftRightBasisValues();

    // Initialise R
    for (int v = 0; v < NVAR2D; ++v)
        R[v].assign(nE * nqVol, 0.0);

    // ========================================================================
    // Step 1: Forward transform U from quadrature to coefficients per element
    // ========================================================================
    // Coefficients: coeff[var][e * nmodes + (i*(P+1)+j)]
    std::vector<std::vector<double>> Ucoeff(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v)
        Ucoeff[v].assign(nE * nmodes, 0.0);

    for (int e = 0; e < nE; ++e)
    {
        for (int v = 0; v < NVAR2D; ++v)
        {
            for (int i = 0; i < P + 1; ++i)
            {
                for (int j = 0; j < P + 1; ++j)
                {
                    int mIdx = e * nmodes + i * (P + 1) + j;
                    double val = 0.0;
                    for (int qx = 0; qx < nq1d; ++qx)
                    {
                        for (int qe = 0; qe < nq1d; ++qe)
                        {
                            int qIdx = e * nqVol + qx * nq1d + qe;
                            double w = wq1d[qx] * wq1d[qe] * geom.detJ[qIdx];
                            double phi = Bmat[i][qx] * Bmat[j][qe];
                            val += w * phi * U[v][qIdx];
                        }
                    }
                    Ucoeff[v][mIdx] = val;
                }
            }
        }

        // Solve M * uhat = rhs  =>  uhat = M^{-1} rhs
        for (int v = 0; v < NVAR2D; ++v)
        {
            std::vector<double> rhs(nmodes);
            for (int m = 0; m < nmodes; ++m)
                rhs[m] = Ucoeff[v][e * nmodes + m];

            unsigned char TRANS = 'N';
            int NRHS = 1, INFO;
            std::vector<double> Mcopy(nmodes * nmodes);
            std::vector<int> pivCopy(nmodes);
            std::memcpy(Mcopy.data(), &massLU[e * nmodes * nmodes],
                        nmodes * nmodes * sizeof(double));
            std::memcpy(pivCopy.data(), &massPiv[e * nmodes],
                        nmodes * sizeof(int));

            dgetrs_(&TRANS, &nmodes, &NRHS, Mcopy.data(), &nmodes,
                    pivCopy.data(), rhs.data(), &nmodes, &INFO);

            for (int m = 0; m < nmodes; ++m)
                Ucoeff[v][e * nmodes + m] = rhs[m];
        }
    }

    // ========================================================================
    // Step 2: Compute Euler fluxes at all volume quadrature points
    // ========================================================================
    std::vector<double> Fx(nE * nqVol * NVAR2D);
    std::vector<double> Fy(nE * nqVol * NVAR2D);

    for (int e = 0; e < nE; ++e)
    {
        for (int q = 0; q < nqVol; ++q)
        {
            int idx = e * nqVol + q;
            double rho  = U[0][idx];
            double rhou = U[1][idx];
            double rhov = U[2][idx];
            double rhoE = U[3][idx];

            double Fq[NVAR2D], Gq[NVAR2D];
            eulerFluxX(rho, rhou, rhov, rhoE, Fq);
            eulerFluxY(rho, rhou, rhov, rhoE, Gq);

            for (int n = 0; n < NVAR2D; ++n)
            {
                Fx[idx * NVAR2D + n] = Fq[n];
                Fy[idx * NVAR2D + n] = Gq[n];
            }
        }
    }

    // ========================================================================
    // Step 3: Volume integral
    // ========================================================================
    // RHS_coeff[var][e * nmodes + mode] =
    //   sum_q  w[q] * (Fx * dphi/dx + Fy * dphi/dy) * detJ
    // where dphi/dx = dxidx * dphi/dxi + detadx * dphi/deta  (etc)
    std::vector<std::vector<double>> rhsCoeff(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v)
        rhsCoeff[v].assign(nE * nmodes, 0.0);

    for (int e = 0; e < nE; ++e)
    {
        for (int qx = 0; qx < nq1d; ++qx)
        {
            for (int qe = 0; qe < nq1d; ++qe)
            {
                int qIdx = e * nqVol + qx * nq1d + qe;
                double w   = wq1d[qx] * wq1d[qe] * geom.detJ[qIdx];
                double jxx = geom.dxidx[qIdx];
                double jxy = geom.dxidy[qIdx];
                double jyx = geom.detadx[qIdx];
                double jyy = geom.detady[qIdx];

                for (int n = 0; n < NVAR2D; ++n)
                {
                    double fx = Fx[qIdx * NVAR2D + n];
                    double fy = Fy[qIdx * NVAR2D + n];

                    for (int i = 0; i < P + 1; ++i)
                    {
                        for (int j = 0; j < P + 1; ++j)
                        {
                            // dphi/dxi * Bmat_eta  and  Bmat_xi * dphi/deta
                            double dphidxi  = Dmat[i][qx] * Bmat[j][qe];
                            double dphideta = Bmat[i][qx] * Dmat[j][qe];

                            double dphidx = jxx * dphidxi + jyx * dphideta;
                            double dphidy = jxy * dphidxi + jyy * dphideta;

                            int mIdx = e * nmodes + i * (P + 1) + j;
                            rhsCoeff[n][mIdx] += w * (fx * dphidx + fy * dphidy);
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Step 4: Surface integral – compute numerical flux and subtract from RHS
    // ========================================================================
    // Evaluate traces: solution at each face quadrature point from left and right
    int nF = mesh.nFaces;

    for (int f = 0; f < nF; ++f)
    {
        int eL = mesh.faces[f].elemL;
        int eR = mesh.faces[f].elemR;
        int lfL = mesh.faces[f].faceL;
        int lfR = mesh.faces[f].faceR;

        for (int q = 0; q < nqFace; ++q)
        {
            double s = zqFace[q];

            // Left element trace
            double xiL, etaL;
            faceRefCoords(lfL, s, xiL, etaL);

            double UL[NVAR2D] = {0, 0, 0, 0};
            for (int v = 0; v < NVAR2D; ++v)
            {
                for (int i = 0; i < P + 1; ++i)
                {
                    for (int j = 0; j < P + 1; ++j)
                    {
                        int mIdx = eL * nmodes + i * (P + 1) + j;
                        double phiXi, phiEta;

                        // Evaluate 1D basis at the face coordinate
                        phiXi = 0.0;
                        for (int k = 0; k < nq1d; ++k)
                            phiXi += (zq1d[k] == xiL) ? Bmat[i][k] : 0.0;

                        phiEta = 0.0;
                        for (int k = 0; k < nq1d; ++k)
                            phiEta += (zq1d[k] == etaL) ? Bmat[j][k] : 0.0;

                        // Better: evaluate basis at arbitrary point using blr or direct eval
                        // The face sits at xi or eta = +/-1, so use left/right values
                        if (lfL == 0)       { phiXi = Bmat[i][q]; phiEta = blr[j][0]; } // eta=-1
                        else if (lfL == 1)  { phiXi = blr[i][1];  phiEta = Bmat[j][q]; } // xi=+1
                        else if (lfL == 2)  { phiXi = Bmat[i][nqFace - 1 - q]; phiEta = blr[j][1]; } // eta=+1, reversed
                        else if (lfL == 3)  { phiXi = blr[i][0];  phiEta = Bmat[j][nqFace - 1 - q]; } // xi=-1, reversed

                        UL[v] += Ucoeff[v][mIdx] * phiXi * phiEta;
                    }
                }
            }

            // Right element trace (or boundary ghost)
            double UR[NVAR2D];
            bool isBoundary = (eR < 0);

            if (!isBoundary)
            {
                double xiR, etaR;
                faceRefCoords(lfR, s, xiR, etaR);

                // The face parametrisation s runs in opposite directions
                // for L and R elements sharing the same physical face.
                // We need to reverse q for the right element.
                int qR = nqFace - 1 - q;

                for (int v = 0; v < NVAR2D; ++v)
                {
                    UR[v] = 0.0;
                    for (int i = 0; i < P + 1; ++i)
                    {
                        for (int j = 0; j < P + 1; ++j)
                        {
                            int mIdx = eR * nmodes + i * (P + 1) + j;
                            double phiXi, phiEta;

                            if (lfR == 0)       { phiXi = Bmat[i][qR]; phiEta = blr[j][0]; }
                            else if (lfR == 1)  { phiXi = blr[i][1];   phiEta = Bmat[j][qR]; }
                            else if (lfR == 2)  { phiXi = Bmat[i][nqFace - 1 - qR]; phiEta = blr[j][1]; }
                            else if (lfR == 3)  { phiXi = blr[i][0];   phiEta = Bmat[j][nqFace - 1 - qR]; }
                            else { phiXi = 0; phiEta = 0; }

                            UR[v] += Ucoeff[v][mIdx] * phiXi * phiEta;
                        }
                    }
                }
            }
            else
            {
                int fIdx = f * nqFace + q;
                double xf = geom.faceXPhys[fIdx];
                double yf = geom.faceYPhys[fIdx];
                double nx = geom.faceNx[fIdx];
                double ny = geom.faceNy[fIdx];
                auto it = bcMap.find(mesh.faces[f].bcTag);
                if (it != bcMap.end())
                {
                    it->second(UL, nx, ny, xf, yf, time, UR);
                }
                else
                {
                    for (int v = 0; v < NVAR2D; ++v)
                        UR[v] = UL[v];
                }
            }

            int fIdx = f * nqFace + q;
            double nx = geom.faceNx[fIdx];
            double ny = geom.faceNy[fIdx];

            double Fnum[NVAR2D];
            laxFriedrichsFlux2D(UL, UR, nx, ny, Fnum);

            // Subtract surface integral from rhsCoeff for left element
            {
                double wf = wqFace[q] * geom.faceJac[fIdx];
                for (int n = 0; n < NVAR2D; ++n)
                {
                    for (int i = 0; i < P + 1; ++i)
                    {
                        for (int j = 0; j < P + 1; ++j)
                        {
                            double phiXi, phiEta;
                            if (lfL == 0)       { phiXi = Bmat[i][q]; phiEta = blr[j][0]; }
                            else if (lfL == 1)  { phiXi = blr[i][1];  phiEta = Bmat[j][q]; }
                            else if (lfL == 2)  { phiXi = Bmat[i][nqFace - 1 - q]; phiEta = blr[j][1]; }
                            else if (lfL == 3)  { phiXi = blr[i][0];  phiEta = Bmat[j][nqFace - 1 - q]; }
                            else { phiXi = 0; phiEta = 0; }

                            int mIdx = eL * nmodes + i * (P + 1) + j;
                            rhsCoeff[n][mIdx] -= wf * Fnum[n] * phiXi * phiEta;
                        }
                    }
                }
            }

            // Add surface integral contribution to right element (opposite normal)
            if (!isBoundary)
            {
                double wf = wqFace[q] * geom.faceJac[fIdx];
                int qR = nqFace - 1 - q;
                for (int n = 0; n < NVAR2D; ++n)
                {
                    for (int i = 0; i < P + 1; ++i)
                    {
                        for (int j = 0; j < P + 1; ++j)
                        {
                            double phiXi, phiEta;
                            if (lfR == 0)       { phiXi = Bmat[i][qR]; phiEta = blr[j][0]; }
                            else if (lfR == 1)  { phiXi = blr[i][1];   phiEta = Bmat[j][qR]; }
                            else if (lfR == 2)  { phiXi = Bmat[i][nqFace - 1 - qR]; phiEta = blr[j][1]; }
                            else if (lfR == 3)  { phiXi = blr[i][0];   phiEta = Bmat[j][nqFace - 1 - qR]; }
                            else { phiXi = 0; phiEta = 0; }

                            int mIdx = eR * nmodes + i * (P + 1) + j;
                            // Opposite normal for right element: +Fnum instead of -Fnum
                            rhsCoeff[n][mIdx] += wf * Fnum[n] * phiXi * phiEta;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Step 5: Solve M * dU/dt_coeff = rhsCoeff  per element
    // ========================================================================
    for (int e = 0; e < nE; ++e)
    {
        for (int v = 0; v < NVAR2D; ++v)
        {
            std::vector<double> rhs(nmodes);
            for (int m = 0; m < nmodes; ++m)
                rhs[m] = rhsCoeff[v][e * nmodes + m];

            unsigned char TRANS = 'N';
            int NRHS = 1, INFO;
            std::vector<double> Mcopy(nmodes * nmodes);
            std::vector<int> pivCopy(nmodes);
            std::memcpy(Mcopy.data(), &massLU[e * nmodes * nmodes],
                        nmodes * nmodes * sizeof(double));
            std::memcpy(pivCopy.data(), &massPiv[e * nmodes],
                        nmodes * sizeof(int));

            dgetrs_(&TRANS, &nmodes, &NRHS, Mcopy.data(), &nmodes,
                    pivCopy.data(), rhs.data(), &nmodes, &INFO);

            // Backward transform: coefficients -> quadrature values
            for (int qx = 0; qx < nq1d; ++qx)
            {
                for (int qe = 0; qe < nq1d; ++qe)
                {
                    int qIdx = e * nqVol + qx * nq1d + qe;
                    double val = 0.0;
                    for (int i = 0; i < P + 1; ++i)
                    {
                        for (int j = 0; j < P + 1; ++j)
                        {
                            val += rhs[i * (P + 1) + j] * Bmat[i][qx] * Bmat[j][qe];
                        }
                    }
                    R[v][qIdx] = val;
                }
            }
        }
    }
}
