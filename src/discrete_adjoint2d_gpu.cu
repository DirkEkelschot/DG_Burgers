#include "discrete_adjoint2d_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>

#define DA_CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static constexpr double GAMMA_DA = 1.4;

#define DA_MAX_P1   8
#define DA_MAX_NQ1D 10

__constant__ double da_Bmat[DA_MAX_P1 * DA_MAX_NQ1D];
__constant__ double da_Dmat[DA_MAX_P1 * DA_MAX_NQ1D];
__constant__ double da_blr[DA_MAX_P1 * 2];
__constant__ double da_wq[DA_MAX_NQ1D];
__constant__ double da_NodalToModal[DA_MAX_P1 * DA_MAX_P1];

// ============================================================================
// Device helper functions
// ============================================================================

__device__ inline double pressure_da(double rho, double rhou, double rhov, double rhoE)
{
    double u = rhou / rho;
    double v = rhov / rho;
    return (GAMMA_DA - 1.0) * (rhoE - 0.5 * rho * (u * u + v * v));
}

__device__ inline double evalPhiFace_da(int lf, int i, int j, int q,
                                        int P1, int nq1d)
{
    int nqF = nq1d;
    double phiXi, phiEta;
    switch (lf) {
        case 0: phiXi = da_Bmat[i * nq1d + q];             phiEta = da_blr[j * 2 + 0]; break;
        case 1: phiXi = da_blr[i * 2 + 1];                 phiEta = da_Bmat[j * nq1d + q]; break;
        case 2: phiXi = da_Bmat[i * nq1d + (nqF - 1 - q)]; phiEta = da_blr[j * 2 + 1]; break;
        case 3: phiXi = da_blr[i * 2 + 0];                 phiEta = da_Bmat[j * nq1d + (nqF - 1 - q)]; break;
        default: phiXi = 0; phiEta = 0;
    }
    return phiXi * phiEta;
}

// ============================================================================
// Euler flux Jacobians (same as continuous adjoint -- volume term is exact)
// ============================================================================

__device__ void eulerFluxJacX_da(const double U[4], double A[4][4])
{
    const double gm1 = GAMMA_DA - 1.0;
    double rho = U[0], u = U[1]/rho, v = U[2]/rho;
    double q2 = u*u + v*v;
    double p = gm1 * (U[3] - 0.5*rho*q2);
    double H = (U[3] + p) / rho;

    A[0][0] = 0.0;            A[0][1] = 1.0;           A[0][2] = 0.0;      A[0][3] = 0.0;
    A[1][0] = 0.5*gm1*q2-u*u; A[1][1] = (3.0-GAMMA_DA)*u; A[1][2] = -gm1*v; A[1][3] = gm1;
    A[2][0] = -u*v;           A[2][1] = v;             A[2][2] = u;         A[2][3] = 0.0;
    A[3][0] = u*(0.5*gm1*q2-H); A[3][1] = H-gm1*u*u;  A[3][2] = -gm1*u*v;  A[3][3] = GAMMA_DA*u;
}

__device__ void eulerFluxJacY_da(const double U[4], double B[4][4])
{
    const double gm1 = GAMMA_DA - 1.0;
    double rho = U[0], u = U[1]/rho, v = U[2]/rho;
    double q2 = u*u + v*v;
    double p = gm1 * (U[3] - 0.5*rho*q2);
    double H = (U[3] + p) / rho;

    B[0][0] = 0.0;            B[0][1] = 0.0;           B[0][2] = 1.0;      B[0][3] = 0.0;
    B[1][0] = -u*v;           B[1][1] = v;             B[1][2] = u;         B[1][3] = 0.0;
    B[2][0] = 0.5*gm1*q2-v*v; B[2][1] = -gm1*u;       B[2][2] = (3.0-GAMMA_DA)*v; B[2][3] = gm1;
    B[3][0] = v*(0.5*gm1*q2-H); B[3][1] = -gm1*u*v;   B[3][2] = H-gm1*v*v; B[3][3] = GAMMA_DA*v;
}

// ============================================================================
// Lax-Friedrichs numerical flux (duplicated from forward solver for FD)
// ============================================================================

__device__ void laxFriedrichs_da(const double UL[4], const double UR[4],
                                 double nx, double ny, double Fnum[4])
{
    double rhoL = UL[0], uL = UL[1]/UL[0], vL = UL[2]/UL[0];
    double pL = pressure_da(UL[0], UL[1], UL[2], UL[3]);
    double HL = (UL[3] + pL) / rhoL;

    double rhoR = UR[0], uR = UR[1]/UR[0], vR = UR[2]/UR[0];
    double pR = pressure_da(UR[0], UR[1], UR[2], UR[3]);
    double HR = (UR[3] + pR) / rhoR;

    double FL[4], GL[4], FR[4], GR[4];
    FL[0] = UL[1]; FL[1] = UL[1]*uL+pL; FL[2] = UL[1]*vL; FL[3] = (UL[3]+pL)*uL;
    GL[0] = UL[2]; GL[1] = UL[2]*uL;    GL[2] = UL[2]*vL+pL; GL[3] = (UL[3]+pL)*vL;
    FR[0] = UR[1]; FR[1] = UR[1]*uR+pR; FR[2] = UR[1]*vR; FR[3] = (UR[3]+pR)*uR;
    GR[0] = UR[2]; GR[1] = UR[2]*uR;    GR[2] = UR[2]*vR+pR; GR[3] = (UR[3]+pR)*vR;

    double srL = sqrt(rhoL), srR = sqrt(rhoR), srLR = srL + srR;
    double uRoe = (srL*uL + srR*uR) / srLR;
    double vRoe = (srL*vL + srR*vR) / srLR;
    double HRoe = (srL*HL + srR*HR) / srLR;
    double qRoe2 = uRoe*uRoe + vRoe*vRoe;
    double cRoe = sqrt(fmax((GAMMA_DA - 1.0)*(HRoe - 0.5*qRoe2), 1e-14));
    double alpha = fabs(uRoe*nx + vRoe*ny) + cRoe;

    for (int n = 0; n < 4; ++n)
        Fnum[n] = 0.5*((FL[n]*nx + GL[n]*ny) + (FR[n]*nx + GR[n]*ny))
                - 0.5*alpha*(UR[n] - UL[n]);
}

// ============================================================================
// HLLC numerical flux (duplicated from forward solver for FD)
// ============================================================================

__device__ void hllc_da(const double UL[4], const double UR[4],
                        double nx, double ny, double Fnum[4])
{
    const double gm1 = GAMMA_DA - 1.0;
    const double RHOMIN = 1e-10;
    const double PMIN   = 1e-10;

    double rhoL = fmax(UL[0], RHOMIN), uL = UL[1]/rhoL, vL = UL[2]/rhoL;
    double pL   = fmax(pressure_da(rhoL, UL[1], UL[2], UL[3]), PMIN);
    double cL   = sqrt(GAMMA_DA * pL / rhoL);
    double EL   = UL[3];
    double HL   = (EL + pL) / rhoL;

    double rhoR = fmax(UR[0], RHOMIN), uR = UR[1]/rhoR, vR = UR[2]/rhoR;
    double pR   = fmax(pressure_da(rhoR, UR[1], UR[2], UR[3]), PMIN);
    double cR   = sqrt(GAMMA_DA * pR / rhoR);
    double ER   = UR[3];
    double HR   = (ER + pR) / rhoR;

    double vnL = uL*nx + vL*ny;
    double vnR = uR*nx + vR*ny;
    double vtL = -uL*ny + vL*nx;
    double vtR = -uR*ny + vR*nx;

    double srL = sqrt(rhoL), srR = sqrt(rhoR), srLR = srL + srR;
    double vnRoe = (srL*vnL + srR*vnR) / srLR;
    double HRoe  = (srL*HL  + srR*HR)  / srLR;
    double uRoe  = (srL*uL  + srR*uR)  / srLR;
    double vRoe  = (srL*vL  + srR*vR)  / srLR;
    double qRoe2 = uRoe*uRoe + vRoe*vRoe;
    double cRoe  = sqrt(fmax(gm1 * (HRoe - 0.5*qRoe2), 1e-14));

    double SL = fmin(vnL - cL, vnRoe - cRoe);
    double SR = fmax(vnR + cR, vnRoe + cRoe);

    double denom = rhoL*(SL - vnL) - rhoR*(SR - vnR);
    double SS = (pR - pL + rhoL*vnL*(SL - vnL) - rhoR*vnR*(SR - vnR))
              / fmax(fabs(denom), 1e-14) * ((denom >= 0.0) ? 1.0 : -1.0);

    if (SL >= 0.0) {
        Fnum[0] = (UL[1]*nx + UL[2]*ny);
        Fnum[1] = (UL[1]*uL + pL)*nx + UL[1]*vL*ny;
        Fnum[2] = UL[2]*uL*nx + (UL[2]*vL + pL)*ny;
        Fnum[3] = (EL + pL)*(uL*nx + vL*ny);
    } else if (SR <= 0.0) {
        Fnum[0] = (UR[1]*nx + UR[2]*ny);
        Fnum[1] = (UR[1]*uR + pR)*nx + UR[1]*vR*ny;
        Fnum[2] = UR[2]*uR*nx + (UR[2]*vR + pR)*ny;
        Fnum[3] = (ER + pR)*(uR*nx + vR*ny);
    } else {
        double rhoK, vnK, vtK, EK, pK, SK;
        if (SS >= 0.0) {
            rhoK = rhoL; vnK = vnL; vtK = vtL; EK = EL; pK = pL; SK = SL;
        } else {
            rhoK = rhoR; vnK = vnR; vtK = vtR; EK = ER; pK = pR; SK = SR;
        }
        double factor = rhoK * (SK - vnK) / (SK - SS);
        double rhoSK = factor;
        double unSK  = SS;
        double utSK  = vtK;
        double ESK   = factor * (EK/rhoK + (SS - vnK)*(SS + pK/(rhoK*(SK - vnK))));
        double uK = vnK*nx - vtK*ny;
        double vK = vnK*ny + vtK*nx;
        double FnK[4];
        FnK[0] = rhoK*vnK;
        FnK[1] = (rhoK*uK*vnK + pK*nx);
        FnK[2] = (rhoK*vK*vnK + pK*ny);
        FnK[3] = (EK + pK)*vnK;
        double uSK = unSK*nx - utSK*ny;
        double vSK = unSK*ny + utSK*nx;
        double USK[4] = {rhoSK, rhoSK*uSK, rhoSK*vSK, ESK};
        double UK[4]  = {rhoK, rhoK*uK, rhoK*vK, EK};
        for (int n = 0; n < 4; ++n)
            Fnum[n] = FnK[n] + SK * (USK[n] - UK[n]);
    }
}

// ============================================================================
// Unified FD-based adjoint flux: (dh/dU_side)^T * dpsi
// Works for both LF and HLLC -- differentiates through the entire flux
// including wave speed computation.
//   fluxType: 0 = Lax-Friedrichs,  1 = HLLC
//   side:     0 = perturb UL,       1 = perturb UR
// ============================================================================

__device__ void numFluxAdjFD_da(const double UL[4], const double UR[4],
                                double nx, double ny,
                                const double dpsi[4], int side,
                                int fluxType,
                                double adj_flux[4])
{
    double h0[4];
    if (fluxType == 1)
        hllc_da(UL, UR, nx, ny, h0);
    else
        laxFriedrichs_da(UL, UR, nx, ny, h0);

    for (int w = 0; w < 4; ++w) {
        double ULp[4] = {UL[0], UL[1], UL[2], UL[3]};
        double URp[4] = {UR[0], UR[1], UR[2], UR[3]};
        double* Up = (side == 0) ? ULp : URp;
        double fd_h = 1e-7 * fmax(fabs(Up[w]), 1.0);
        Up[w] += fd_h;
        double hp[4];
        if (fluxType == 1)
            hllc_da(ULp, URp, nx, ny, hp);
        else
            laxFriedrichs_da(ULp, URp, nx, ny, hp);
        double val = 0.0;
        for (int v = 0; v < 4; ++v)
            val += (hp[v] - h0[v]) / fd_h * dpsi[v];
        adj_flux[w] = val;
    }
}

// ============================================================================
// Boundary condition helpers (reused from continuous adjoint)
// ============================================================================

__device__ void slipWallBC_da(const double UL[4], double nx, double ny, double UR[4])
{
    double rho = UL[0];
    double u = UL[1] / rho;
    double v = UL[2] / rho;
    double Vn = u * nx + v * ny;
    UR[0] = rho;
    UR[1] = rho * (u - 2.0 * Vn * nx);
    UR[2] = rho * (v - 2.0 * Vn * ny);
    UR[3] = UL[3];
}

__device__ void riemannBC_da(const double UL[4], double nx, double ny,
                             double rhoInf, double uInf, double vInf, double pInf,
                             double UR[4])
{
    const double gm1 = GAMMA_DA - 1.0;
    double rhoI = UL[0], uI = UL[1]/rhoI, vI = UL[2]/rhoI;
    double pI = pressure_da(UL[0], UL[1], UL[2], UL[3]);
    double cI = sqrt(GAMMA_DA * pI / rhoI);
    double VnI = uI*nx + vI*ny;

    if (VnI >= cI) {
        UR[0] = UL[0]; UR[1] = UL[1]; UR[2] = UL[2]; UR[3] = UL[3];
        return;
    }
    if (VnI <= -cI) {
        UR[0] = rhoInf;
        UR[1] = rhoInf * uInf;
        UR[2] = rhoInf * vInf;
        UR[3] = pInf / gm1 + 0.5 * rhoInf * (uInf*uInf + vInf*vInf);
        return;
    }

    double cInfVal = sqrt(GAMMA_DA * pInf / rhoInf);
    double VnInf = uInf*nx + vInf*ny;
    double Rplus  = VnI + 2.0*cI/gm1;
    double Rminus = VnInf - 2.0*cInfVal/gm1;
    double sB, VtB_x, VtB_y;
    if (VnI < 0.0) {
        sB = pInf / pow(rhoInf, GAMMA_DA);
        VtB_x = uInf - VnInf*nx;
        VtB_y = vInf - VnInf*ny;
    } else {
        sB = pI / pow(rhoI, GAMMA_DA);
        VtB_x = uI - VnI*nx;
        VtB_y = vI - VnI*ny;
    }
    double VnB = 0.5*(Rplus + Rminus);
    double cB  = 0.25*gm1*(Rplus - Rminus);
    if (cB < 1e-14) cB = 1e-14;
    double rhoB = pow(cB*cB/(GAMMA_DA*sB), 1.0/gm1);
    double uB = VtB_x + VnB*nx;
    double vB = VtB_y + VnB*ny;
    double pB = rhoB*cB*cB/GAMMA_DA;
    UR[0] = rhoB;
    UR[1] = rhoB*uB;
    UR[2] = rhoB*vB;
    UR[3] = pB/gm1 + 0.5*rhoB*(uB*uB + vB*vB);
}

// ============================================================================
// Kernel: Fused discrete adjoint volume + surface integral (modal, 1 block/elem)
//
// Key differences from continuous adjoint (adjointVolumeSurfaceKernel):
//   1. LF flux: FD-based Jacobian (not frozen alpha + analytical An)
//   2. Farfield BC: full (dh/dU_L + dh/dU_R * Rbc)^T psi (no dropped terms)
//   3. AV: uses forward epsilon directly (no adjoint-specific sensor)
// ============================================================================

__global__ void discreteAdjointVolumeSurfaceKernel(
    const double* __restrict__ d_psiCoeff,
    const double* __restrict__ d_U,
    const double* __restrict__ d_Ucoeff,
    double* __restrict__ d_rhsCoeff,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    const double* __restrict__ d_faceNx,
    const double* __restrict__ d_faceNy,
    const double* __restrict__ d_faceJac,
    const int* __restrict__ d_elem2face,
    const int* __restrict__ d_face_elemL,
    const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_faceL,
    const int* __restrict__ d_face_faceR,
    const int* __restrict__ d_face_bcType,
    const double* __restrict__ d_epsilon,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf,
    int fluxType)
{
    int eL = blockIdx.x;
    if (eL >= nE) return;
    int e = eL;

    int tid = threadIdx.x;
    int nwork   = NVAR_GPU * nmodes;
    int nFaceQP = 4 * nqFace;

    extern __shared__ double smem[];
    double* sPsiCoeff = smem;
    double* sU        = sPsiCoeff + NVAR_GPU * nmodes;
    double* sFnumW    = sU + NVAR_GPU * nqVol;

    // ===== Phase 1: Load element data =====
    for (int i = tid; i < NVAR_GPU * nmodes; i += blockDim.x) {
        int v = i / nmodes;
        int m = i % nmodes;
        sPsiCoeff[i] = d_psiCoeff[v * nE * nmodes + e * nmodes + m];
    }
    int baseQ = e * nqVol;
    for (int i = tid; i < nqVol; i += blockDim.x) {
        int gIdx = baseQ + i;
        sU[0 * nqVol + i] = d_U[0 * totalDOF + gIdx];
        sU[1 * nqVol + i] = d_U[1 * totalDOF + gIdx];
        sU[2 * nqVol + i] = d_U[2 * totalDOF + gIdx];
        sU[3 * nqVol + i] = d_U[3 * totalDOF + gIdx];
    }
    for (int i = tid; i < nFaceQP * 4; i += blockDim.x)
        sFnumW[i] = 0.0;
    __syncthreads();

    // ===== Phase 2: Volume integral (threads 0..nwork-1) =====
    double volResult = 0.0;
    int v_id = -1, m_id = -1, mi_id = 0, mj_id = 0;

    if (tid < nwork) {
        v_id  = tid / nmodes;
        m_id  = tid % nmodes;
        mi_id = m_id / P1;
        mj_id = m_id % P1;

        for (int qx = 0; qx < nq1d; ++qx) {
            double Bi_m = da_Bmat[mi_id * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                int gIdx   = baseQ + qLocal;

                double Bj_m = da_Bmat[mj_id * nq1d + qe];
                double phi_m = Bi_m * Bj_m;
                double wqdetJ = da_wq[qx] * da_wq[qe] * d_detJ[gIdx];

                double gx[4] = {0,0,0,0};
                double gy[4] = {0,0,0,0};
                for (int i = 0; i < P1; ++i) {
                    double Bi = da_Bmat[i * nq1d + qx];
                    double Di = da_Dmat[i * nq1d + qx];
                    for (int j = 0; j < P1; ++j) {
                        double Bj = da_Bmat[j * nq1d + qe];
                        double Dj = da_Dmat[j * nq1d + qe];
                        double dphidxi  = Di * Bj;
                        double dphideta = Bi * Dj;
                        double dphidx = d_dxidx[gIdx]*dphidxi + d_detadx[gIdx]*dphideta;
                        double dphidy = d_dxidy[gIdx]*dphidxi + d_detady[gIdx]*dphideta;

                        for (int v = 0; v < 4; ++v) {
                            double c = sPsiCoeff[v * nmodes + i * P1 + j];
                            gx[v] += c * dphidx;
                            gy[v] += c * dphidy;
                        }
                    }
                }

                double Uq[4];
                for (int v = 0; v < 4; ++v) Uq[v] = sU[v*nqVol + qLocal];

                double A[4][4], B_mat[4][4];
                eulerFluxJacX_da(Uq, A);
                eulerFluxJacY_da(Uq, B_mat);

                double val = 0.0;
                for (int v = 0; v < 4; ++v)
                    val += A[v][v_id] * gx[v] + B_mat[v][v_id] * gy[v];

                volResult += wqdetJ * phi_m * val;
            }
        }
    }

    // ===== Phase 3: Surface flux (threads nwork..nwork+nFaceQP-1) =====
    else if (tid < nwork + nFaceQP) {
        int fq_id = tid - nwork;
        int lf = fq_id / nqFace;
        int q  = fq_id % nqFace;

        int f  = d_elem2face[e * 4 + lf];
        int eL_f = d_face_elemL[f];
        int eR_f = d_face_elemR[f];
        bool is_left = (e == eL_f);
        int eN  = is_left ? eR_f : eL_f;
        int lfN = is_left ? d_face_faceR[f] : d_face_faceL[f];
        bool is_boundary = (eN < 0);

        int q_face = is_left ? q : (nqFace - 1 - q);
        int fIdx = f * nqFace + q_face;
        double nx = d_faceNx[fIdx];
        double ny = d_faceNy[fIdx];
        double wf = da_wq[q] * d_faceJac[fIdx];

        double psi_me[4] = {0,0,0,0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    psi_me[vv] += sPsiCoeff[vv*nmodes + ii*P1+jj]
                                * evalPhiFace_da(lf, ii, jj, q, P1, nq1d);

        double U_me[4] = {0,0,0,0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    U_me[vv] += d_Ucoeff[vv*nE*nmodes + e*nmodes + ii*P1+jj]
                              * evalPhiFace_da(lf, ii, jj, q, P1, nq1d);

        double adj_flux[4] = {0,0,0,0};

        if (!is_boundary) {
            // ---------- Interior face ----------
            int qN = nqFace - 1 - q;

            double psi_nb[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        psi_nb[vv] += d_psiCoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                    * evalPhiFace_da(lfN, ii, jj, qN, P1, nq1d);

            double U_nb[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        U_nb[vv] += d_Ucoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                  * evalPhiFace_da(lfN, ii, jj, qN, P1, nq1d);

            double UL[4], UR_st[4], psi_L[4], psi_R[4];
            if (is_left) {
                for (int vv=0;vv<4;++vv) { UL[vv]=U_me[vv]; UR_st[vv]=U_nb[vv]; psi_L[vv]=psi_me[vv]; psi_R[vv]=psi_nb[vv]; }
            } else {
                for (int vv=0;vv<4;++vv) { UL[vv]=U_nb[vv]; UR_st[vv]=U_me[vv]; psi_L[vv]=psi_nb[vv]; psi_R[vv]=psi_me[vv]; }
            }

            double dpsi_RL[4];
            for (int vv=0;vv<4;++vv) dpsi_RL[vv] = psi_R[vv] - psi_L[vv];

            int side = is_left ? 0 : 1;
            numFluxAdjFD_da(UL, UR_st, nx, ny, dpsi_RL, side, fluxType, adj_flux);

            if (d_epsilon != nullptr) {
                double eps_me = d_epsilon[e];
                double eps_nb = d_epsilon[eN];
                double eps_f  = fmax(eps_me, eps_nb);
                if (eps_f > 0.0) {
                    double sigma = eps_f * (double)(P1*P1)
                                 / fmax(2.0 * d_faceJac[fIdx], 1e-30);
                    double dpsi_nb_me[4];
                    for (int vv=0;vv<4;++vv) dpsi_nb_me[vv] = psi_nb[vv] - psi_me[vv];
                    for (int w = 0; w < 4; ++w)
                        adj_flux[w] += sigma * dpsi_nb_me[w];
                }
            }

        } else {
            // ---------- Boundary face ----------
            int bcType = d_face_bcType[f];

            double h0[4];
            double UR_bc0[4];

            if (bcType == 1)
                slipWallBC_da(U_me, nx, ny, UR_bc0);
            else if (bcType == 2)
                riemannBC_da(U_me, nx, ny, rhoInf, uInf, vInf, pInf, UR_bc0);
            else
                for (int vv=0;vv<4;++vv) UR_bc0[vv] = U_me[vv];

            if (fluxType == 1)
                hllc_da(U_me, UR_bc0, nx, ny, h0);
            else
                laxFriedrichs_da(U_me, UR_bc0, nx, ny, h0);

            for (int w = 0; w < 4; ++w) {
                double ULp[4] = {U_me[0], U_me[1], U_me[2], U_me[3]};
                double fd_h = 1e-7 * fmax(fabs(ULp[w]), 1.0);
                ULp[w] += fd_h;

                double UR_bc_p[4];
                if (bcType == 1)
                    slipWallBC_da(ULp, nx, ny, UR_bc_p);
                else if (bcType == 2)
                    for (int vv=0;vv<4;++vv) UR_bc_p[vv] = UR_bc0[vv];
                else
                    for (int vv=0;vv<4;++vv) UR_bc_p[vv] = ULp[vv];

                double hp[4];
                if (fluxType == 1)
                    hllc_da(ULp, UR_bc_p, nx, ny, hp);
                else
                    laxFriedrichs_da(ULp, UR_bc_p, nx, ny, hp);

                double val = 0.0;
                for (int v = 0; v < 4; ++v)
                    val += (hp[v] - h0[v]) / fd_h * psi_me[v];
                adj_flux[w] = -val;
            }

            if (d_epsilon != nullptr) {
                double eps_me = d_epsilon[e];
                if (eps_me > 0.0) {
                    double sigma = eps_me * (double)(P1*P1)
                                 / fmax(2.0 * d_faceJac[fIdx], 1e-30);
                    // For wall: Rbc reflects normal component
                    // For farfield or other: use identity
                    if (bcType == 1) {
                        double nx2 = nx*nx, ny2 = ny*ny, nxny = nx*ny;
                        double Rbc[4][4];
                        Rbc[0][0]=1; Rbc[0][1]=0;          Rbc[0][2]=0;          Rbc[0][3]=0;
                        Rbc[1][0]=0; Rbc[1][1]=1-2*nx2;    Rbc[1][2]=-2*nxny;    Rbc[1][3]=0;
                        Rbc[2][0]=0; Rbc[2][1]=-2*nxny;    Rbc[2][2]=1-2*ny2;    Rbc[2][3]=0;
                        Rbc[3][0]=0; Rbc[3][1]=0;          Rbc[3][2]=0;          Rbc[3][3]=1;
                        for (int w = 0; w < 4; ++w) {
                            double ip_val = psi_me[w];
                            for (int v = 0; v < 4; ++v)
                                ip_val -= Rbc[v][w] * psi_me[v];
                            adj_flux[w] += -sigma * ip_val;
                        }
                    } else {
                        for (int w = 0; w < 4; ++w)
                            adj_flux[w] -= sigma * psi_me[w];
                    }
                }
            }
        }

        for (int w = 0; w < 4; ++w)
            sFnumW[fq_id * 4 + w] = wf * adj_flux[w];
    }

    __syncthreads();

    // ===== Phase 4: Surface integral + combine with volume =====
    if (tid < nwork) {
        double surfResult = 0.0;
        for (int fq = 0; fq < nFaceQP; ++fq) {
            int lf = fq / nqFace;
            int q  = fq % nqFace;
            double phi = evalPhiFace_da(lf, mi_id, mj_id, q, P1, nq1d);
            surfResult += sFnumW[fq * 4 + v_id] * phi;
        }
        d_rhsCoeff[v_id * nE * nmodes + e * nmodes + m_id] = volResult + surfResult;
    }
}

// ============================================================================
// Kernel: AV volume diffusion (self-adjoint, uses forward epsilon)
// ============================================================================

__global__ void daAVVolumeKernel(
    const double* __restrict__ d_psiCoeff,
    double* __restrict__ d_adjRhsCoeff,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    const double* __restrict__ d_epsilon,
    int nE, int P1, int nq1d, int nmodes, int nqVol)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    double eps = d_epsilon[e];
    if (eps <= 0.0) return;

    int tid = threadIdx.x;
    int nwork = NVAR_GPU * nmodes;

    extern __shared__ double smem[];
    double* sCoeff  = smem;
    double* gradXi  = sCoeff  + NVAR_GPU * nmodes;
    double* gradEta = gradXi  + NVAR_GPU * nqVol;

    for (int i = tid; i < NVAR_GPU * nmodes; i += blockDim.x) {
        int v = i / nmodes;
        int m = i % nmodes;
        sCoeff[i] = d_psiCoeff[v * nE * nmodes + e * nmodes + m];
    }
    __syncthreads();

    for (int w = tid; w < NVAR_GPU * nqVol; w += blockDim.x) {
        int v  = w / nqVol;
        int q  = w % nqVol;
        int qx = q / nq1d;
        int qe = q % nq1d;

        double dxi = 0.0, deta = 0.0;
        for (int i = 0; i < P1; ++i) {
            double Di_qx = da_Dmat[i * nq1d + qx];
            double Bi_qx = da_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j) {
                double c = sCoeff[v * nmodes + i * P1 + j];
                dxi  += c * Di_qx * da_Bmat[j * nq1d + qe];
                deta += c * Bi_qx * da_Dmat[j * nq1d + qe];
            }
        }
        gradXi[w]  = dxi;
        gradEta[w] = deta;
    }
    __syncthreads();

    for (int w = tid; w < nwork; w += blockDim.x) {
        int v  = w / nmodes;
        int m  = w % nmodes;
        int mi = m / P1;
        int mj = m % P1;

        double result = 0.0;
        for (int qx = 0; qx < nq1d; ++qx) {
            double Bi = da_Bmat[mi * nq1d + qx];
            double Di = da_Dmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                int gIdx   = e * nqVol + qLocal;
                double wq  = da_wq[qx] * da_wq[qe] * d_detJ[gIdx];

                double Bj = da_Bmat[mj * nq1d + qe];
                double Dj = da_Dmat[mj * nq1d + qe];

                double dphidxi  = Di * Bj;
                double dphideta = Bi * Dj;
                double dphidx = d_dxidx[gIdx]*dphidxi + d_detadx[gIdx]*dphideta;
                double dphidy = d_dxidy[gIdx]*dphidxi + d_detady[gIdx]*dphideta;

                double grXi  = gradXi [v * nqVol + qLocal];
                double grEta = gradEta[v * nqVol + qLocal];
                double dUdx = d_dxidx[gIdx]*grXi + d_detadx[gIdx]*grEta;
                double dUdy = d_dxidy[gIdx]*grXi + d_detady[gIdx]*grEta;

                result -= eps * wq * (dUdx*dphidx + dUdy*dphidy);
            }
        }

        d_adjRhsCoeff[v * nE * nmodes + e * nmodes + m] += result;
    }
}

// ============================================================================
// Kernel: Modal mass solve (coefficient-to-coefficient)
// ============================================================================

__global__ void daMassSolveModalKernel(
    const double* __restrict__ d_rhsCoeff,
    double* __restrict__ d_R,
    const double* __restrict__ d_Minv,
    int nE, int nmodes)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork = NVAR_GPU * nmodes;

    for (int w = tid; w < nwork; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;

        double val = 0.0;
        for (int mp = 0; mp < nmodes; ++mp)
            val += d_Minv[e * nmodes * nmodes + m * nmodes + mp]
                 * d_rhsCoeff[v * nE * nmodes + e * nmodes + mp];

        d_R[v * nE * nmodes + e * nmodes + m] = val;
    }
}

// ============================================================================
// Kernel: RK4 stage update
// ============================================================================

__global__ void daRK4StageKernel(
    double* __restrict__ d_psi,
    double* __restrict__ d_psiTmp,
    double* __restrict__ d_k,
    const double* __restrict__ d_R,
    const double* __restrict__ d_k1,
    const double* __restrict__ d_k2,
    const double* __restrict__ d_k3,
    double dt, int stage, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double ki = dt * d_R[i];
    d_k[i] = ki;

    switch (stage) {
        case 1: d_psiTmp[i] = d_psi[i] + 0.5 * ki; break;
        case 2: d_psiTmp[i] = d_psi[i] + 0.5 * ki; break;
        case 3: d_psiTmp[i] = d_psi[i] + ki;        break;
        case 4:
            d_psi[i] += (1.0/6.0) * (d_k1[i] + 2.0*d_k2[i] + 2.0*d_k3[i] + ki);
            break;
    }
}

// ============================================================================
// Kernel: Objective gradient in coefficient space
// ============================================================================

__global__ void daObjectiveGradientCoeffKernel(
    const double* __restrict__ d_Ucoeff,
    double* __restrict__ d_dJdUcoeff,
    const double* __restrict__ d_faceNx,
    const double* __restrict__ d_faceNy,
    const double* __restrict__ d_faceJac,
    const int* __restrict__ d_elem2face,
    const int* __restrict__ d_face_elemL,
    const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_bcType,
    int nE, int P1, int nq1d, int nmodes, int nqFace,
    double normalization, double forceNx, double forceNy)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork = NVAR_GPU * nmodes;

    for (int work = tid; work < nwork; work += blockDim.x) {
        int w = work / nmodes;
        int m = work % nmodes;
        int mi = m / P1;
        int mj = m % P1;

        double result = 0.0;

        for (int lf = 0; lf < 4; ++lf) {
            int f = d_elem2face[e * 4 + lf];
            int eR_f = d_face_elemR[f];
            if (eR_f >= 0) continue;
            int bcType = d_face_bcType[f];
            if (bcType != 1) continue;

            bool is_left = (e == d_face_elemL[f]);
            if (!is_left) continue;

            for (int q = 0; q < nqFace; ++q) {
                int fIdx = f * nqFace + q;
                double nx = d_faceNx[fIdx];
                double ny = d_faceNy[fIdx];
                double wf = da_wq[q] * d_faceJac[fIdx];

                double Uf[4] = {0,0,0,0};
                for (int vv = 0; vv < 4; ++vv)
                    for (int ii = 0; ii < P1; ++ii)
                        for (int jj = 0; jj < P1; ++jj)
                            Uf[vv] += d_Ucoeff[vv*nE*nmodes + e*nmodes + ii*P1+jj]
                                    * evalPhiFace_da(lf, ii, jj, q, P1, nq1d);

                double rho = Uf[0];
                double u_v = Uf[1]/rho, v_v = Uf[2]/rho;
                double q2 = u_v*u_v + v_v*v_v;
                const double gm1 = GAMMA_DA - 1.0;

                double dpdU[4];
                dpdU[0] = gm1 * 0.5 * q2;
                dpdU[1] = -gm1 * u_v;
                dpdU[2] = -gm1 * v_v;
                dpdU[3] = gm1;

                double phi = evalPhiFace_da(lf, mi, mj, q, P1, nq1d);

                result += wf * dpdU[w] * (forceNx * nx + forceNy * ny) * phi / normalization;
            }
        }

        d_dJdUcoeff[w * nE * nmodes + e * nmodes + m] = result;
    }
}

// ============================================================================
// Kernel: Force reduction (same as continuous adjoint)
// ============================================================================

__global__ void daForceReductionKernel(
    const double* __restrict__ d_Ucoeff,
    const double* __restrict__ d_faceNx,
    const double* __restrict__ d_faceNy,
    const double* __restrict__ d_faceJac,
    const int* __restrict__ d_elem2face,
    const int* __restrict__ d_face_elemL,
    const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_bcType,
    double* __restrict__ d_liftBuf,
    int nE, int P1, int nq1d, int nmodes, int nqFace,
    double normalization, double forceNx, double forceNy)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int e = blockIdx.x;
    double mySum = 0.0;

    if (e < nE) {
        for (int lf = 0; lf < 4; ++lf) {
            int f = d_elem2face[e * 4 + lf];
            int eR_f = d_face_elemR[f];
            if (eR_f >= 0) continue;
            int bcType = d_face_bcType[f];
            if (bcType != 1) continue;
            bool is_left = (e == d_face_elemL[f]);
            if (!is_left) continue;

            for (int q = tid; q < nqFace; q += blockDim.x) {
                int fIdx = f * nqFace + q;
                double nx = d_faceNx[fIdx];
                double ny = d_faceNy[fIdx];
                double wf = da_wq[q] * d_faceJac[fIdx];

                double Uf[4] = {0,0,0,0};
                for (int vv = 0; vv < 4; ++vv)
                    for (int ii = 0; ii < P1; ++ii)
                        for (int jj = 0; jj < P1; ++jj)
                            Uf[vv] += d_Ucoeff[vv*nE*nmodes + e*nmodes + ii*P1+jj]
                                    * evalPhiFace_da(lf, ii, jj, q, P1, nq1d);

                double p = pressure_da(Uf[0], Uf[1], Uf[2], Uf[3]);
                mySum += wf * p * (forceNx * nx + forceNy * ny) / normalization;
            }
        }
    }

    sdata[tid] = mySum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_liftBuf[e] = sdata[0];
}

// ============================================================================
// Kernel: Coefficient add  d_out[i] += d_in[i]
// ============================================================================

__global__ void daCoeffAddKernel(
    double* __restrict__ d_out,
    const double* __restrict__ d_in,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_out[idx] += d_in[idx];
}

// ============================================================================
// Kernel: NaN check
// ============================================================================

__global__ void daNanCheckKernel(const double* __restrict__ data, int N,
                                 int* __restrict__ flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && (isnan(data[i]) || isinf(data[i])))
        *flag = 1;
}

// ============================================================================
// Kernel: L2 norm reduction
// ============================================================================

__device__ double atomicAddDouble_da(double* addr, double val)
{
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void daL2NormKernel(const double* __restrict__ data, int N,
                               double* __restrict__ d_result)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double mySum = 0.0;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
        mySum += data[i] * data[i];

    sdata[tid] = mySum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAddDouble_da(d_result, sdata[0]);
}

// ============================================================================
// Kernel: Backward transform (coefficients -> quad points)
// ============================================================================

__global__ void daBackwardTransformKernel(
    const double* __restrict__ d_coeffs,
    double* __restrict__ d_quad,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork = NVAR_GPU * nqVol;

    for (int w = tid; w < nwork; w += blockDim.x) {
        int v  = w / nqVol;
        int q  = w % nqVol;
        int qx = q / nq1d;
        int qe = q % nq1d;

        double val = 0.0;
        for (int i = 0; i < P1; ++i) {
            double Bi = da_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j)
                val += d_coeffs[v * nE * nmodes + e * nmodes + i * P1 + j]
                     * Bi * da_Bmat[j * nq1d + qe];
        }

        d_quad[v * totalDOF + e * nqVol + q] = val;
    }
}

// ============================================================================
// Kernel: Per-variable residual norm
// ============================================================================

__global__ void daResNormPerVarKernel(
    const double* __restrict__ d_adjR,
    double* __restrict__ d_norms,
    int totalDOF)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = blockIdx.x * bs + tid; i < totalDOF; i += gridDim.x * bs) {
        double v0 = d_adjR[i];
        double v1 = d_adjR[totalDOF + i];
        double v2 = d_adjR[2 * totalDOF + i];
        double v3 = d_adjR[3 * totalDOF + i];
        s0 += v0 * v0;
        s1 += v1 * v1;
        s2 += v2 * v2;
        s3 += v3 * v3;
    }

    sdata[tid]          = s0;
    sdata[tid + bs]     = s1;
    sdata[tid + 2 * bs] = s2;
    sdata[tid + 3 * bs] = s3;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]          += sdata[tid + s];
            sdata[tid + bs]     += sdata[tid + s + bs];
            sdata[tid + 2 * bs] += sdata[tid + s + 2 * bs];
            sdata[tid + 3 * bs] += sdata[tid + s + 3 * bs];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddDouble_da(&d_norms[0], sdata[0]);
        atomicAddDouble_da(&d_norms[1], sdata[bs]);
        atomicAddDouble_da(&d_norms[2], sdata[2 * bs]);
        atomicAddDouble_da(&d_norms[3], sdata[3 * bs]);
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

void discreteAdjointAllocate(DiscreteAdjointGPUData& da, const GPUSolverData& gpu)
{
    da.modalMode  = gpu.modalMode;
    da.totalCoeff = gpu.nE * gpu.nmodes;
    da.primaryDOF = da.modalMode ? da.totalCoeff : gpu.totalDOF;

    int solSize   = NVAR_GPU * da.primaryDOF;
    int coeffSize = NVAR_GPU * gpu.nE * gpu.nmodes;
    int quadSize  = NVAR_GPU * gpu.totalDOF;

    DA_CUDA_CHECK(cudaMalloc(&da.d_psi,    solSize * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_psiTmp, solSize * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_pk1,    solSize * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_pk2,    solSize * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_pk3,    solSize * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_pk4,    solSize * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_adjR,   solSize * sizeof(double)));

    if (da.modalMode) {
        da.d_psiCoeff = nullptr;
    } else {
        DA_CUDA_CHECK(cudaMalloc(&da.d_psiCoeff, coeffSize * sizeof(double)));
    }
    DA_CUDA_CHECK(cudaMalloc(&da.d_adjRhsCoeff, coeffSize * sizeof(double)));

    da.d_psi_quad   = nullptr;
    da.d_dJdUcoeff  = nullptr;
    if (da.modalMode) {
        DA_CUDA_CHECK(cudaMalloc(&da.d_psi_quad,  quadSize * sizeof(double)));
        DA_CUDA_CHECK(cudaMalloc(&da.d_dJdUcoeff, coeffSize * sizeof(double)));
        DA_CUDA_CHECK(cudaMemset(da.d_dJdUcoeff, 0, coeffSize * sizeof(double)));
    } else {
        DA_CUDA_CHECK(cudaMalloc(&da.d_dJdUcoeff, coeffSize * sizeof(double)));
        DA_CUDA_CHECK(cudaMemset(da.d_dJdUcoeff, 0, coeffSize * sizeof(double)));
    }

    DA_CUDA_CHECK(cudaMalloc(&da.d_dtMin,   sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_nanFlag, sizeof(int)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_liftBuf, gpu.nE * sizeof(double)));
    DA_CUDA_CHECK(cudaMalloc(&da.d_normBuf, 4 * sizeof(double)));

    DA_CUDA_CHECK(cudaMemset(da.d_psi, 0, solSize * sizeof(double)));

    da.chordRef = 1.0;
}

void discreteAdjointFree(DiscreteAdjointGPUData& da)
{
    cudaFree(da.d_psi);     cudaFree(da.d_psiTmp);
    cudaFree(da.d_pk1);     cudaFree(da.d_pk2);
    cudaFree(da.d_pk3);     cudaFree(da.d_pk4);
    cudaFree(da.d_adjR);
    if (da.d_psiCoeff) cudaFree(da.d_psiCoeff);
    cudaFree(da.d_adjRhsCoeff);
    if (da.d_dJdUcoeff) cudaFree(da.d_dJdUcoeff);
    if (da.d_psi_quad) cudaFree(da.d_psi_quad);
    cudaFree(da.d_dtMin);   cudaFree(da.d_nanFlag);
    cudaFree(da.d_liftBuf); cudaFree(da.d_normBuf);
}

void discreteAdjointSetBasisData(const double* Bmat, const double* Dmat,
                                 const double* blr, const double* wq,
                                 int P1, int nq1d)
{
    DA_CUDA_CHECK(cudaMemcpyToSymbol(da_Bmat, Bmat, P1*nq1d*sizeof(double)));
    DA_CUDA_CHECK(cudaMemcpyToSymbol(da_Dmat, Dmat, P1*nq1d*sizeof(double)));
    DA_CUDA_CHECK(cudaMemcpyToSymbol(da_blr,  blr,  P1*2*sizeof(double)));
    DA_CUDA_CHECK(cudaMemcpyToSymbol(da_wq,   wq,   nq1d*sizeof(double)));
}

void discreteAdjointSetNodalToModal(const double* T, int P1)
{
    DA_CUDA_CHECK(cudaMemcpyToSymbol(da_NodalToModal, T, P1*P1*sizeof(double)));
}

double discreteAdjointComputeForceCoeff(DiscreteAdjointGPUData& da,
                                        const GPUSolverData& gpu,
                                        double chordRef,
                                        double forceNx, double forceNy)
{
    int nE = gpu.nE;
    double Vinf2 = gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf;
    double normalization = 0.5 * gpu.rhoInf * Vinf2 * chordRef;
    if (normalization < 1e-30) normalization = 1.0;

    int bk = 32;
    int smem = bk * sizeof(double);
    daForceReductionKernel<<<nE, bk, smem>>>(
        gpu.d_Ucoeff,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_bcType,
        da.d_liftBuf,
        nE, gpu.P1, gpu.nq1d, gpu.nmodes, gpu.nqFace,
        normalization, forceNx, forceNy);

    std::vector<double> buf(nE);
    DA_CUDA_CHECK(cudaMemcpy(buf.data(), da.d_liftBuf, nE*sizeof(double), cudaMemcpyDeviceToHost));
    double result = 0.0;
    for (int i = 0; i < nE; ++i) result += buf[i];
    return result;
}

void discreteAdjointComputeObjectiveGradient(DiscreteAdjointGPUData& da,
                                             const GPUSolverData& gpu,
                                             double chordRef,
                                             double forceNx, double forceNy)
{
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes;

    double Vinf2 = gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf;
    double normalization = 0.5 * gpu.rhoInf * Vinf2 * chordRef;
    if (normalization < 1e-30) normalization = 1.0;

    int coeffSize = NVAR_GPU * nE * nmodes;
    DA_CUDA_CHECK(cudaMemset(da.d_adjRhsCoeff, 0, coeffSize * sizeof(double)));

    int bk = std::max(64, NVAR_GPU * nmodes);
    if (bk % 32 != 0) bk = ((bk + 31) / 32) * 32;
    daObjectiveGradientCoeffKernel<<<nE, bk>>>(
        gpu.d_Ucoeff, da.d_adjRhsCoeff,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_bcType,
        nE, P1, nq1d, nmodes, gpu.nqFace,
        normalization, forceNx, forceNy);

    int bk2 = std::max(64, NVAR_GPU * nmodes);
    if (bk2 % 32 != 0) bk2 = ((bk2 + 31) / 32) * 32;
    daMassSolveModalKernel<<<nE, bk2>>>(
        da.d_adjRhsCoeff, da.d_dJdUcoeff, gpu.d_Minv,
        nE, nmodes);
}

// ============================================================================
// Lift-over-Drag objective: J = Cl / Cd
// ============================================================================

__global__ void daLinearCombineKernel(double a, const double* __restrict__ A,
                                      double b, const double* __restrict__ B,
                                      double* __restrict__ C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = a * A[i] + b * B[i];
}

double discreteAdjointComputeLiftOverDrag(DiscreteAdjointGPUData& da,
                                          const GPUSolverData& gpu,
                                          double chordRef, double AoA_rad,
                                          double& Cl_out, double& Cd_out)
{
    double liftNx = -std::sin(AoA_rad), liftNy = std::cos(AoA_rad);
    double dragNx =  std::cos(AoA_rad), dragNy = std::sin(AoA_rad);
    Cl_out = discreteAdjointComputeForceCoeff(da, gpu, chordRef, liftNx, liftNy);
    Cd_out = discreteAdjointComputeForceCoeff(da, gpu, chordRef, dragNx, dragNy);
    return Cl_out / Cd_out;
}

void discreteAdjointComputeLiftOverDragGradient(DiscreteAdjointGPUData& da,
                                                const GPUSolverData& gpu,
                                                double chordRef, double AoA_rad)
{
    double Cl, Cd;
    discreteAdjointComputeLiftOverDrag(da, gpu, chordRef, AoA_rad, Cl, Cd);

    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes;
    double Vinf2 = gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf;
    double normalization = 0.5 * gpu.rhoInf * Vinf2 * chordRef;
    if (normalization < 1e-30) normalization = 1.0;

    int coeffSize = NVAR_GPU * nE * nmodes;
    int bk = std::max(64, NVAR_GPU * nmodes);
    if (bk % 32 != 0) bk = ((bk + 31) / 32) * 32;

    // Temporary buffer for lift gradient
    double* d_gradLift = nullptr;
    DA_CUDA_CHECK(cudaMalloc(&d_gradLift, coeffSize * sizeof(double)));

    // --- Compute lift gradient ---
    double liftNx = -std::sin(AoA_rad), liftNy = std::cos(AoA_rad);
    DA_CUDA_CHECK(cudaMemset(da.d_adjRhsCoeff, 0, coeffSize * sizeof(double)));
    daObjectiveGradientCoeffKernel<<<nE, bk>>>(
        gpu.d_Ucoeff, da.d_adjRhsCoeff,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_bcType,
        nE, P1, nq1d, nmodes, gpu.nqFace,
        normalization, liftNx, liftNy);

    int bk2 = std::max(64, NVAR_GPU * nmodes);
    if (bk2 % 32 != 0) bk2 = ((bk2 + 31) / 32) * 32;
    daMassSolveModalKernel<<<nE, bk2>>>(
        da.d_adjRhsCoeff, d_gradLift, gpu.d_Minv, nE, nmodes);

    // --- Compute drag gradient into d_dJdUcoeff ---
    double dragNx = std::cos(AoA_rad), dragNy = std::sin(AoA_rad);
    DA_CUDA_CHECK(cudaMemset(da.d_adjRhsCoeff, 0, coeffSize * sizeof(double)));
    daObjectiveGradientCoeffKernel<<<nE, bk>>>(
        gpu.d_Ucoeff, da.d_adjRhsCoeff,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_bcType,
        nE, P1, nq1d, nmodes, gpu.nqFace,
        normalization, dragNx, dragNy);

    daMassSolveModalKernel<<<nE, bk2>>>(
        da.d_adjRhsCoeff, da.d_dJdUcoeff, gpu.d_Minv, nE, nmodes);

    // dJ/dU = (1/Cd)*dCl/dU - (Cl/Cd^2)*dCd/dU
    double a = 1.0 / Cd;
    double b = -Cl / (Cd * Cd);
    int gd = (coeffSize + 255) / 256;
    daLinearCombineKernel<<<gd, 256>>>(a, d_gradLift, b, da.d_dJdUcoeff,
                                       da.d_dJdUcoeff, coeffSize);

    DA_CUDA_CHECK(cudaFree(d_gradLift));
}

// ============================================================================

void discreteAdjointComputeRHS(DiscreteAdjointGPUData& da,
                               const GPUSolverData& gpu,
                               bool usePsiTmp)
{
    const double* d_psi_in = usePsiTmp ? da.d_psiTmp : da.d_psi;
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;
    int nqFace = gpu.nqFace;
    int coeffSize = NVAR_GPU * nE * nmodes;

    const double* psiCoeffPtr = d_psi_in;

    // Fused volume + surface integral
    int nwork    = NVAR_GPU * nmodes;
    int nFaceQP  = 4 * nqFace;
    int blockDim2 = ((nwork + nFaceQP + 31) / 32) * 32;
    int smemFused = (NVAR_GPU * nmodes + NVAR_GPU * nqVol + nFaceQP * 4) * sizeof(double);
    const double* d_eps_ptr = gpu.useAV ? gpu.d_epsilon : nullptr;
    const double* d_U_quad = gpu.modalMode ? gpu.d_U_quad : gpu.d_U;
    discreteAdjointVolumeSurfaceKernel<<<nE, blockDim2, smemFused>>>(
        psiCoeffPtr, d_U_quad, gpu.d_Ucoeff,
        da.d_adjRhsCoeff, 
        gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
        d_eps_ptr,
        nE, P1, nq1d, nmodes, nqVol, totalDOF, nqFace,
        gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf,
        gpu.fluxType);

    // AV volume diffusion (uses forward epsilon directly)
    if (gpu.useAV) {
        int bkAV = std::max(64, NVAR_GPU * nmodes);
        if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
        bkAV = std::max(bkAV, NVAR_GPU * nqVol);
        if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
        int smemAV = (NVAR_GPU * nmodes + 2 * NVAR_GPU * nqVol) * sizeof(double);
        daAVVolumeKernel<<<nE, bkAV, smemAV>>>(
            psiCoeffPtr, da.d_adjRhsCoeff,
            gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            gpu.d_epsilon,
            nE, P1, nq1d, nmodes, nqVol);
    }

    // Mass solve (coefficient-to-coefficient)
    int blockDim3 = std::max(64, NVAR_GPU * nmodes);
    if (blockDim3 % 32 != 0) blockDim3 = ((blockDim3 + 31) / 32) * 32;
    daMassSolveModalKernel<<<nE, blockDim3>>>(
        da.d_adjRhsCoeff, da.d_adjR, gpu.d_Minv,
        nE, nmodes);

    // Add objective gradient
    int bk5 = 256;
    int gd5 = (coeffSize + bk5 - 1) / bk5;
    daCoeffAddKernel<<<gd5, bk5>>>(da.d_adjR, da.d_dJdUcoeff, coeffSize);
}

void discreteAdjointRK4Stage(DiscreteAdjointGPUData& da, double dt,
                             int stage, int N)
{
    int bk = 256;
    int gd = (N + bk - 1) / bk;

    double* d_k;
    switch (stage) {
        case 1: d_k = da.d_pk1; break;
        case 2: d_k = da.d_pk2; break;
        case 3: d_k = da.d_pk3; break;
        default: d_k = da.d_pk4; break;
    }

    daRK4StageKernel<<<gd, bk>>>(
        da.d_psi, da.d_psiTmp, d_k, da.d_adjR,
        da.d_pk1, da.d_pk2, da.d_pk3,
        dt, stage, N);
}

bool discreteAdjointCheckNaN(DiscreteAdjointGPUData& da, int N)
{
    int zero = 0;
    DA_CUDA_CHECK(cudaMemcpy(da.d_nanFlag, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int bk = 256;
    int gd = (N + bk - 1) / bk;
    daNanCheckKernel<<<gd, bk>>>(da.d_psi, N, da.d_nanFlag);

    int flag;
    DA_CUDA_CHECK(cudaMemcpy(&flag, da.d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost));
    return (flag != 0);
}

double discreteAdjointResidualL2(DiscreteAdjointGPUData& da, int N)
{
    double zero = 0.0;
    DA_CUDA_CHECK(cudaMemcpy(da.d_dtMin, &zero, sizeof(double), cudaMemcpyHostToDevice));

    int bk = 256;
    int gd = std::min((N + bk - 1) / bk, 1024);
    int smem = bk * sizeof(double);
    daL2NormKernel<<<gd, bk, smem>>>(da.d_adjR, N, da.d_dtMin);

    double result;
    DA_CUDA_CHECK(cudaMemcpy(&result, da.d_dtMin, sizeof(double), cudaMemcpyDeviceToHost));
    return sqrt(result);
}

void discreteAdjointResidualNormPerVar(DiscreteAdjointGPUData& da,
                                       int dofPerVar, double norms[4])
{
    DA_CUDA_CHECK(cudaMemset(da.d_normBuf, 0, 4 * sizeof(double)));

    int bk = 256;
    int gd = std::min((dofPerVar + bk - 1) / bk, 1024);
    daResNormPerVarKernel<<<gd, bk, 4 * bk * sizeof(double)>>>(
        da.d_adjR, da.d_normBuf, dofPerVar);

    DA_CUDA_CHECK(cudaMemcpy(norms, da.d_normBuf, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int v = 0; v < 4; ++v)
        norms[v] = sqrt(norms[v]);
}

void discreteAdjointCopySolutionToHost(DiscreteAdjointGPUData& da,
                                       double* psi_flat, int N)
{
    DA_CUDA_CHECK(cudaMemcpy(psi_flat, da.d_psi, N * sizeof(double), cudaMemcpyDeviceToHost));
}

void discreteAdjointCopyQuadPointsToHost(DiscreteAdjointGPUData& da,
                                         const GPUSolverData& gpu,
                                         double* psi_quad_flat)
{
    if (da.modalMode) {
        int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
        int nmodes = gpu.nmodes, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;
        int blockBT = std::max(64, NVAR_GPU * nqVol);
        if (blockBT % 32 != 0) blockBT = ((blockBT + 31) / 32) * 32;
        daBackwardTransformKernel<<<nE, blockBT>>>(
            da.d_psi, da.d_psi_quad,
            nE, P1, nq1d, nmodes, nqVol, totalDOF);
        DA_CUDA_CHECK(cudaMemcpy(psi_quad_flat, da.d_psi_quad,
                       NVAR_GPU * totalDOF * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        DA_CUDA_CHECK(cudaMemcpy(psi_quad_flat, da.d_psi,
                       NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

void discreteAdjointCopySolutionToDevice(DiscreteAdjointGPUData& da,
                                         const double* psi_flat, int N)
{
    DA_CUDA_CHECK(cudaMemcpy(da.d_psi, psi_flat, N * sizeof(double), cudaMemcpyHostToDevice));
}
