#include "adjoint2d_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>

#define ADJ_CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static constexpr double GAMMA_ADJ = 1.4;

#define MAX_P1   8
#define MAX_NQ1D 10

__constant__ double ac_Bmat[MAX_P1 * MAX_NQ1D];
__constant__ double ac_Dmat[MAX_P1 * MAX_NQ1D];
__constant__ double ac_blr[MAX_P1 * 2];
__constant__ double ac_wq[MAX_NQ1D];
__constant__ double ac_NodalToModal[MAX_P1 * MAX_P1];

// ============================================================================
// Device helper functions
// ============================================================================

__device__ inline double pressure_ad(double rho, double rhou, double rhov, double rhoE)
{
    double u = rhou / rho;
    double v = rhov / rho;
    return (GAMMA_ADJ - 1.0) * (rhoE - 0.5 * rho * (u * u + v * v));
}

__device__ inline double soundSpeed_ad(double rho, double p)
{
    return sqrt(GAMMA_ADJ * p / rho);
}

__device__ inline double evalPhiFace_ad(int lf, int i, int j, int q,
                                        int P1, int nq1d)
{
    int nqF = nq1d;
    double phiXi, phiEta;
    switch (lf) {
        case 0: phiXi = ac_Bmat[i * nq1d + q];             phiEta = ac_blr[j * 2 + 0]; break;
        case 1: phiXi = ac_blr[i * 2 + 1];                 phiEta = ac_Bmat[j * nq1d + q]; break;
        case 2: phiXi = ac_Bmat[i * nq1d + (nqF - 1 - q)]; phiEta = ac_blr[j * 2 + 1]; break;
        case 3: phiXi = ac_blr[i * 2 + 0];                 phiEta = ac_Bmat[j * nq1d + (nqF - 1 - q)]; break;
        default: phiXi = 0; phiEta = 0;
    }
    return phiXi * phiEta;
}

// ============================================================================
// Euler flux Jacobians
// ============================================================================

__device__ void eulerFluxJacX_ad(const double U[4], double A[4][4])
{
    const double gm1 = GAMMA_ADJ - 1.0;
    double rho = U[0], u = U[1]/rho, v = U[2]/rho;
    double q2 = u*u + v*v;
    double p = gm1 * (U[3] - 0.5*rho*q2);
    double H = (U[3] + p) / rho;

    A[0][0] = 0.0;            A[0][1] = 1.0;           A[0][2] = 0.0;      A[0][3] = 0.0;
    A[1][0] = 0.5*gm1*q2-u*u; A[1][1] = (3.0-GAMMA_ADJ)*u; A[1][2] = -gm1*v; A[1][3] = gm1;
    A[2][0] = -u*v;           A[2][1] = v;             A[2][2] = u;         A[2][3] = 0.0;
    A[3][0] = u*(0.5*gm1*q2-H); A[3][1] = H-gm1*u*u;  A[3][2] = -gm1*u*v;  A[3][3] = GAMMA_ADJ*u;
}

__device__ void eulerFluxJacY_ad(const double U[4], double B[4][4])
{
    const double gm1 = GAMMA_ADJ - 1.0;
    double rho = U[0], u = U[1]/rho, v = U[2]/rho;
    double q2 = u*u + v*v;
    double p = gm1 * (U[3] - 0.5*rho*q2);
    double H = (U[3] + p) / rho;

    B[0][0] = 0.0;            B[0][1] = 0.0;           B[0][2] = 1.0;      B[0][3] = 0.0;
    B[1][0] = -u*v;           B[1][1] = v;             B[1][2] = u;         B[1][3] = 0.0;
    B[2][0] = 0.5*gm1*q2-v*v; B[2][1] = -gm1*u;       B[2][2] = (3.0-GAMMA_ADJ)*v; B[2][3] = gm1;
    B[3][0] = v*(0.5*gm1*q2-H); B[3][1] = -gm1*u*v;   B[3][2] = H-gm1*v*v; B[3][3] = GAMMA_ADJ*v;
}

__device__ void normalFluxJac_ad(const double U[4], double nx, double ny, double An[4][4])
{
    double A[4][4], B[4][4];
    eulerFluxJacX_ad(U, A);
    eulerFluxJacY_ad(U, B);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            An[i][j] = A[i][j]*nx + B[i][j]*ny;
}

__device__ void slipWallBC_ad(const double UL[4], double nx, double ny, double UR[4])
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

__device__ void riemannBC_ad(const double UL[4], double nx, double ny,
                             double rhoInf, double uInf, double vInf, double pInf,
                             double UR[4])
{
    const double gm1 = GAMMA_ADJ - 1.0;
    double rhoI = UL[0], uI = UL[1]/rhoI, vI = UL[2]/rhoI;
    double pI = pressure_ad(UL[0], UL[1], UL[2], UL[3]);
    double cI = sqrt(GAMMA_ADJ * pI / rhoI);
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

    double cInfVal = sqrt(GAMMA_ADJ * pInf / rhoInf);
    double VnInf = uInf*nx + vInf*ny;
    double Rplus  = VnI + 2.0*cI/gm1;
    double Rminus = VnInf - 2.0*cInfVal/gm1;
    double sB, VtB_x, VtB_y;
    if (VnI < 0.0) {
        sB = pInf / pow(rhoInf, GAMMA_ADJ);
        VtB_x = uInf - VnInf*nx;
        VtB_y = vInf - VnInf*ny;
    } else {
        sB = pI / pow(rhoI, GAMMA_ADJ);
        VtB_x = uI - VnI*nx;
        VtB_y = vI - VnI*ny;
    }
    double VnB = 0.5*(Rplus + Rminus);
    double cB  = 0.25*gm1*(Rplus - Rminus);
    if (cB < 1e-14) cB = 1e-14;
    double rhoB = pow(cB*cB/(GAMMA_ADJ*sB), 1.0/gm1);
    double uB = VtB_x + VnB*nx;
    double vB = VtB_y + VnB*ny;
    double pB = rhoB*cB*cB/GAMMA_ADJ;
    UR[0] = rhoB;
    UR[1] = rhoB*uB;
    UR[2] = rhoB*vB;
    UR[3] = pB/gm1 + 0.5*rhoB*(uB*uB + vB*vB);
}

__device__ void slipWallJac_ad(double nx, double ny, double Rbc[4][4])
{
    double nx2 = nx*nx, ny2 = ny*ny, nxny = nx*ny;
    Rbc[0][0] = 1.0; Rbc[0][1] = 0.0;        Rbc[0][2] = 0.0;        Rbc[0][3] = 0.0;
    Rbc[1][0] = 0.0; Rbc[1][1] = 1.0-2.0*nx2; Rbc[1][2] = -2.0*nxny;  Rbc[1][3] = 0.0;
    Rbc[2][0] = 0.0; Rbc[2][1] = -2.0*nxny;   Rbc[2][2] = 1.0-2.0*ny2; Rbc[2][3] = 0.0;
    Rbc[3][0] = 0.0; Rbc[3][1] = 0.0;        Rbc[3][2] = 0.0;        Rbc[3][3] = 1.0;
}

__device__ void riemannBCJac_ad(const double UL[4], double nx, double ny,
                                double rhoInf, double uInf, double vInf, double pInf,
                                double Rbc[4][4])
{
    double UR0[4];
    riemannBC_ad(UL, nx, ny, rhoInf, uInf, vInf, pInf, UR0);
    for (int w = 0; w < 4; ++w) {
        double ULp[4] = {UL[0], UL[1], UL[2], UL[3]};
        double h = 1e-7 * fmax(fabs(ULp[w]), 1.0);
        ULp[w] += h;
        double URp[4];
        riemannBC_ad(ULp, nx, ny, rhoInf, uInf, vInf, pInf, URp);
        for (int v = 0; v < 4; ++v)
            Rbc[v][w] = (URp[v] - UR0[v]) / h;
    }
}

// ============================================================================
// HLLC numerical flux (mirror of forward solver for adjoint FD Jacobian)
// ============================================================================

__device__ void hllc_adjoint(const double UL[4], const double UR[4],
                             double nx, double ny, double Fnum[4])
{
    const double gm1 = GAMMA_ADJ - 1.0;
    const double RHOMIN = 1e-10;
    const double PMIN   = 1e-10;

    double rhoL = fmax(UL[0], RHOMIN), uL = UL[1]/rhoL, vL = UL[2]/rhoL;
    double pL   = fmax(pressure_ad(rhoL, UL[1], UL[2], UL[3]), PMIN);
    double cL   = sqrt(GAMMA_ADJ * pL / rhoL);
    double EL   = UL[3];
    double HL   = (EL + pL) / rhoL;

    double rhoR = fmax(UR[0], RHOMIN), uR = UR[1]/rhoR, vR = UR[2]/rhoR;
    double pR   = fmax(pressure_ad(rhoR, UR[1], UR[2], UR[3]), PMIN);
    double cR   = sqrt(GAMMA_ADJ * pR / rhoR);
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
// FD-based adjoint flux Jacobian–vector product: (dh/dU_side)^T * dpsi
//   side=0 ⟹ perturb UL,  side=1 ⟹ perturb UR
// ============================================================================

__device__ void hllcAdjFluxFD(const double UL[4], const double UR[4],
                              double nx, double ny,
                              const double dpsi[4], int side,
                              double adj_flux[4])
{
    double h0[4];
    hllc_adjoint(UL, UR, nx, ny, h0);

    for (int w = 0; w < 4; ++w) {
        double ULp[4] = {UL[0], UL[1], UL[2], UL[3]};
        double URp[4] = {UR[0], UR[1], UR[2], UR[3]};
        double* Up = (side == 0) ? ULp : URp;
        double fd_h = 1e-7 * fmax(fabs(Up[w]), 1.0);
        Up[w] += fd_h;
        double hp[4];
        hllc_adjoint(ULp, URp, nx, ny, hp);
        double val = 0.0;
        for (int v = 0; v < 4; ++v)
            val += (hp[v] - h0[v]) / fd_h * dpsi[v];
        adj_flux[w] = val;
    }
}

// ============================================================================
// Kernel: Compute frozen wave speeds at face quad points
// ============================================================================

__global__ void frozenAlphaKernel(
    const double* __restrict__ d_Ucoeff,
    double* __restrict__ d_alphaFace,
    const double* __restrict__ d_faceNx,
    const double* __restrict__ d_faceNy,
    const int* __restrict__ d_face_elemL,
    const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_faceL,
    const int* __restrict__ d_face_faceR,
    const int* __restrict__ d_face_bcType,
    int nF, int nE, int P1, int nq1d, int nmodes, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * nqFace) return;

    int f = idx / nqFace;
    int q = idx % nqFace;

    int eL = d_face_elemL[f];
    int eR = d_face_elemR[f];
    int lfL = d_face_faceL[f];

    double nx = d_faceNx[f * nqFace + q];
    double ny = d_faceNy[f * nqFace + q];

    double UL[4] = {0,0,0,0};
    for (int vv = 0; vv < 4; ++vv)
        for (int ii = 0; ii < P1; ++ii)
            for (int jj = 0; jj < P1; ++jj)
                UL[vv] += d_Ucoeff[vv*nE*nmodes + eL*nmodes + ii*P1+jj]
                        * evalPhiFace_ad(lfL, ii, jj, q, P1, nq1d);

    double UR[4];
    if (eR >= 0) {
        int lfR = d_face_faceR[f];
        int qN = nqFace - 1 - q;
        for (int vv = 0; vv < 4; ++vv) {
            UR[vv] = 0.0;
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    UR[vv] += d_Ucoeff[vv*nE*nmodes + eR*nmodes + ii*P1+jj]
                            * evalPhiFace_ad(lfR, ii, jj, qN, P1, nq1d);
        }
    } else {
        int bcType = d_face_bcType[f];
        if (bcType == 1)
            slipWallBC_ad(UL, nx, ny, UR);
        else if (bcType == 2)
            riemannBC_ad(UL, nx, ny, rhoInf, uInf, vInf, pInf, UR);
        else
            for (int vv = 0; vv < 4; ++vv) UR[vv] = UL[vv];
    }

    double rhoL = UL[0], uL_ = UL[1]/rhoL, vL_ = UL[2]/rhoL;
    double pL = pressure_ad(UL[0], UL[1], UL[2], UL[3]);
    double HL = (UL[3] + pL) / rhoL;

    double rhoR = UR[0], uR_ = UR[1]/rhoR, vR_ = UR[2]/rhoR;
    double pR = pressure_ad(UR[0], UR[1], UR[2], UR[3]);
    double HR = (UR[3] + pR) / rhoR;

    double srL = sqrt(rhoL), srR = sqrt(rhoR), srLR = srL + srR;
    double uRoe = (srL*uL_ + srR*uR_) / srLR;
    double vRoe = (srL*vL_ + srR*vR_) / srLR;
    double HRoe = (srL*HL  + srR*HR)  / srLR;
    double qRoe2 = uRoe*uRoe + vRoe*vRoe;
    double cRoe = sqrt(fmax((GAMMA_ADJ - 1.0)*(HRoe - 0.5*qRoe2), 1e-14));
    double alpha = fabs(uRoe*nx + vRoe*ny) + cRoe;

    d_alphaFace[f * nqFace + q] = alpha;
}

// ============================================================================
// Kernel: Adjoint project  ψ(quad) → ψ_coeff = Minv * Φ^T * ψ
// ============================================================================

__global__ void adjointProjectKernel(
    const double* __restrict__ d_psi,
    double* __restrict__ d_psiCoeff,
    const double* __restrict__ d_Minv,
    const double* __restrict__ d_detJ,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF,
    const int* __restrict__ d_elemIdx)
{
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork = NVAR_GPU * nmodes;

    extern __shared__ double smem[];
    double* phiT_psi = smem;

    int nmodesLocal = P1 * P1;
    for (int w = tid; w < nwork; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;
        if (m >= nmodesLocal) { phiT_psi[w] = 0.0; continue; }
        int mi = m / P1;
        int mj = m % P1;

        double val = 0.0;
        for (int qx = 0; qx < nq1d; ++qx) {
            double Bi = ac_Bmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qIdx = e * nqVol + qx * nq1d + qe;
                double weight = ac_wq[qx] * ac_wq[qe] * d_detJ[qIdx];
                val += weight * Bi * ac_Bmat[mj * nq1d + qe]
                     * d_psi[v * totalDOF + qIdx];
            }
        }
        phiT_psi[w] = val;
    }
    __syncthreads();

    for (int w = tid; w < nwork; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;

        double val = 0.0;
        if (m < nmodesLocal) {
            for (int mp = 0; mp < nmodesLocal; ++mp)
                val += d_Minv[e * nmodes * nmodes + m * nmodes + mp]
                     * phiT_psi[v * nmodes + mp];
        }
        d_psiCoeff[v * nE * nmodes + e * nmodes + m] = val;
    }
}

// ============================================================================
// Kernel: Adjoint volume integral (coefficient space)
//   adj_vol_coeff[w,m] = Σ_q w_q * detJ * φ_m(q) * Σ_v (A^T_wv * gx^v + B^T_wv * gy^v)
//   where gx^v = Σ_n ψ_coeff[v,n] * ∂φ_n/∂x(q)
// ============================================================================

__global__ void adjointVolumeKernel(
    const double* __restrict__ d_psiCoeff,
    const double* __restrict__ d_U,
    double* __restrict__ d_adjVolCoeff,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF,
    const int* __restrict__ d_elemIdx)
{
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;

    extern __shared__ double smem[];
    double* sPsiCoeff = smem;
    double* sU = sPsiCoeff + NVAR_GPU * nmodes;

    for (int i = tid; i < NVAR_GPU * nmodes; i += blockDim.x) {
        int v = i / nmodes;
        int m = i % nmodes;
        sPsiCoeff[i] = d_psiCoeff[v * nE * nmodes + e * nmodes + m];
    }
    int baseQ = e * nqVol;
    for (int i = tid; i < nqVol; i += blockDim.x) {
        int gIdx = baseQ + i;
        sU[0*nqVol + i] = d_U[0*totalDOF + gIdx];
        sU[1*nqVol + i] = d_U[1*totalDOF + gIdx];
        sU[2*nqVol + i] = d_U[2*totalDOF + gIdx];
        sU[3*nqVol + i] = d_U[3*totalDOF + gIdx];
    }
    __syncthreads();

    int nwork = NVAR_GPU * nmodes;
    int nmodesLocal = P1 * P1;
    for (int work = tid; work < nwork; work += blockDim.x) {
        int w  = work / nmodes;
        int m  = work % nmodes;
        if (m >= nmodesLocal) { d_adjVolCoeff[w * nE * nmodes + e * nmodes + m] = 0.0; continue; }
        int mi = m / P1;
        int mj = m % P1;

        double result = 0.0;

        for (int qx = 0; qx < nq1d; ++qx) {
            double Bi_m = ac_Bmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                int gIdx   = baseQ + qLocal;

                double Bj_m = ac_Bmat[mj * nq1d + qe];
                double phi_m = Bi_m * Bj_m;
                double wqdetJ = ac_wq[qx] * ac_wq[qe] * d_detJ[gIdx];

                double gx[4] = {0,0,0,0};
                double gy[4] = {0,0,0,0};
                for (int i = 0; i < P1; ++i) {
                    double Bi = ac_Bmat[i * nq1d + qx];
                    double Di = ac_Dmat[i * nq1d + qx];
                    for (int j = 0; j < P1; ++j) {
                        double Bj = ac_Bmat[j * nq1d + qe];
                        double Dj = ac_Dmat[j * nq1d + qe];
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
                eulerFluxJacX_ad(Uq, A);
                eulerFluxJacY_ad(Uq, B_mat);

                double val = 0.0;
                for (int v = 0; v < 4; ++v)
                    val += A[v][w] * gx[v] + B_mat[v][w] * gy[v];

                result += wqdetJ * phi_m * val;
            }
        }

        d_adjVolCoeff[w * nE * nmodes + e * nmodes + m] = result;
    }
}

// ============================================================================
// Kernel: Adjoint surface integral
// ============================================================================

__global__ void adjointSurfaceKernel(
    const double* __restrict__ d_psiCoeff,
    const double* __restrict__ d_Ucoeff,
    double* __restrict__ d_adjRhsCoeff,
    const double* __restrict__ d_alphaFace,
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
    int nE, int P1, int nq1d, int nmodes, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf,
    int fluxType,
    const int* __restrict__ d_elemIdx)
{
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork   = NVAR_GPU * nmodes;
    int nFaceQP = 4 * nqFace;

    extern __shared__ double smem[];
    double* sPsiCoeff = smem;
    double* sUcoeff   = sPsiCoeff + NVAR_GPU * nmodes;
    double* sFnumW    = sUcoeff + NVAR_GPU * nmodes;

    for (int i = tid; i < NVAR_GPU * nmodes; i += blockDim.x) {
        int v = i / nmodes;
        int m = i % nmodes;
        sPsiCoeff[i] = d_psiCoeff[v * nE * nmodes + e * nmodes + m];
        sUcoeff[i]   = d_Ucoeff[v * nE * nmodes + e * nmodes + m];
    }
    for (int i = tid; i < nFaceQP * 4; i += blockDim.x)
        sFnumW[i] = 0.0;
    __syncthreads();

    if (tid < nFaceQP) {
        int lf = tid / nqFace;
        int q  = tid % nqFace;

        int f  = d_elem2face[e * 4 + lf];
        int eL = d_face_elemL[f];
        int eR = d_face_elemR[f];
        bool is_left = (e == eL);
        int eN  = is_left ? eR : eL;
        int lfN = is_left ? d_face_faceR[f] : d_face_faceL[f];
        bool is_boundary = (eN < 0);

        int q_face = is_left ? q : (nqFace - 1 - q);
        int fIdx = f * nqFace + q_face;
        double nx = d_faceNx[fIdx];
        double ny = d_faceNy[fIdx];
        double wf = ac_wq[q] * d_faceJac[fIdx];
        double alpha = d_alphaFace[fIdx];

        double psi_me[4] = {0,0,0,0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    psi_me[vv] += sPsiCoeff[vv*nmodes + ii*P1+jj]
                                * evalPhiFace_ad(lf, ii, jj, q, P1, nq1d);

        double U_me[4] = {0,0,0,0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    U_me[vv] += sUcoeff[vv*nmodes + ii*P1+jj]
                              * evalPhiFace_ad(lf, ii, jj, q, P1, nq1d);

        double adj_flux[4] = {0,0,0,0};

        if (!is_boundary) {
            int qN = nqFace - 1 - q;

            double psi_nb[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        psi_nb[vv] += d_psiCoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                    * evalPhiFace_ad(lfN, ii, jj, qN, P1, nq1d);

            double U_nb[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        U_nb[vv] += d_Ucoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                  * evalPhiFace_ad(lfN, ii, jj, qN, P1, nq1d);

            double UL[4], UR_st[4], psi_L[4], psi_R[4];
            if (is_left) {
                for (int vv=0;vv<4;++vv) { UL[vv]=U_me[vv]; UR_st[vv]=U_nb[vv]; psi_L[vv]=psi_me[vv]; psi_R[vv]=psi_nb[vv]; }
            } else {
                for (int vv=0;vv<4;++vv) { UL[vv]=U_nb[vv]; UR_st[vv]=U_me[vv]; psi_L[vv]=psi_nb[vv]; psi_R[vv]=psi_me[vv]; }
            }

            double dpsi_RL[4];
            for (int vv=0;vv<4;++vv) dpsi_RL[vv] = psi_R[vv] - psi_L[vv];

            if (fluxType == 1) {
                int side = is_left ? 0 : 1;
                hllcAdjFluxFD(UL, UR_st, nx, ny, dpsi_RL, side, adj_flux);
            } else {
                double An_L[4][4], An_R[4][4];
                normalFluxJac_ad(UL, nx, ny, An_L);
                normalFluxJac_ad(UR_st, nx, ny, An_R);

                if (is_left) {
                    for (int w = 0; w < 4; ++w) {
                        double val = 0.5 * alpha * dpsi_RL[w];
                        for (int v = 0; v < 4; ++v)
                            val += 0.5 * An_L[v][w] * dpsi_RL[v];
                        adj_flux[w] = val;
                    }
                } else {
                    for (int w = 0; w < 4; ++w) {
                        double val = -0.5 * alpha * dpsi_RL[w];
                        for (int v = 0; v < 4; ++v)
                            val += 0.5 * An_R[v][w] * dpsi_RL[v];
                        adj_flux[w] = val;
                    }
                }
            }

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
            int bcType = d_face_bcType[f];

            if (bcType == 2) {
                // Farfield: non-reflecting adjoint BC (ψ_ghost = 0).
                // adj_flux = -(∂h/∂UL)^T · ψ_me, dropping the ∂h/∂UR·Rbc
                // term that causes spurious reflection of adjoint waves.
                double UR_bc[4];
                riemannBC_ad(U_me, nx, ny, rhoInf, uInf, vInf, pInf, UR_bc);

                if (fluxType == 1) {
                    hllcAdjFluxFD(U_me, UR_bc, nx, ny, psi_me, 0, adj_flux);
                    for (int w = 0; w < 4; ++w) adj_flux[w] = -adj_flux[w];
                } else {
                    double An_L[4][4];
                    normalFluxJac_ad(U_me, nx, ny, An_L);
                    for (int w = 0; w < 4; ++w) {
                        double val = 0.5 * alpha * psi_me[w];
                        for (int v = 0; v < 4; ++v)
                            val += 0.5 * An_L[v][w] * psi_me[v];
                        adj_flux[w] = -val;
                    }
                }

                if (d_epsilon != nullptr) {
                    double eps_me = d_epsilon[e];
                    if (eps_me > 0.0) {
                        double sigma = eps_me * (double)(P1*P1)
                                     / fmax(2.0 * d_faceJac[fIdx], 1e-30);
                        for (int w = 0; w < 4; ++w)
                            adj_flux[w] -= sigma * psi_me[w];
                    }
                }

            } else {
                // Wall (bcType==1) or other: discrete adjoint BC
                double UR_bc[4];
                double Rbc[4][4];

                if (bcType == 1) {
                    slipWallBC_ad(U_me, nx, ny, UR_bc);
                    slipWallJac_ad(nx, ny, Rbc);
                } else {
                    for (int vv=0;vv<4;++vv) UR_bc[vv] = U_me[vv];
                    for (int i=0;i<4;++i) for (int j=0;j<4;++j) Rbc[i][j] = (i==j)?1.0:0.0;
                }

                if (fluxType == 1) {
                    double h0[4];
                    hllc_adjoint(U_me, UR_bc, nx, ny, h0);

                    for (int w = 0; w < 4; ++w) {
                        double ULp[4] = {U_me[0], U_me[1], U_me[2], U_me[3]};
                        double fd_h = 1e-7 * fmax(fabs(ULp[w]), 1.0);
                        ULp[w] += fd_h;
                        double UR_bc_p[4];
                        if (bcType == 1)
                            slipWallBC_ad(ULp, nx, ny, UR_bc_p);
                        else
                            for (int vv=0;vv<4;++vv) UR_bc_p[vv] = ULp[vv];
                        double hp[4];
                        hllc_adjoint(ULp, UR_bc_p, nx, ny, hp);
                        double val = 0.0;
                        for (int v = 0; v < 4; ++v)
                            val += (hp[v] - h0[v]) / fd_h * psi_me[v];
                        adj_flux[w] = -val;
                    }
                } else {
                    double An_L[4][4], An_R[4][4];
                    normalFluxJac_ad(U_me, nx, ny, An_L);
                    normalFluxJac_ad(UR_bc, nx, ny, An_R);

                    double dh_dUL[4][4], dh_dUR[4][4];
                    for (int i=0;i<4;++i)
                        for (int j=0;j<4;++j) {
                            dh_dUL[i][j] = 0.5*(An_L[i][j] + ((i==j)?alpha:0.0));
                            dh_dUR[i][j] = 0.5*(An_R[i][j] - ((i==j)?alpha:0.0));
                        }

                    double total[4][4];
                    for (int i=0;i<4;++i)
                        for (int j=0;j<4;++j) {
                            total[i][j] = dh_dUL[i][j];
                            for (int k=0;k<4;++k)
                                total[i][j] += dh_dUR[i][k] * Rbc[k][j];
                        }

                    for (int w = 0; w < 4; ++w) {
                        double val = 0.0;
                        for (int v = 0; v < 4; ++v)
                            val += total[v][w] * psi_me[v];
                        adj_flux[w] = -val;
                    }
                }

                if (d_epsilon != nullptr) {
                    double eps_me = d_epsilon[e];
                    if (eps_me > 0.0) {
                        double sigma = eps_me * (double)(P1*P1)
                                     / fmax(2.0 * d_faceJac[fIdx], 1e-30);
                        for (int w = 0; w < 4; ++w) {
                            double ip_val = psi_me[w];
                            for (int v = 0; v < 4; ++v)
                                ip_val -= Rbc[v][w] * psi_me[v];
                            adj_flux[w] += -sigma * ip_val;
                        }
                    }
                }
            }
        }

        for (int w = 0; w < 4; ++w)
            sFnumW[tid * 4 + w] = wf * adj_flux[w];
    }
    __syncthreads();

    for (int work = tid; work < nwork; work += blockDim.x) {
        int w  = work / nmodes;
        int m  = work % nmodes;
        int mi = m / P1;
        int mj = m % P1;

        double surfResult = 0.0;
        for (int fq = 0; fq < nFaceQP; ++fq) {
            int lf = fq / nqFace;
            int q  = fq % nqFace;
            double phi = evalPhiFace_ad(lf, mi, mj, q, P1, nq1d);
            surfResult += sFnumW[fq * 4 + w] * phi;
        }
        d_adjRhsCoeff[w * nE * nmodes + e * nmodes + m] = surfResult;
    }
}

// ============================================================================
// Kernel: Adjoint AV volume (same bilinear form as forward, self-adjoint)
// ============================================================================

__global__ void adjointAVVolumeKernel(
    const double* __restrict__ d_psiCoeff,
    double* __restrict__ d_adjRhsCoeff,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    const double* __restrict__ d_epsilon,
    int nE, int P1, int nq1d, int nmodes, int nqVol,
    const int* __restrict__ d_elemIdx)
{
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
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
            double Di_qx = ac_Dmat[i * nq1d + qx];
            double Bi_qx = ac_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j) {
                double c = sCoeff[v * nmodes + i * P1 + j];
                dxi  += c * Di_qx * ac_Bmat[j * nq1d + qe];
                deta += c * Bi_qx * ac_Dmat[j * nq1d + qe];
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
            double Bi = ac_Bmat[mi * nq1d + qx];
            double Di = ac_Dmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                int gIdx   = e * nqVol + qLocal;
                double wq  = ac_wq[qx] * ac_wq[qe] * d_detJ[gIdx];

                double Bj = ac_Bmat[mj * nq1d + qe];
                double Dj = ac_Dmat[mj * nq1d + qe];

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
// Kernel: Adjoint mass-solve backward (Φ * Minv * coeff → quad pts)
// ============================================================================

__global__ void adjointMSBKernel(
    const double* __restrict__ d_coeff,
    double* __restrict__ d_out,
    const double* __restrict__ d_Minv,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF,
    const int* __restrict__ d_elemIdx)
{
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork_modes = NVAR_GPU * nmodes;

    extern __shared__ double smem[];
    double* temp = smem;

    for (int w = tid; w < nwork_modes; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;

        double val = 0.0;
        for (int mp = 0; mp < nmodes; ++mp)
            val += d_Minv[e * nmodes * nmodes + m * nmodes + mp]
                 * d_coeff[v * nE * nmodes + e * nmodes + mp];

        temp[w] = val;
    }
    __syncthreads();

    int nwork_quad = NVAR_GPU * nqVol;
    for (int w = tid; w < nwork_quad; w += blockDim.x) {
        int v  = w / nqVol;
        int q  = w % nqVol;
        int qx = q / nq1d;
        int qe = q % nq1d;

        double val = 0.0;
        for (int i = 0; i < P1; ++i) {
            double Bi = ac_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j)
                val += temp[v * nmodes + i * P1 + j] * Bi * ac_Bmat[j * nq1d + qe];
        }

        d_out[v * totalDOF + e * nqVol + q] = val;
    }
}

// ============================================================================
// Kernel: Combine adjoint RHS at quad points (unweighted)
//   d_adjR[i] = d_adjRhsQuad[i] + d_dJdU[i]
// ============================================================================

__global__ void adjointCombineKernel(
    double* __restrict__ d_adjR,
    const double* __restrict__ d_adjRhsQuad,
    const double* __restrict__ d_dJdU,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    d_adjR[idx] = d_adjRhsQuad[idx] + d_dJdU[idx];
}

// ============================================================================
// Kernel: Add coefficients  d_out[i] += d_in[i]
// ============================================================================

__global__ void coeffAddKernel(
    double* __restrict__ d_out,
    const double* __restrict__ d_in,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_out[idx] += d_in[idx];
}

// ============================================================================
// Kernel: Adjoint RK4 stage (no positivity enforcement)
// ============================================================================

__global__ void adjointRK4StageKernel(
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
// Kernel: Objective gradient in coefficient space (dJ̃/dUcoeff)
//   J = -∫_wall p * ny ds / normalization
//   Only nonzero for elements with wall boundary faces (bcType=1)
// ============================================================================

__global__ void objectiveGradientCoeffKernel(
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
    double normalization, double forceNx, double forceNy,
    const int* __restrict__ d_elemIdx)
{
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
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
                int q_face = q;
                int fIdx = f * nqFace + q_face;
                double nx = d_faceNx[fIdx];
                double ny = d_faceNy[fIdx];
                double wf = ac_wq[q] * d_faceJac[fIdx];

                double Uf[4] = {0,0,0,0};
                for (int vv = 0; vv < 4; ++vv)
                    for (int ii = 0; ii < P1; ++ii)
                        for (int jj = 0; jj < P1; ++jj)
                            Uf[vv] += d_Ucoeff[vv*nE*nmodes + e*nmodes + ii*P1+jj]
                                    * evalPhiFace_ad(lf, ii, jj, q, P1, nq1d);

                double rho = Uf[0];
                double u_v = Uf[1]/rho, v_v = Uf[2]/rho;
                double q2 = u_v*u_v + v_v*v_v;
                const double gm1 = GAMMA_ADJ - 1.0;

                double dpdU[4];
                dpdU[0] = gm1 * 0.5 * q2;
                dpdU[1] = -gm1 * u_v;
                dpdU[2] = -gm1 * v_v;
                dpdU[3] = gm1;

                double phi = evalPhiFace_ad(lf, mi, mj, q, P1, nq1d);

                result += wf * dpdU[w] * (forceNx * nx + forceNy * ny) * phi / normalization;
            }
        }

        d_dJdUcoeff[w * nE * nmodes + e * nmodes + m] = result;
    }
}

// ============================================================================
// Kernel: Lift computation (reduction)
// ============================================================================

__global__ void forceReductionKernel(
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
    double normalization, double forceNx, double forceNy,
    const int* __restrict__ d_elemIdx)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int e = (d_elemIdx != nullptr) ? d_elemIdx[blockIdx.x] : blockIdx.x;
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
                double wf = ac_wq[q] * d_faceJac[fIdx];

                double Uf[4] = {0,0,0,0};
                for (int vv = 0; vv < 4; ++vv)
                    for (int ii = 0; ii < P1; ++ii)
                        for (int jj = 0; jj < P1; ++jj)
                            Uf[vv] += d_Ucoeff[vv*nE*nmodes + e*nmodes + ii*P1+jj]
                                    * evalPhiFace_ad(lf, ii, jj, q, P1, nq1d);

                double p = pressure_ad(Uf[0], Uf[1], Uf[2], Uf[3]);
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
// Kernel: Adjoint smoothness sensor (Persson-Peraire on adjoint coefficients)
// ============================================================================

__global__ void adjointSensorKernel(
    const double* __restrict__ d_psiCoeff,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    const double* __restrict__ d_fwdEpsilon,
    double* __restrict__ d_adjEpsilon,
    double* __restrict__ d_adjSensor,
    int nE, int P1, int nmodes, int nqVol,
    double s0, double kappa, double smaxRef, double avScale,
    bool capFwdEps,
    const int* __restrict__ d_elemIdx)
{
    int eIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eIdx >= nE) return;
    int e = (d_elemIdx != nullptr) ? d_elemIdx[eIdx] : eIdx;

    int P = P1 - 1;

    double hMax = 0.0;
    int base = e * nqVol;
    for (int q = 0; q < nqVol; ++q) {
        int i = base + q;
        double gXi  = sqrt(d_dxidx[i]*d_dxidx[i] + d_dxidy[i]*d_dxidy[i]);
        double gEta = sqrt(d_detadx[i]*d_detadx[i] + d_detady[i]*d_detady[i]);
        double h = 1.0 / fmin(gXi, gEta);
        hMax = fmax(hMax, h);
    }

    double epsilon0 = avScale * hMax * smaxRef / fmax(2.0 * P, 1.0);

    double maxSe = -20.0;
    for (int var = 0; var < NVAR_GPU; ++var) {
        double c_modal[MAX_P1 * MAX_P1];
        for (int k = 0; k < P1; ++k)
            for (int l = 0; l < P1; ++l) {
                double val = 0.0;
                for (int ii = 0; ii < P1; ++ii) {
                    double Tki = ac_NodalToModal[k * P1 + ii];
                    for (int jj = 0; jj < P1; ++jj)
                        val += Tki * ac_NodalToModal[l * P1 + jj]
                             * d_psiCoeff[var * nE * nmodes + e * nmodes + ii * P1 + jj];
                }
                c_modal[k * P1 + l] = val;
            }

        int highDegThresh = max(2 * P - 1, P + 1);
        double total_energy = 0.0;
        double high_energy  = 0.0;
        for (int ii = 0; ii < P1; ++ii)
            for (int jj = 0; jj < P1; ++jj) {
                double c = c_modal[ii * P1 + jj];
                total_energy += c * c;
                if (ii + jj >= highDegThresh)
                    high_energy += c * c;
            }

        double se = (total_energy > 1e-30) ? log10(high_energy / total_energy) : -20.0;
        maxSe = fmax(maxSe, se);
    }

    double eps;
    if (maxSe < s0 - kappa)
        eps = 0.0;
    else if (maxSe > s0 + kappa)
        eps = epsilon0;
    else
        eps = 0.5 * epsilon0 * (1.0 + sin(M_PI * (maxSe - s0) / (2.0 * kappa)));

    double fwdFloor = capFwdEps ? fmin(d_fwdEpsilon[e], epsilon0)
                                : d_fwdEpsilon[e];
    d_adjEpsilon[e] = fmax(eps, fwdFloor);
    d_adjSensor[e]  = maxSe;
}

// ============================================================================
// Kernel: NaN check
// ============================================================================

__global__ void adjointNanCheckKernel(const double* __restrict__ data, int N,
                                      int* __restrict__ flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && (isnan(data[i]) || isinf(data[i])))
        *flag = 1;
}

// ============================================================================
// Kernel: L2 norm reduction
// ============================================================================

__device__ double atomicAddDouble_adj(double* addr, double val)
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

__global__ void l2NormKernel(const double* __restrict__ data, int N,
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
        atomicAddDouble_adj(d_result, sdata[0]);
}

// ============================================================================
// Kernel: Adjoint backward transform (coefficients -> quad points)
//   Uses ac_Bmat constant memory (adjoint compilation unit).
// ============================================================================

__global__ void adjointBackwardTransformKernel(
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
            double Bi = ac_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j)
                val += d_coeffs[v * nE * nmodes + e * nmodes + i * P1 + j]
                     * Bi * ac_Bmat[j * nq1d + qe];
        }

        d_quad[v * totalDOF + e * nqVol + q] = val;
    }
}

// ============================================================================
// Kernel: Adjoint mass solve (modal mode, coefficient-to-coefficient)
//   dPsi/dt_coeff = Minv * rhsCoeff  (no backward transform)
// ============================================================================

__global__ void adjointMassSolveModalKernel(
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
// Kernel: Fused adjoint volume + surface integral (1 block per element)
//   Mirrors the forward solver's volumeSurfaceKernel warp-partitioning:
//   Phase 1: Load psiCoeff, U(quad), metrics into shared memory
//   Phase 2: Volume integral   -- threads [0, NVAR*nmodes)
//   Phase 3: Surface flux      -- threads [NVAR*nmodes, NVAR*nmodes+4*nqFace)
//   Phase 4: Surface integral  -- threads [0, NVAR*nmodes) accumulate
//   Phases 2 and 3 run concurrently on separate warps.
// ============================================================================

__global__ void adjointVolumeSurfaceKernel(
    const double* __restrict__ d_psiCoeff,
    const double* __restrict__ d_U,          // quad-point values of frozen state
    const double* __restrict__ d_Ucoeff,
    double* __restrict__ d_rhsCoeff,
    const double* __restrict__ d_alphaFace,
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
    int fluxType,
    const int* __restrict__ d_elemIdx)
{
    int eL = blockIdx.x;
    if (eL >= nE) return;
    int e = (d_elemIdx != nullptr) ? d_elemIdx[eL] : eL;

    int tid = threadIdx.x;
    int nwork   = NVAR_GPU * nmodes;
    int nFaceQP = 4 * nqFace;

    // Shared memory layout:
    //   sPsiCoeff [4*nmodes] + sU [4*nqVol] + sFnumW [nFaceQP*4]
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
            double Bi_m = ac_Bmat[mi_id * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                int gIdx   = baseQ + qLocal;

                double Bj_m = ac_Bmat[mj_id * nq1d + qe];
                double phi_m = Bi_m * Bj_m;
                double wqdetJ = ac_wq[qx] * ac_wq[qe] * d_detJ[gIdx];

                double gx[4] = {0,0,0,0};
                double gy[4] = {0,0,0,0};
                for (int i = 0; i < P1; ++i) {
                    double Bi = ac_Bmat[i * nq1d + qx];
                    double Di = ac_Dmat[i * nq1d + qx];
                    for (int j = 0; j < P1; ++j) {
                        double Bj = ac_Bmat[j * nq1d + qe];
                        double Dj = ac_Dmat[j * nq1d + qe];
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
                eulerFluxJacX_ad(Uq, A);
                eulerFluxJacY_ad(Uq, B_mat);

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
        double wf = ac_wq[q] * d_faceJac[fIdx];
        double alpha = d_alphaFace[fIdx];

        double psi_me[4] = {0,0,0,0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    psi_me[vv] += sPsiCoeff[vv*nmodes + ii*P1+jj]
                                * evalPhiFace_ad(lf, ii, jj, q, P1, nq1d);

        double U_me[4] = {0,0,0,0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    U_me[vv] += d_Ucoeff[vv*nE*nmodes + e*nmodes + ii*P1+jj]
                              * evalPhiFace_ad(lf, ii, jj, q, P1, nq1d);

        double adj_flux[4] = {0,0,0,0};

        if (!is_boundary) {
            int qN = nqFace - 1 - q;

            double psi_nb[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        psi_nb[vv] += d_psiCoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                    * evalPhiFace_ad(lfN, ii, jj, qN, P1, nq1d);

            double U_nb[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        U_nb[vv] += d_Ucoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                  * evalPhiFace_ad(lfN, ii, jj, qN, P1, nq1d);

            double UL[4], UR_st[4], psi_L[4], psi_R[4];
            if (is_left) {
                for (int vv=0;vv<4;++vv) { UL[vv]=U_me[vv]; UR_st[vv]=U_nb[vv]; psi_L[vv]=psi_me[vv]; psi_R[vv]=psi_nb[vv]; }
            } else {
                for (int vv=0;vv<4;++vv) { UL[vv]=U_nb[vv]; UR_st[vv]=U_me[vv]; psi_L[vv]=psi_nb[vv]; psi_R[vv]=psi_me[vv]; }
            }

            double dpsi_RL[4];
            for (int vv=0;vv<4;++vv) dpsi_RL[vv] = psi_R[vv] - psi_L[vv];

            if (fluxType == 1) {
                int side = is_left ? 0 : 1;
                hllcAdjFluxFD(UL, UR_st, nx, ny, dpsi_RL, side, adj_flux);
            } else {
                double An_L[4][4], An_R[4][4];
                normalFluxJac_ad(UL, nx, ny, An_L);
                normalFluxJac_ad(UR_st, nx, ny, An_R);

                if (is_left) {
                    for (int w = 0; w < 4; ++w) {
                        double val = 0.5 * alpha * dpsi_RL[w];
                        for (int v = 0; v < 4; ++v)
                            val += 0.5 * An_L[v][w] * dpsi_RL[v];
                        adj_flux[w] = val;
                    }
                } else {
                    for (int w = 0; w < 4; ++w) {
                        double val = -0.5 * alpha * dpsi_RL[w];
                        for (int v = 0; v < 4; ++v)
                            val += 0.5 * An_R[v][w] * dpsi_RL[v];
                        adj_flux[w] = val;
                    }
                }
            }

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
            int bcType = d_face_bcType[f];

            double UR_bc[4];
            double Rbc[4][4];

            if (bcType == 1) {
                slipWallBC_ad(U_me, nx, ny, UR_bc);
                slipWallJac_ad(nx, ny, Rbc);
            } else if (bcType == 2) {
                riemannBC_ad(U_me, nx, ny, rhoInf, uInf, vInf, pInf, UR_bc);
                riemannBCJac_ad(U_me, nx, ny, rhoInf, uInf, vInf, pInf, Rbc);
            } else {
                for (int vv=0;vv<4;++vv) UR_bc[vv] = U_me[vv];
                for (int i=0;i<4;++i) for (int j=0;j<4;++j) Rbc[i][j] = (i==j)?1.0:0.0;
            }

            if (fluxType == 1) {
                double h0[4];
                hllc_adjoint(U_me, UR_bc, nx, ny, h0);

                for (int w = 0; w < 4; ++w) {
                    double ULp[4] = {U_me[0], U_me[1], U_me[2], U_me[3]};
                    double fd_h = 1e-7 * fmax(fabs(ULp[w]), 1.0);
                    ULp[w] += fd_h;
                    double UR_bc_p[4];
                    if (bcType == 1)
                        slipWallBC_ad(ULp, nx, ny, UR_bc_p);
                    else if (bcType == 2)
                        riemannBC_ad(ULp, nx, ny, rhoInf, uInf, vInf, pInf, UR_bc_p);
                    else
                        for (int vv=0;vv<4;++vv) UR_bc_p[vv] = ULp[vv];
                    double hp[4];
                    hllc_adjoint(ULp, UR_bc_p, nx, ny, hp);
                    double val = 0.0;
                    for (int v = 0; v < 4; ++v)
                        val += (hp[v] - h0[v]) / fd_h * psi_me[v];
                    adj_flux[w] = -val;
                }
            } else {
                double An_L[4][4], An_R[4][4];
                normalFluxJac_ad(U_me, nx, ny, An_L);
                normalFluxJac_ad(UR_bc, nx, ny, An_R);

                double dh_dUL[4][4], dh_dUR[4][4];
                for (int i=0;i<4;++i)
                    for (int j=0;j<4;++j) {
                        dh_dUL[i][j] = 0.5*(An_L[i][j] + ((i==j)?alpha:0.0));
                        dh_dUR[i][j] = 0.5*(An_R[i][j] - ((i==j)?alpha:0.0));
                    }

                double total[4][4];
                for (int i=0;i<4;++i)
                    for (int j=0;j<4;++j) {
                        total[i][j] = dh_dUL[i][j];
                        for (int k=0;k<4;++k)
                            total[i][j] += dh_dUR[i][k] * Rbc[k][j];
                    }

                for (int w = 0; w < 4; ++w) {
                    double val = 0.0;
                    for (int v = 0; v < 4; ++v)
                        val += total[v][w] * psi_me[v];
                    adj_flux[w] = -val;
                }
            }

            if (d_epsilon != nullptr) {
                double eps_me = d_epsilon[e];
                if (eps_me > 0.0) {
                    double sigma = eps_me * (double)(P1*P1)
                                 / fmax(2.0 * d_faceJac[fIdx], 1e-30);
                    for (int w = 0; w < 4; ++w) {
                        double ip_val = psi_me[w];
                        for (int v = 0; v < 4; ++v)
                            ip_val -= Rbc[v][w] * psi_me[v];
                        adj_flux[w] += -sigma * ip_val;
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
            double phi = evalPhiFace_ad(lf, mi_id, mj_id, q, P1, nq1d);
            surfResult += sFnumW[fq * 4 + v_id] * phi;
        }
        d_rhsCoeff[v_id * nE * nmodes + e * nmodes + m_id] = volResult + surfResult;
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

void adjointGpuAllocate(AdjointGPUData& adj, const GPUSolverData& gpu)
{
    adj.modalMode  = gpu.modalMode;
    adj.totalCoeff = gpu.nE * gpu.nmodes;
    adj.primaryDOF = adj.modalMode ? adj.totalCoeff : gpu.totalDOF;

    int solSize   = NVAR_GPU * adj.primaryDOF;
    int coeffSize = NVAR_GPU * gpu.nE * gpu.nmodes;
    int quadSize  = NVAR_GPU * gpu.totalDOF;

    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_psi,    solSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_psiTmp, solSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_pk1,    solSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_pk2,    solSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_pk3,    solSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_pk4,    solSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_adjR,   solSize * sizeof(double)));

    if (adj.modalMode) {
        adj.d_psiCoeff = nullptr;  // modal: psi IS coefficients, no separate buffer
    } else {
        ADJ_CUDA_CHECK(cudaMalloc(&adj.d_psiCoeff, coeffSize * sizeof(double)));
    }
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_adjVolCoeff, coeffSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_adjRhsCoeff, coeffSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_adjSurfQuad, quadSize * sizeof(double)));

    adj.d_psi_quad   = nullptr;
    adj.d_dJdU       = nullptr;
    adj.d_dJdUcoeff  = nullptr;
    if (adj.modalMode) {
        ADJ_CUDA_CHECK(cudaMalloc(&adj.d_psi_quad,  quadSize * sizeof(double)));
        ADJ_CUDA_CHECK(cudaMalloc(&adj.d_dJdUcoeff, coeffSize * sizeof(double)));
        ADJ_CUDA_CHECK(cudaMemset(adj.d_dJdUcoeff, 0, coeffSize * sizeof(double)));
    } else {
        ADJ_CUDA_CHECK(cudaMalloc(&adj.d_dJdU, quadSize * sizeof(double)));
        ADJ_CUDA_CHECK(cudaMemset(adj.d_dJdU, 0, quadSize * sizeof(double)));
    }

    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_alphaFace, gpu.nF * gpu.nqFace * sizeof(double)));

    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_dtMin,   sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_nanFlag, sizeof(int)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_liftBuf, gpu.nE * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_normBuf, 4 * sizeof(double)));

    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_adjEpsilon, gpu.nE * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMalloc(&adj.d_adjSensor,  gpu.nE * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemset(adj.d_adjEpsilon, 0, gpu.nE * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemset(adj.d_adjSensor,  0, gpu.nE * sizeof(double)));

    ADJ_CUDA_CHECK(cudaMemset(adj.d_psi, 0, solSize * sizeof(double)));

    adj.chordRef = 1.0;
    adj.fullAV   = false;
}

void adjointGpuFree(AdjointGPUData& adj)
{
    cudaFree(adj.d_psi);     cudaFree(adj.d_psiTmp);
    cudaFree(adj.d_pk1);     cudaFree(adj.d_pk2);
    cudaFree(adj.d_pk3);     cudaFree(adj.d_pk4);
    cudaFree(adj.d_adjR);
    if (adj.d_psiCoeff) cudaFree(adj.d_psiCoeff);
    cudaFree(adj.d_adjVolCoeff);
    cudaFree(adj.d_adjRhsCoeff); cudaFree(adj.d_adjSurfQuad);
    if (adj.d_dJdU) cudaFree(adj.d_dJdU);
    if (adj.d_dJdUcoeff) cudaFree(adj.d_dJdUcoeff);
    if (adj.d_psi_quad) cudaFree(adj.d_psi_quad);
    cudaFree(adj.d_alphaFace);
    cudaFree(adj.d_dtMin);   cudaFree(adj.d_nanFlag);
    cudaFree(adj.d_liftBuf); cudaFree(adj.d_normBuf);
    cudaFree(adj.d_adjEpsilon); cudaFree(adj.d_adjSensor);
}

void adjointGpuSetBasisData(const double* Bmat, const double* Dmat,
                            const double* blr, const double* wq,
                            int P1, int nq1d)
{
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_Bmat, Bmat, P1*nq1d*sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_Dmat, Dmat, P1*nq1d*sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_blr,  blr,  P1*2*sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_wq,   wq,   nq1d*sizeof(double)));
}

void adjointGpuSetNodalToModal(const double* T, int P1)
{
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_NodalToModal, T, P1*P1*sizeof(double)));
}

void adjointComputeFrozenAlpha(AdjointGPUData& adj, const GPUSolverData& gpu)
{
    int total = gpu.nF * gpu.nqFace;
    int bk = 256;
    int gd = (total + bk - 1) / bk;
    frozenAlphaKernel<<<gd, bk>>>(
        gpu.d_Ucoeff, adj.d_alphaFace,
        gpu.d_faceNx, gpu.d_faceNy,
        gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_faceL, gpu.d_face_faceR,
        gpu.d_face_bcType,
        gpu.nF, gpu.nE, gpu.P1, gpu.nq1d, gpu.nmodes, gpu.nqFace,
        gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf);
}

double adjointComputeForceCoeff(AdjointGPUData& adj, const GPUSolverData& gpu,
                                double chordRef, double forceNx, double forceNy)
{
    int nE = gpu.nE;
    double Vinf2 = gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf;
    double normalization = 0.5 * gpu.rhoInf * Vinf2 * chordRef;
    if (normalization < 1e-30) normalization = 1.0;

    int bk = 32;
    int smem = bk * sizeof(double);
    forceReductionKernel<<<nE, bk, smem>>>(
        gpu.d_Ucoeff,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_bcType,
        adj.d_liftBuf,
        nE, gpu.P1, gpu.nq1d, gpu.nmodes, gpu.nqFace,
        normalization, forceNx, forceNy, nullptr);

    std::vector<double> buf(nE);
    ADJ_CUDA_CHECK(cudaMemcpy(buf.data(), adj.d_liftBuf, nE*sizeof(double), cudaMemcpyDeviceToHost));
    double result = 0.0;
    for (int i = 0; i < nE; ++i) result += buf[i];
    return result;
}

void adjointComputeObjectiveGradient(AdjointGPUData& adj, const GPUSolverData& gpu,
                                     double chordRef, double forceNx, double forceNy)
{
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;

    double Vinf2 = gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf;
    double normalization = 0.5 * gpu.rhoInf * Vinf2 * chordRef;
    if (normalization < 1e-30) normalization = 1.0;

    int coeffSize = NVAR_GPU * nE * nmodes;
    ADJ_CUDA_CHECK(cudaMemset(adj.d_adjRhsCoeff, 0, coeffSize * sizeof(double)));

    int bk = std::max(64, NVAR_GPU * nmodes);
    if (bk % 32 != 0) bk = ((bk + 31) / 32) * 32;
    objectiveGradientCoeffKernel<<<nE, bk>>>(
        gpu.d_Ucoeff, adj.d_adjRhsCoeff,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_bcType,
        nE, P1, nq1d, nmodes, gpu.nqFace,
        normalization, forceNx, forceNy, nullptr);

    if (adj.modalMode) {
        // Store dJ/dU as coefficients (mass-solve only, no backward transform)
        int bk2 = std::max(64, NVAR_GPU * nmodes);
        if (bk2 % 32 != 0) bk2 = ((bk2 + 31) / 32) * 32;
        adjointMassSolveModalKernel<<<nE, bk2>>>(
            adj.d_adjRhsCoeff, adj.d_dJdUcoeff, gpu.d_Minv,
            nE, nmodes);
    } else {
        // Store dJ/dU at quad points (mass-solve + backward transform)
        int bk2 = std::max(64, NVAR_GPU * nmodes);
        if (bk2 % 32 != 0) bk2 = ((bk2 + 31) / 32) * 32;
        int smem2 = NVAR_GPU * nmodes * sizeof(double);
        adjointMSBKernel<<<nE, bk2, smem2>>>(
            adj.d_adjRhsCoeff, adj.d_dJdU, gpu.d_Minv,
            nE, P1, nq1d, nmodes, nqVol, totalDOF, nullptr);
    }
}

void adjointComputeRHS(AdjointGPUData& adj, const GPUSolverData& gpu, bool usePsiTmp)
{
    const double* d_psi_in = usePsiTmp ? adj.d_psiTmp : adj.d_psi;
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;
    int nqFace = gpu.nqFace;
    int coeffSize = NVAR_GPU * nE * nmodes;

    // AV sensor helper (shared by both paths)
    auto launchAVSensor = [&](const double* psiCoeffPtr) {
        if (!gpu.useAV) return;
        int bkS = 256;
        int gdS = (nE + bkS - 1) / bkS;
        double uMag = sqrt(gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf);
        double cInf = sqrt(1.4 * gpu.pInf / gpu.rhoInf);
        double smaxRef = uMag + cInf;
        int P = P1 - 1;
        double adjAVscale;
        bool capFwdEps;
        if (adj.fullAV) {
            adjAVscale = (double)gpu.AVscale;
            capFwdEps = false;
        } else {
            adjAVscale = std::min((double)gpu.AVscale,
                                  0.5 * P / (0.75 * (2.0 * P + 1.0)));
            capFwdEps = true;
        }
        adjointSensorKernel<<<gdS, bkS>>>(
            psiCoeffPtr,
            gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            gpu.d_epsilon, adj.d_adjEpsilon, adj.d_adjSensor,
            nE, P1, nmodes, nqVol,
            gpu.AVs0, gpu.AVkappa, smaxRef, adjAVscale, capFwdEps, nullptr);
    };

    if (adj.modalMode) {
        // ===== MODAL (coefficient-space) path =====
        // d_psi_in already contains coefficients.
        const double* psiCoeffPtr = d_psi_in;

        // Backward transform: psi coefficients -> quad-point working buffer
        int blockBT = std::max(64, NVAR_GPU * nqVol);
        if (blockBT % 32 != 0) blockBT = ((blockBT + 31) / 32) * 32;
        adjointBackwardTransformKernel<<<nE, blockBT>>>(
            psiCoeffPtr, adj.d_psi_quad,
            nE, P1, nq1d, nmodes, nqVol, totalDOF);

        // Shock sensor (reads coefficients directly)
        launchAVSensor(psiCoeffPtr);

        // Fused volume + surface integral
        int nwork    = NVAR_GPU * nmodes;
        int nFaceQP  = 4 * nqFace;
        int blockDim2 = ((nwork + nFaceQP + 31) / 32) * 32;
        int smemFused = (NVAR_GPU * nmodes + NVAR_GPU * nqVol + nFaceQP * 4) * sizeof(double);
        const double* d_eps_ptr = gpu.useAV ? adj.d_adjEpsilon : nullptr;
        const double* d_U_quad = gpu.modalMode ? gpu.d_U_quad : gpu.d_U;
        adjointVolumeSurfaceKernel<<<nE, blockDim2, smemFused>>>(
            psiCoeffPtr, d_U_quad, gpu.d_Ucoeff,
            adj.d_adjRhsCoeff, adj.d_alphaFace,
            gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
            gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
            gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
            d_eps_ptr,
            nE, P1, nq1d, nmodes, nqVol, totalDOF, nqFace,
            gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf,
            gpu.fluxType, nullptr);

        // AV volume diffusion (reads coefficients)
        if (gpu.useAV) {
            int bkAV = std::max(64, NVAR_GPU * nmodes);
            if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
            bkAV = std::max(bkAV, NVAR_GPU * nqVol);
            if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
            int smemAV = (NVAR_GPU * nmodes + 2 * NVAR_GPU * nqVol) * sizeof(double);
            adjointAVVolumeKernel<<<nE, bkAV, smemAV>>>(
                psiCoeffPtr, adj.d_adjRhsCoeff,
                gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
                adj.d_adjEpsilon,
                nE, P1, nq1d, nmodes, nqVol, nullptr);
        }

        // Mass solve only (coefficient-to-coefficient, no backward transform)
        int blockDim3 = std::max(64, NVAR_GPU * nmodes);
        if (blockDim3 % 32 != 0) blockDim3 = ((blockDim3 + 31) / 32) * 32;
        adjointMassSolveModalKernel<<<nE, blockDim3>>>(
            adj.d_adjRhsCoeff, adj.d_adjR, gpu.d_Minv,
            nE, nmodes);

        // Add objective gradient in coefficient space
        int bk5 = 256;
        int gd5 = (coeffSize + bk5 - 1) / bk5;
        coeffAddKernel<<<gd5, bk5>>>(adj.d_adjR, adj.d_dJdUcoeff, coeffSize);

    } else {
        // ===== NODAL (quadrature-point) path =====

        // Step 1: L2 project ψ (quad pts) → ψ_coeff (modal)
        int bk1 = std::max(64, NVAR_GPU * nmodes);
        if (bk1 % 32 != 0) bk1 = ((bk1 + 31) / 32) * 32;
        int smem1 = NVAR_GPU * nmodes * sizeof(double);
        adjointProjectKernel<<<nE, bk1, smem1>>>(
            d_psi_in, adj.d_psiCoeff, gpu.d_Minv, gpu.d_detJ,
            nE, P1, nq1d, nmodes, nqVol, totalDOF, nullptr);

        // Shock sensor
        launchAVSensor(adj.d_psiCoeff);

        // Step 2: Volume integral → coefficient space
        int bk2 = std::max(64, NVAR_GPU * nmodes);
        if (bk2 % 32 != 0) bk2 = ((bk2 + 31) / 32) * 32;
        int smem2 = (NVAR_GPU * nmodes + NVAR_GPU * nqVol) * sizeof(double);
        adjointVolumeKernel<<<nE, bk2, smem2>>>(
            adj.d_psiCoeff, gpu.d_U, adj.d_adjVolCoeff,
            gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            nE, P1, nq1d, nmodes, nqVol, totalDOF, nullptr);

        // Step 3: Surface integral → coefficient space
        int nFaceQP = 4 * nqFace;
        int bk3 = std::max(nFaceQP, NVAR_GPU * nmodes);
        if (bk3 % 32 != 0) bk3 = ((bk3 + 31) / 32) * 32;
        bk3 = std::max(bk3, 32);
        int smem3 = (2 * NVAR_GPU * nmodes + nFaceQP * 4) * sizeof(double);
        const double* d_eps_ptr = gpu.useAV ? adj.d_adjEpsilon : nullptr;
        adjointSurfaceKernel<<<nE, bk3, smem3>>>(
            adj.d_psiCoeff, gpu.d_Ucoeff, adj.d_adjRhsCoeff,
            adj.d_alphaFace,
            gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
            gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
            gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
            d_eps_ptr,
            nE, P1, nq1d, nmodes, nqFace,
            gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf,
            gpu.fluxType, nullptr);

        if (gpu.useAV) {
            int bkAV = std::max(64, NVAR_GPU * nmodes);
            if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
            bkAV = std::max(bkAV, NVAR_GPU * nqVol);
            if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
            int smemAV = (NVAR_GPU * nmodes + 2 * NVAR_GPU * nqVol) * sizeof(double);
            adjointAVVolumeKernel<<<nE, bkAV, smemAV>>>(
                adj.d_psiCoeff, adj.d_adjRhsCoeff,
                gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
                adj.d_adjEpsilon,
                nE, P1, nq1d, nmodes, nqVol, nullptr);
        }

        // Step 4: Combine volume + surface in coefficient space
        {
            int bkA = 256;
            int gdA = (coeffSize + bkA - 1) / bkA;
            coeffAddKernel<<<gdA, bkA>>>(adj.d_adjRhsCoeff, adj.d_adjVolCoeff, coeffSize);
        }

        // Step 5: Mass-solve backward → unweighted quad-point RHS
        int bk4 = std::max(64, NVAR_GPU * nmodes);
        if (bk4 % 32 != 0) bk4 = ((bk4 + 31) / 32) * 32;
        int smem4 = NVAR_GPU * nmodes * sizeof(double);
        adjointMSBKernel<<<nE, bk4, smem4>>>(
            adj.d_adjRhsCoeff, adj.d_adjR, gpu.d_Minv,
            nE, P1, nq1d, nmodes, nqVol, totalDOF, nullptr);

        // Step 6: Add source term (quad-point dJ/dU)
        int N = NVAR_GPU * totalDOF;
        int bk5 = 256;
        int gd5 = (N + bk5 - 1) / bk5;
        adjointCombineKernel<<<gd5, bk5>>>(adj.d_adjR, adj.d_adjR, adj.d_dJdU, N);
    }
}

void adjointRK4Stage(AdjointGPUData& adj, double dt, int stage, int N)
{
    int bk = 256;
    int gd = (N + bk - 1) / bk;

    double* d_k;
    switch (stage) {
        case 1: d_k = adj.d_pk1; break;
        case 2: d_k = adj.d_pk2; break;
        case 3: d_k = adj.d_pk3; break;
        default: d_k = adj.d_pk4; break;
    }

    adjointRK4StageKernel<<<gd, bk>>>(
        adj.d_psi, adj.d_psiTmp, d_k, adj.d_adjR,
        adj.d_pk1, adj.d_pk2, adj.d_pk3,
        dt, stage, N);
}

bool adjointCheckNaN(AdjointGPUData& adj, int N)
{
    int zero = 0;
    ADJ_CUDA_CHECK(cudaMemcpy(adj.d_nanFlag, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int bk = 256;
    int gd = (N + bk - 1) / bk;
    adjointNanCheckKernel<<<gd, bk>>>(adj.d_psi, N, adj.d_nanFlag);

    int flag;
    ADJ_CUDA_CHECK(cudaMemcpy(&flag, adj.d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost));
    return (flag != 0);
}

double adjointResidualL2(AdjointGPUData& adj, int N)
{
    double zero = 0.0;
    ADJ_CUDA_CHECK(cudaMemcpy(adj.d_dtMin, &zero, sizeof(double), cudaMemcpyHostToDevice));

    int bk = 256;
    int gd = std::min((N + bk - 1) / bk, 1024);
    int smem = bk * sizeof(double);
    l2NormKernel<<<gd, bk, smem>>>(adj.d_adjR, N, adj.d_dtMin);

    double result;
    ADJ_CUDA_CHECK(cudaMemcpy(&result, adj.d_dtMin, sizeof(double), cudaMemcpyDeviceToHost));
    return sqrt(result);
}

__global__ void adjResNormPerVarKernel(
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
        atomicAddDouble_adj(&d_norms[0], sdata[0]);
        atomicAddDouble_adj(&d_norms[1], sdata[bs]);
        atomicAddDouble_adj(&d_norms[2], sdata[2 * bs]);
        atomicAddDouble_adj(&d_norms[3], sdata[3 * bs]);
    }
}

void adjointResidualNormPerVar(AdjointGPUData& adj, int dofPerVar, double norms[4])
{
    ADJ_CUDA_CHECK(cudaMemset(adj.d_normBuf, 0, 4 * sizeof(double)));

    int bk = 256;
    int gd = std::min((dofPerVar + bk - 1) / bk, 1024);
    adjResNormPerVarKernel<<<gd, bk, 4 * bk * sizeof(double)>>>(
        adj.d_adjR, adj.d_normBuf, dofPerVar);

    ADJ_CUDA_CHECK(cudaMemcpy(norms, adj.d_normBuf, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int v = 0; v < 4; ++v)
        norms[v] = sqrt(norms[v]);
}

void adjointCopySolutionToHost(AdjointGPUData& adj, double* psi_flat, int N)
{
    ADJ_CUDA_CHECK(cudaMemcpy(psi_flat, adj.d_psi, N * sizeof(double), cudaMemcpyDeviceToHost));
}

void adjointCopyQuadPointsToHost(AdjointGPUData& adj, const GPUSolverData& gpu,
                                 double* psi_quad_flat)
{
    if (adj.modalMode) {
        int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
        int nmodes = gpu.nmodes, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;
        int blockBT = std::max(64, NVAR_GPU * nqVol);
        if (blockBT % 32 != 0) blockBT = ((blockBT + 31) / 32) * 32;
        adjointBackwardTransformKernel<<<nE, blockBT>>>(
            adj.d_psi, adj.d_psi_quad,
            nE, P1, nq1d, nmodes, nqVol, totalDOF);
        ADJ_CUDA_CHECK(cudaMemcpy(psi_quad_flat, adj.d_psi_quad,
                       NVAR_GPU * totalDOF * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        ADJ_CUDA_CHECK(cudaMemcpy(psi_quad_flat, adj.d_psi,
                       NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

void adjointCopySolutionToDevice(AdjointGPUData& adj, const double* psi_flat, int N)
{
    ADJ_CUDA_CHECK(cudaMemcpy(adj.d_psi, psi_flat, N * sizeof(double), cudaMemcpyHostToDevice));
}

// ============================================================================
// Variable-P adjoint support
// ============================================================================

void adjointAllocatePGroup(AdjointPGroup& grp, int P, int nEGroup)
{
    grp.P = P;
    grp.P1 = P + 1;
    grp.nmodes = (P + 1) * (P + 1);
    grp.nEGroup = nEGroup;
    ADJ_CUDA_CHECK(cudaMalloc(&grp.d_elemIdx, nEGroup * sizeof(int)));
}

void adjointFreePGroup(AdjointPGroup& grp)
{
    cudaFree(grp.d_elemIdx);
}

void adjointUploadPGroupElemIdx(AdjointPGroup& grp, const int* elemIdx)
{
    ADJ_CUDA_CHECK(cudaMemcpy(grp.d_elemIdx, elemIdx,
                   grp.nEGroup * sizeof(int), cudaMemcpyHostToDevice));
}

void adjointZeroCoeffArrays(AdjointGPUData& adj, const GPUSolverData& gpu)
{
    int coeffSize = NVAR_GPU * gpu.nE * gpu.nmodes;
    int solSize = NVAR_GPU * adj.primaryDOF;
    if (adj.d_psiCoeff)
        ADJ_CUDA_CHECK(cudaMemset(adj.d_psiCoeff, 0, coeffSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemset(adj.d_adjVolCoeff, 0, coeffSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemset(adj.d_adjRhsCoeff, 0, coeffSize * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemset(adj.d_adjR, 0, solSize * sizeof(double)));
}

void adjointAddObjectiveGradient(AdjointGPUData& adj, const GPUSolverData& gpu)
{
    if (adj.modalMode) {
        int N = NVAR_GPU * adj.totalCoeff;
        int bk = 256, gd = (N + bk - 1) / bk;
        coeffAddKernel<<<gd, bk>>>(adj.d_adjR, adj.d_dJdUcoeff, N);
    } else {
        int N = NVAR_GPU * gpu.totalDOF;
        int bk = 256, gd = (N + bk - 1) / bk;
        adjointCombineKernel<<<gd, bk>>>(adj.d_adjR, adj.d_adjR, adj.d_dJdU, N);
    }
}

void adjointUploadPGroupBasis(const AdjointPGroup& grp)
{
    int P1 = grp.P1;
    int nq1d = (int)grp.h_wq.size();
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_Bmat, grp.h_Bmat.data(), P1 * nq1d * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_Dmat, grp.h_Dmat.data(), P1 * nq1d * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_blr,  grp.h_blr.data(),  P1 * 2 * sizeof(double)));
    ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_wq,   grp.h_wq.data(),   nq1d * sizeof(double)));
    if (!grp.h_NodalToModal.empty())
        ADJ_CUDA_CHECK(cudaMemcpyToSymbol(ac_NodalToModal, grp.h_NodalToModal.data(),
                                          P1 * P1 * sizeof(double)));
}

void adjointComputeRHS_group(AdjointGPUData& adj, const GPUSolverData& gpu,
                             const AdjointPGroup& grp, bool usePsiTmp)
{
    const double* d_psi_in = usePsiTmp ? adj.d_psiTmp : adj.d_psi;
    int nE = gpu.nE;
    int nEG = grp.nEGroup;
    int P1 = grp.P1;
    int nmodes = gpu.nmodes;  // nmodes_max for global array stride
    int nq1d = gpu.nq1d, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;
    int nqFace = gpu.nqFace;
    int coeffSize = NVAR_GPU * nE * nmodes;

    adjointUploadPGroupBasis(grp);

    // Step 1: Project psi → psiCoeff (only this group's elements)
    int bk1 = std::max(64, NVAR_GPU * nmodes);
    if (bk1 % 32 != 0) bk1 = ((bk1 + 31) / 32) * 32;
    int smem1 = NVAR_GPU * nmodes * sizeof(double);
    adjointProjectKernel<<<nEG, bk1, smem1>>>(
        d_psi_in, adj.d_psiCoeff, gpu.d_Minv, gpu.d_detJ,
        nE, P1, nq1d, nmodes, nqVol, totalDOF, grp.d_elemIdx);

    // Step 1b: Adjoint sensor
    if (gpu.useAV) {
        int bkS = 256;
        int gdS = (nEG + bkS - 1) / bkS;
        double uMag = sqrt(gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf);
        double cInf = sqrt(1.4 * gpu.pInf / gpu.rhoInf);
        double smaxRef = uMag + cInf;
        int P = P1 - 1;
        double adjAVscale;
        bool capFwdEps;
        if (adj.fullAV) {
            adjAVscale = (double)gpu.AVscale;
            capFwdEps = false;
        } else {
            adjAVscale = std::min((double)gpu.AVscale,
                                  0.5 * P / (0.75 * (2.0 * P + 1.0)));
            capFwdEps = true;
        }
        adjointSensorKernel<<<gdS, bkS>>>(
            adj.d_psiCoeff,
            gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            gpu.d_epsilon, adj.d_adjEpsilon, adj.d_adjSensor,
            nE, P1, nmodes, nqVol,
            gpu.AVs0, gpu.AVkappa, smaxRef, adjAVscale, capFwdEps, grp.d_elemIdx);
    }

    // Step 2: Volume integral
    int bk2 = std::max(64, NVAR_GPU * nmodes);
    if (bk2 % 32 != 0) bk2 = ((bk2 + 31) / 32) * 32;
    int smem2 = (NVAR_GPU * nmodes + NVAR_GPU * nqVol) * sizeof(double);
    adjointVolumeKernel<<<nEG, bk2, smem2>>>(
        adj.d_psiCoeff, gpu.d_U, adj.d_adjVolCoeff,
        gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        nE, P1, nq1d, nmodes, nqVol, totalDOF, grp.d_elemIdx);

    // Step 3: Surface integral
    int nFaceQP = 4 * nqFace;
    int bk3 = std::max(nFaceQP, NVAR_GPU * nmodes);
    if (bk3 % 32 != 0) bk3 = ((bk3 + 31) / 32) * 32;
    bk3 = std::max(bk3, 32);
    int smem3 = (2 * NVAR_GPU * nmodes + nFaceQP * 4) * sizeof(double);
    const double* d_eps_ptr = gpu.useAV ? adj.d_adjEpsilon : nullptr;
    adjointSurfaceKernel<<<nEG, bk3, smem3>>>(
        adj.d_psiCoeff, gpu.d_Ucoeff, adj.d_adjRhsCoeff,
        adj.d_alphaFace,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
        d_eps_ptr,
        nE, P1, nq1d, nmodes, nqFace,
        gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf,
        gpu.fluxType, grp.d_elemIdx);

    // Step 3b: AV volume
    if (gpu.useAV) {
        int bkAV = std::max(64, NVAR_GPU * nmodes);
        if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
        bkAV = std::max(bkAV, NVAR_GPU * nqVol);
        if (bkAV % 32 != 0) bkAV = ((bkAV + 31) / 32) * 32;
        int smemAV = (NVAR_GPU * nmodes + 2 * NVAR_GPU * nqVol) * sizeof(double);
        adjointAVVolumeKernel<<<nEG, bkAV, smemAV>>>(
            adj.d_psiCoeff, adj.d_adjRhsCoeff,
            gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            adj.d_adjEpsilon,
            nE, P1, nq1d, nmodes, nqVol, grp.d_elemIdx);
    }

    // Step 4: Combine vol + surf coefficients (only this group's elements)
    // Note: coeffAddKernel operates on flat arrays, but we need to add only
    // this group's portion. With global arrays, we add the full array --
    // elements not in this group have zero contributions from the kernels above.
    {
        int bkA = 256;
        int gdA = (coeffSize + bkA - 1) / bkA;
        coeffAddKernel<<<gdA, bkA>>>(adj.d_adjRhsCoeff, adj.d_adjVolCoeff, coeffSize);
    }

    // Step 5: Mass-solve backward → quad-point RHS
    int bk4 = std::max(64, NVAR_GPU * nmodes);
    if (bk4 % 32 != 0) bk4 = ((bk4 + 31) / 32) * 32;
    int smem4 = NVAR_GPU * nmodes * sizeof(double);
    adjointMSBKernel<<<nEG, bk4, smem4>>>(
        adj.d_adjRhsCoeff, adj.d_adjR, gpu.d_Minv,
        nE, P1, nq1d, nmodes, nqVol, totalDOF, grp.d_elemIdx);
}
