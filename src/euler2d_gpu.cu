#include "euler2d_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cfloat>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static constexpr double GAMMA_GPU = 1.4;

#define MAX_P1   8
#define MAX_NQ1D 10

__constant__ double c_Bmat[MAX_P1 * MAX_NQ1D];
__constant__ double c_Dmat[MAX_P1 * MAX_NQ1D];
__constant__ double c_blr[MAX_P1 * 2];
__constant__ double c_wq[MAX_NQ1D];
__constant__ double c_NodalToModal[MAX_P1 * MAX_P1];

// ============================================================================
// Device helper functions
// ============================================================================

__device__ inline double pressure_d(double rho, double rhou, double rhov, double rhoE)
{
    double u = rhou / rho;
    double v = rhov / rho;
    return (GAMMA_GPU - 1.0) * (rhoE - 0.5 * rho * (u * u + v * v));
}

__device__ inline double soundSpeed_d(double rho, double p)
{
    return sqrt(GAMMA_GPU * p / rho);
}

__device__ void laxFriedrichs_d(const double UL[4], const double UR[4],
                                double nx, double ny, double Fnum[4])
{
    double rhoL = UL[0], uL = UL[1]/UL[0], vL = UL[2]/UL[0];
    double pL = pressure_d(UL[0], UL[1], UL[2], UL[3]);
    double HL = (UL[3] + pL) / rhoL;

    double rhoR = UR[0], uR = UR[1]/UR[0], vR = UR[2]/UR[0];
    double pR = pressure_d(UR[0], UR[1], UR[2], UR[3]);
    double HR = (UR[3] + pR) / rhoR;

    double FL[4], GL[4], FR[4], GR[4];
    FL[0] = UL[1];
    FL[1] = UL[1] * uL + pL;
    FL[2] = UL[1] * vL;
    FL[3] = (UL[3] + pL) * uL;
    GL[0] = UL[2];
    GL[1] = UL[2] * uL;
    GL[2] = UL[2] * vL + pL;
    GL[3] = (UL[3] + pL) * vL;

    FR[0] = UR[1];
    FR[1] = UR[1] * uR + pR;
    FR[2] = UR[1] * vR;
    FR[3] = (UR[3] + pR) * uR;
    GR[0] = UR[2];
    GR[1] = UR[2] * uR;
    GR[2] = UR[2] * vR + pR;
    GR[3] = (UR[3] + pR) * vR;

    double srL = sqrt(rhoL), srR = sqrt(rhoR), srLR = srL + srR;
    double uRoe = (srL * uL + srR * uR) / srLR;
    double vRoe = (srL * vL + srR * vR) / srLR;
    double HRoe = (srL * HL + srR * HR) / srLR;
    double qRoe2 = uRoe * uRoe + vRoe * vRoe;
    double cRoe = sqrt(fmax((GAMMA_GPU - 1.0) * (HRoe - 0.5 * qRoe2), 1e-14));
    double alpha = fabs(uRoe * nx + vRoe * ny) + cRoe;

    for (int n = 0; n < 4; ++n)
        Fnum[n] = 0.5 * ((FL[n]*nx + GL[n]*ny) + (FR[n]*nx + GR[n]*ny))
                - 0.5 * alpha * (UR[n] - UL[n]);
}

__device__ void hllc_d(const double UL[4], const double UR[4],
                       double nx, double ny, double Fnum[4])
{
    const double gm1 = GAMMA_GPU - 1.0;

    double rhoL = UL[0], uL = UL[1]/UL[0], vL = UL[2]/UL[0];
    double pL   = pressure_d(UL[0], UL[1], UL[2], UL[3]);
    double cL   = sqrt(GAMMA_GPU * pL / rhoL);
    double EL   = UL[3];
    double HL   = (EL + pL) / rhoL;

    double rhoR = UR[0], uR = UR[1]/UR[0], vR = UR[2]/UR[0];
    double pR   = pressure_d(UR[0], UR[1], UR[2], UR[3]);
    double cR   = sqrt(GAMMA_GPU * pR / rhoR);
    double ER   = UR[3];
    double HR   = (ER + pR) / rhoR;

    double vnL = uL*nx + vL*ny;
    double vnR = uR*nx + vR*ny;
    double vtL = -uL*ny + vL*nx;
    double vtR = -uR*ny + vR*nx;

    // Roe averages for wave speed estimates
    double srL = sqrt(rhoL), srR = sqrt(rhoR), srLR = srL + srR;
    double vnRoe = (srL*vnL + srR*vnR) / srLR;
    double HRoe  = (srL*HL  + srR*HR)  / srLR;
    double uRoe  = (srL*uL  + srR*uR)  / srLR;
    double vRoe  = (srL*vL  + srR*vR)  / srLR;
    double qRoe2 = uRoe*uRoe + vRoe*vRoe;
    double cRoe  = sqrt(fmax(gm1 * (HRoe - 0.5*qRoe2), 1e-14));

    double SL = fmin(vnL - cL, vnRoe - cRoe);
    double SR = fmax(vnR + cR, vnRoe + cRoe);

    // Contact wave speed
    double denom = rhoL*(SL - vnL) - rhoR*(SR - vnR);
    double SS = (pR - pL + rhoL*vnL*(SL - vnL) - rhoR*vnR*(SR - vnR))
              / fmax(fabs(denom), 1e-14) * ((denom >= 0.0) ? 1.0 : -1.0);

    if (SL >= 0.0) {
        // Supersonic from left
        Fnum[0] = (UL[1]*nx + UL[2]*ny);
        Fnum[1] = (UL[1]*uL + pL)*nx + UL[1]*vL*ny;
        Fnum[2] = UL[2]*uL*nx + (UL[2]*vL + pL)*ny;
        Fnum[3] = (EL + pL)*(uL*nx + vL*ny);
    } else if (SR <= 0.0) {
        // Supersonic from right
        Fnum[0] = (UR[1]*nx + UR[2]*ny);
        Fnum[1] = (UR[1]*uR + pR)*nx + UR[1]*vR*ny;
        Fnum[2] = UR[2]*uR*nx + (UR[2]*vR + pR)*ny;
        Fnum[3] = (ER + pR)*(uR*nx + vR*ny);
    } else {
        // Star region
        double rhoSK, unSK, utSK, ESK, rhoK, vnK, vtK, EK, pK, SK;
        double FnK[4];

        if (SS >= 0.0) {
            // Left star state
            rhoK = rhoL; vnK = vnL; vtK = vtL; EK = EL; pK = pL; SK = SL;
        } else {
            // Right star state
            rhoK = rhoR; vnK = vnR; vtK = vtR; EK = ER; pK = pR; SK = SR;
        }

        double factor = rhoK * (SK - vnK) / (SK - SS);
        rhoSK = factor;
        unSK  = SS;
        utSK  = vtK;
        ESK   = factor * (EK/rhoK + (SS - vnK)*(SS + pK/(rhoK*(SK - vnK))));

        // Normal flux for side K
        double uK = vnK*nx - vtK*ny;
        double vK = vnK*ny + vtK*nx;
        FnK[0] = rhoK*vnK;
        FnK[1] = (rhoK*uK*vnK + pK*nx);
        FnK[2] = (rhoK*vK*vnK + pK*ny);
        FnK[3] = (EK + pK)*vnK;

        // Star conserved state in physical frame
        double uSK = unSK*nx - utSK*ny;
        double vSK = unSK*ny + utSK*nx;
        double USK[4];
        USK[0] = rhoSK;
        USK[1] = rhoSK * uSK;
        USK[2] = rhoSK * vSK;
        USK[3] = ESK;

        double UK[4];
        UK[0] = rhoK;
        UK[1] = rhoK * uK;
        UK[2] = rhoK * vK;
        UK[3] = EK;

        for (int n = 0; n < 4; ++n)
            Fnum[n] = FnK[n] + SK * (USK[n] - UK[n]);
    }
}

__device__ void slipWallBC_d(const double UL[4], double nx, double ny, double UR[4])
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

__device__ void riemannBC_d(const double UL[4], double nx, double ny,
                            double rhoInf, double uInf, double vInf, double pInf,
                            double UR[4])
{
    const double gm1 = GAMMA_GPU - 1.0;

    double rhoI = UL[0];
    double uI = UL[1] / rhoI;
    double vI = UL[2] / rhoI;
    double pI = pressure_d(UL[0], UL[1], UL[2], UL[3]);
    double cI = sqrt(GAMMA_GPU * pI / rhoI);
    double VnI = uI * nx + vI * ny;

    double cInfVal = sqrt(GAMMA_GPU * pInf / rhoInf);
    double VnInf = uInf * nx + vInf * ny;

    double Rplus  = VnI + 2.0 * cI / gm1;
    double Rminus = VnInf - 2.0 * cInfVal / gm1;
    double sB, VtB_x, VtB_y;

    if (VnI < 0.0) {
        sB    = pInf / pow(rhoInf, GAMMA_GPU);
        VtB_x = uInf - VnInf * nx;
        VtB_y = vInf - VnInf * ny;
    } else {
        sB    = pI / pow(rhoI, GAMMA_GPU);
        VtB_x = uI - VnI * nx;
        VtB_y = vI - VnI * ny;
    }

    double VnB = 0.5 * (Rplus + Rminus);
    double cB  = 0.25 * gm1 * (Rplus - Rminus);
    if (cB < 1e-14) cB = 1e-14;

    double rhoB = pow(cB * cB / (GAMMA_GPU * sB), 1.0 / gm1);
    double uB   = VtB_x + VnB * nx;
    double vB   = VtB_y + VnB * ny;
    double pB   = rhoB * cB * cB / GAMMA_GPU;

    UR[0] = rhoB;
    UR[1] = rhoB * uB;
    UR[2] = rhoB * vB;
    UR[3] = pB / gm1 + 0.5 * rhoB * (uB * uB + vB * vB);
}

__device__ inline double evalPhiFace(int lf, int i, int j, int q,
                                     int P1, int nq1d)
{
    int nqF = nq1d;
    double phiXi, phiEta;
    switch (lf) {
        case 0: phiXi = c_Bmat[i * nq1d + q];               phiEta = c_blr[j * 2 + 0]; break;
        case 1: phiXi = c_blr[i * 2 + 1];                   phiEta = c_Bmat[j * nq1d + q]; break;
        case 2: phiXi = c_Bmat[i * nq1d + (nqF - 1 - q)];   phiEta = c_blr[j * 2 + 1]; break;
        case 3: phiXi = c_blr[i * 2 + 0];                   phiEta = c_Bmat[j * nq1d + (nqF - 1 - q)]; break;
        default: phiXi = 0; phiEta = 0;
    }
    return phiXi * phiEta;
}

// ============================================================================
// Kernel 0: Shock sensor (Persson-Peraire modal indicator)
// ============================================================================

__global__ void shockSensorKernel(
    const double* __restrict__ d_Ucoeff,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    double* __restrict__ d_epsilon,
    double* __restrict__ d_sensor,
    int nE, int P1, int nmodes, int nqVol,
    double s0, double kappa, double smaxRef, double avScale)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nE) return;

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

    double c_modal[MAX_P1 * MAX_P1];
    for (int k = 0; k < P1; ++k)
        for (int l = 0; l < P1; ++l) {
            double val = 0.0;
            for (int i = 0; i < P1; ++i) {
                double Tki = c_NodalToModal[k * P1 + i];
                for (int j = 0; j < P1; ++j)
                    val += Tki * c_NodalToModal[l * P1 + j]
                         * d_Ucoeff[0 * nE * nmodes + e * nmodes + i * P1 + j];
            }
            c_modal[k * P1 + l] = val;
        }

    int highDegThresh = max(2 * P - 1, P + 1);
    double total_energy = 0.0;
    double high_energy  = 0.0;
    for (int i = 0; i < P1; ++i)
        for (int j = 0; j < P1; ++j) {
            double c = c_modal[i * P1 + j];
            total_energy += c * c;
            if (i + j >= highDegThresh)
                high_energy += c * c;
        }

    double se = (total_energy > 1e-30) ? log10(high_energy / total_energy) : -20.0;

    double rhoMean = c_modal[0];
    if (high_energy < 1e-6 * rhoMean * rhoMean)
        se = -20.0;

    double eps;
    if (se < s0 - kappa)
        eps = 0.0;
    else if (se > s0 + kappa)
        eps = epsilon0;
    else
        eps = 0.5 * epsilon0 * (1.0 + sin(M_PI * (se - s0) / (2.0 * kappa)));

    d_epsilon[e] = eps;
    d_sensor[e]  = se;
}

// ============================================================================
// Kernel 0.5: AV volume integral  -eps * integral(grad U . grad phi)
// ============================================================================

__global__ void avVolumeKernel(
    const double* __restrict__ d_Ucoeff,
    double* __restrict__ d_rhsCoeff,
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
    double* sUcoeff = smem;
    double* gradXi  = sUcoeff + NVAR_GPU * nmodes;
    double* gradEta = gradXi  + NVAR_GPU * nqVol;

    for (int i = tid; i < NVAR_GPU * nmodes; i += blockDim.x) {
        int v = i / nmodes;
        int m = i % nmodes;
        sUcoeff[i] = d_Ucoeff[v * nE * nmodes + e * nmodes + m];
    }
    __syncthreads();

    for (int w = tid; w < NVAR_GPU * nqVol; w += blockDim.x) {
        int v  = w / nqVol;
        int q  = w % nqVol;
        int qx = q / nq1d;
        int qe = q % nq1d;

        double dxi = 0.0, deta = 0.0;
        for (int i = 0; i < P1; ++i) {
            double Di_qx = c_Dmat[i * nq1d + qx];
            double Bi_qx = c_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j) {
                double c = sUcoeff[v * nmodes + i * P1 + j];
                dxi  += c * Di_qx * c_Bmat[j * nq1d + qe];
                deta += c * Bi_qx * c_Dmat[j * nq1d + qe];
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
            double Bi = c_Bmat[mi * nq1d + qx];
            double Di = c_Dmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                int gIdx   = e * nqVol + qLocal;
                double wq  = c_wq[qx] * c_wq[qe] * d_detJ[gIdx];

                double Bj = c_Bmat[mj * nq1d + qe];
                double Dj = c_Dmat[mj * nq1d + qe];

                double dphidxi  = Di * Bj;
                double dphideta = Bi * Dj;
                double dphidx = d_dxidx[gIdx]  * dphidxi + d_detadx[gIdx] * dphideta;
                double dphidy = d_dxidy[gIdx]  * dphidxi + d_detady[gIdx] * dphideta;

                double grXi  = gradXi [v * nqVol + qLocal];
                double grEta = gradEta[v * nqVol + qLocal];
                double dUdx = d_dxidx[gIdx]  * grXi + d_detadx[gIdx] * grEta;
                double dUdy = d_dxidy[gIdx]  * grXi + d_detady[gIdx] * grEta;

                result -= eps * wq * (dUdx * dphidx + dUdy * dphidy);
            }
        }

        d_rhsCoeff[v * nE * nmodes + e * nmodes + m] += result;
    }
}

// ============================================================================
// Kernel 1: Forward transform (per-element blocks)
//   Phase 1: projection = integral of U * phi * w * detJ
//   Phase 2: Ucoeff = Minv * projection
// ============================================================================

__global__ void forwardTransformKernel(
    const double* __restrict__ d_U,
    double* __restrict__ d_Ucoeff,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_Minv,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork = NVAR_GPU * nmodes;

    extern __shared__ double smem[];
    double* proj = smem;

    for (int w = tid; w < nwork; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;
        int mi = m / P1;
        int mj = m % P1;

        double val = 0.0;
        for (int qx = 0; qx < nq1d; ++qx) {
            double Bi = c_Bmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qIdx = e * nqVol + qx * nq1d + qe;
                double weight = c_wq[qx] * c_wq[qe] * d_detJ[qIdx];
                val += weight * Bi * c_Bmat[mj * nq1d + qe]
                     * d_U[v * totalDOF + qIdx];
            }
        }
        proj[w] = val;
    }

    __syncthreads();

    for (int w = tid; w < nwork; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;

        double val = 0.0;
        for (int mp = 0; mp < nmodes; ++mp)
            val += d_Minv[e * nmodes * nmodes + m * nmodes + mp]
                 * proj[v * nmodes + mp];

        d_Ucoeff[v * nE * nmodes + e * nmodes + m] = val;
    }
}

// ============================================================================
// Kernel 2: Volume + surface integral (1 block per element, shared memory)
//   Phase 1: Load element data (U, metrics, Ucoeff) into shared memory
//   Phase 2: Volume integral   -- threads [0, NVAR*nmodes) compute (v,m) pairs
//   Phase 3: Surface flux      -- threads [NVAR*nmodes, NVAR*nmodes+4*nqFace)
//            precompute Lax-Friedrichs fluxes at face quad points
//   Phase 4: Surface integral  -- threads [0, NVAR*nmodes) accumulate from
//            precomputed fluxes
//   Phases 2 and 3 run concurrently on separate warps.
// ============================================================================

__global__ void volumeSurfaceKernel(
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
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf,
    int fluxType,
    const double* __restrict__ d_epsilon)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork    = NVAR_GPU * nmodes;
    int nFaceQP  = 4 * nqFace;

    extern __shared__ double smem[];
    double* sU      = smem;                             // [4 * nqVol]
    double* sDetJ   = sU      + 4 * nqVol;             // [nqVol]
    double* sDxidx  = sDetJ   + nqVol;                  // [nqVol]
    double* sDxidy  = sDxidx  + nqVol;                  // [nqVol]
    double* sDetadx = sDxidy  + nqVol;                  // [nqVol]
    double* sDetady = sDetadx + nqVol;                  // [nqVol]
    double* sUcoeff = sDetady + nqVol;                  // [4 * nmodes]
    double* sFnumW  = sUcoeff + 4 * nmodes;             // [nFaceQP * 4]

    // ===== Phase 1: Cooperatively load element data =====
    int baseQ = e * nqVol;
    for (int i = tid; i < nqVol; i += blockDim.x) {
        int gIdx = baseQ + i;
        sU[0 * nqVol + i] = d_U[0 * totalDOF + gIdx];
        sU[1 * nqVol + i] = d_U[1 * totalDOF + gIdx];
        sU[2 * nqVol + i] = d_U[2 * totalDOF + gIdx];
        sU[3 * nqVol + i] = d_U[3 * totalDOF + gIdx];
        sDetJ[i]   = d_detJ[gIdx];
        sDxidx[i]  = d_dxidx[gIdx];
        sDxidy[i]  = d_dxidy[gIdx];
        sDetadx[i] = d_detadx[gIdx];
        sDetady[i] = d_detady[gIdx];
    }
    for (int i = tid; i < 4 * nmodes; i += blockDim.x) {
        int vv = i / nmodes;
        int mm = i % nmodes;
        sUcoeff[i] = d_Ucoeff[vv * nE * nmodes + e * nmodes + mm];
    }
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
            double Bi = c_Bmat[mi_id * nq1d + qx];
            double Di = c_Dmat[mi_id * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qLocal = qx * nq1d + qe;
                double w = c_wq[qx] * c_wq[qe] * sDetJ[qLocal];

                double rho  = sU[0 * nqVol + qLocal];
                double rhou = sU[1 * nqVol + qLocal];
                double rhov = sU[2 * nqVol + qLocal];
                double rhoE = sU[3 * nqVol + qLocal];

                double uv = rhou / rho;
                double vv = rhov / rho;
                double p  = (GAMMA_GPU - 1.0) * (rhoE - 0.5 * rho * (uv*uv + vv*vv));

                double fx, fy;
                switch (v_id) {
                    case 0: fx = rhou;           fy = rhov;           break;
                    case 1: fx = rhou*uv + p;    fy = rhov*uv;        break;
                    case 2: fx = rhou*vv;        fy = rhov*vv + p;    break;
                    default: fx = (rhoE+p)*uv;   fy = (rhoE+p)*vv;    break;
                }

                double Bj = c_Bmat[mj_id * nq1d + qe];
                double Dj = c_Dmat[mj_id * nq1d + qe];
                double dphidxi  = Di * Bj;
                double dphideta = Bi * Dj;
                double dphidx = sDxidx[qLocal]  * dphidxi + sDetadx[qLocal] * dphideta;
                double dphidy = sDxidy[qLocal]  * dphidxi + sDetady[qLocal] * dphideta;

                volResult += w * (fx * dphidx + fy * dphidy);
            }
        }
    }

    // ===== Phase 3: Precompute surface numerical fluxes =====
    // Threads [nwork .. nwork+nFaceQP) each handle one (face, quad_point).
    // Runs concurrently with Phase 2 on separate warps.
    else if (tid < nwork + nFaceQP) {
        int fq_id = tid - nwork;
        int lf = fq_id / nqFace;
        int q  = fq_id % nqFace;

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
        double wf = c_wq[q] * d_faceJac[fIdx];

        double UMe[4] = {0, 0, 0, 0};
        for (int vv = 0; vv < 4; ++vv)
            for (int ii = 0; ii < P1; ++ii)
                for (int jj = 0; jj < P1; ++jj)
                    UMe[vv] += sUcoeff[vv * nmodes + ii * P1 + jj]
                             * evalPhiFace(lf, ii, jj, q, P1, nq1d);

        double UNbr[4];
        if (!is_boundary) {
            int qN = nqFace - 1 - q;
            for (int vv = 0; vv < 4; ++vv) {
                UNbr[vv] = 0.0;
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        UNbr[vv] += d_Ucoeff[vv * nE * nmodes + eN * nmodes + ii*P1+jj]
                                  * evalPhiFace(lfN, ii, jj, qN, P1, nq1d);
            }
        } else {
            int bcType = d_face_bcType[f];
            if (bcType == 1)
                slipWallBC_d(UMe, nx, ny, UNbr);
            else if (bcType == 2)
                riemannBC_d(UMe, nx, ny, rhoInf, uInf, vInf, pInf, UNbr);
            else
                for (int vv = 0; vv < 4; ++vv) UNbr[vv] = UMe[vv];
        }

        double Fnum[4];
        if (fluxType == 1) {
            if (is_left) hllc_d(UMe, UNbr, nx, ny, Fnum);
            else         hllc_d(UNbr, UMe, nx, ny, Fnum);
        } else {
            if (is_left) laxFriedrichs_d(UMe, UNbr, nx, ny, Fnum);
            else         laxFriedrichs_d(UNbr, UMe, nx, ny, Fnum);
        }

        if (d_epsilon != nullptr) {
            double eps_me = d_epsilon[e];
            double eps_nb = (is_boundary) ? eps_me
                          : d_epsilon[eN];
            double eps_f  = fmax(eps_me, eps_nb);
            if (eps_f > 0.0) {
                double sigma = eps_f * (double)(P1 * P1)
                             / fmax(2.0 * d_faceJac[fIdx], 1e-30);
                if (is_left) {
                    for (int vv = 0; vv < 4; ++vv)
                        Fnum[vv] += sigma * (UMe[vv] - UNbr[vv]);
                } else {
                    for (int vv = 0; vv < 4; ++vv)
                        Fnum[vv] += sigma * (UNbr[vv] - UMe[vv]);
                }
            }
        }

        double sign = is_left ? -1.0 : 1.0;
        for (int vv = 0; vv < 4; ++vv)
            sFnumW[fq_id * 4 + vv] = sign * wf * Fnum[vv];
    }

    __syncthreads();

    // ===== Phase 4: Surface integral + write result =====
    if (tid < nwork) {
        double surfResult = 0.0;
        for (int fq = 0; fq < nFaceQP; ++fq) {
            int lf = fq / nqFace;
            int q  = fq % nqFace;
            double phi = evalPhiFace(lf, mi_id, mj_id, q, P1, nq1d);
            surfResult += sFnumW[fq * 4 + v_id] * phi;
        }
        d_rhsCoeff[v_id * nE * nmodes + e * nmodes + m_id] = volResult + surfResult;
    }
}

// ============================================================================
// Kernel 3: Mass solve + backward transform (per-element blocks)
//   Phase 1: dUdt_coeff = Minv * rhsCoeff
//   Phase 2: R = sum_m dUdt_coeff[m] * phi_m(q)
// ============================================================================

__global__ void massSolveBackwardKernel(
    const double* __restrict__ d_rhsCoeff,
    double* __restrict__ d_R,
    const double* __restrict__ d_Minv,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int tid = threadIdx.x;
    int nwork_modes = NVAR_GPU * nmodes;

    extern __shared__ double smem[];
    double* dUdt = smem;

    for (int w = tid; w < nwork_modes; w += blockDim.x) {
        int v = w / nmodes;
        int m = w % nmodes;

        double val = 0.0;
        for (int mp = 0; mp < nmodes; ++mp)
            val += d_Minv[e * nmodes * nmodes + m * nmodes + mp]
                 * d_rhsCoeff[v * nE * nmodes + e * nmodes + mp];

        dUdt[w] = val;
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
            double Bi = c_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j)
                val += dUdt[v * nmodes + i * P1 + j] * Bi * c_Bmat[j * nq1d + qe];
        }

        d_R[v * totalDOF + e * nqVol + q] = val;
    }
}

// ============================================================================
// Kernel 4: RK4 stage update
// ============================================================================

__global__ void rk4StageKernel(
    double* __restrict__ d_U,
    double* __restrict__ d_Utmp,
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
        case 1: d_Utmp[i] = d_U[i] + 0.5 * ki; break;
        case 2: d_Utmp[i] = d_U[i] + 0.5 * ki; break;
        case 3: d_Utmp[i] = d_U[i] + ki;        break;
        case 4:
            d_U[i] += (1.0/6.0) * (d_k1[i] + 2.0*d_k2[i] + 2.0*d_k3[i] + ki);
            break;
    }
}

// ============================================================================
// Kernel 4b: Positivity enforcement (density and pressure floor)
// ============================================================================

__global__ void positivityKernel(
    double* __restrict__ d_sol,
    int totalDOF, double rhoMin, double pMin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalDOF) return;

    double rho  = d_sol[0 * totalDOF + idx];
    double rhou = d_sol[1 * totalDOF + idx];
    double rhov = d_sol[2 * totalDOF + idx];
    double rhoE = d_sol[3 * totalDOF + idx];

    bool modified = false;

    if (rho < rhoMin) {
        rho = rhoMin;
        d_sol[0 * totalDOF + idx] = rho;
        modified = true;
    }

    double u  = rhou / rho;
    double v  = rhov / rho;
    double ke = 0.5 * rho * (u * u + v * v);
    double p  = (GAMMA_GPU - 1.0) * (rhoE - ke);

    if (p < pMin) {
        rhoE = pMin / (GAMMA_GPU - 1.0) + ke;
        d_sol[3 * totalDOF + idx] = rhoE;
    }
}

// ============================================================================
// Kernel 5: CFL reduction
// ============================================================================

__device__ double atomicMinDouble(double* addr, double val)
{
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void cflKernel(
    const double* __restrict__ d_U,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    double* __restrict__ d_dtMin,
    int nE, int nqVol, int totalDOF, double CFL, int P)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double myMin = 1e20;

    for (int i = idx; i < nE * nqVol; i += gridDim.x * blockDim.x) {
        double gradXi  = sqrt(d_dxidx[i]*d_dxidx[i] + d_dxidy[i]*d_dxidy[i]);
        double gradEta = sqrt(d_detadx[i]*d_detadx[i] + d_detady[i]*d_detady[i]);
        double h = 1.0 / fmax(gradXi, gradEta);

        double rho  = d_U[0 * totalDOF + i];
        double u    = d_U[1 * totalDOF + i] / rho;
        double v    = d_U[2 * totalDOF + i] / rho;
        double p    = pressure_d(d_U[0*totalDOF+i], d_U[1*totalDOF+i],
                                 d_U[2*totalDOF+i], d_U[3*totalDOF+i]);
        double c    = soundSpeed_d(rho, p);
        double smax = sqrt(u*u + v*v) + c;
        double dt_local = CFL * h / (smax * (2.0 * P + 1.0));
        myMin = fmin(myMin, dt_local);
    }

    sdata[tid] = myMin;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMinDouble(d_dtMin, sdata[0]);
}

// ============================================================================
// Kernel 5b: Per-element CFL (for local time stepping)
// ============================================================================

__global__ void elementCflKernel(
    const double* __restrict__ d_U,
    const double* __restrict__ d_dxidx,
    const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx,
    const double* __restrict__ d_detady,
    double* __restrict__ d_dtLocal,
    double* __restrict__ d_dtMin,
    int nE, int nqVol, int totalDOF, double CFL, int P)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nE) return;

    double myMin = 1e20;
    for (int q = 0; q < nqVol; ++q) {
        int i = e * nqVol + q;
        double gradXi  = sqrt(d_dxidx[i]*d_dxidx[i] + d_dxidy[i]*d_dxidy[i]);
        double gradEta = sqrt(d_detadx[i]*d_detadx[i] + d_detady[i]*d_detady[i]);
        double h = 1.0 / fmax(gradXi, gradEta);

        double rho  = d_U[0 * totalDOF + i];
        double u    = d_U[1 * totalDOF + i] / rho;
        double v    = d_U[2 * totalDOF + i] / rho;
        double p    = pressure_d(d_U[0*totalDOF+i], d_U[1*totalDOF+i],
                                 d_U[2*totalDOF+i], d_U[3*totalDOF+i]);
        double c    = soundSpeed_d(rho, p);
        double smax = sqrt(u*u + v*v) + c;
        double dt_local = CFL * h / (smax * (2.0 * P + 1.0));
        myMin = fmin(myMin, dt_local);
    }

    d_dtLocal[e] = myMin;
    atomicMinDouble(d_dtMin, myMin);
}

// ============================================================================
// Kernel 6: NaN check
// ============================================================================

__global__ void nanCheckKernel(const double* __restrict__ data, int N,
                               int* __restrict__ flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && isnan(data[i]))
        *flag = 1;
}

// ============================================================================
// Kernel 7: Fused per-variable residual norm (all 4 variables in one pass)
// ============================================================================

__global__ void residualNormPerVarKernel(
    const double* __restrict__ d_R,
    double* __restrict__ d_norms,
    int totalDOF)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = blockIdx.x * bs + tid; i < totalDOF; i += gridDim.x * bs) {
        double v0 = d_R[i];
        double v1 = d_R[totalDOF + i];
        double v2 = d_R[2 * totalDOF + i];
        double v3 = d_R[3 * totalDOF + i];
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
        atomicAdd(&d_norms[0], sdata[0]);
        atomicAdd(&d_norms[1], sdata[bs]);
        atomicAdd(&d_norms[2], sdata[2 * bs]);
        atomicAdd(&d_norms[3], sdata[3 * bs]);
    }
}

// ============================================================================
// Host wrapper: allocate GPU memory
// ============================================================================

void gpuAllocate(GPUSolverData& gpu, int nE, int nF, int P, int nq1d)
{
    gpu.nE = nE;
    gpu.nF = nF;
    gpu.P  = P;
    gpu.P1 = P + 1;
    gpu.nq1d = nq1d;
    gpu.nqVol = nq1d * nq1d;
    gpu.nqFace = nq1d;
    gpu.nmodes = (P+1) * (P+1);
    gpu.totalDOF = nE * gpu.nqVol;

    int nqVol = gpu.nqVol;
    int nmodes = gpu.nmodes;
    int totalDOF = gpu.totalDOF;

    CUDA_CHECK(cudaMalloc(&gpu.d_detJ,   nE * nqVol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_dxidx,  nE * nqVol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_dxidy,  nE * nqVol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_detadx, nE * nqVol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_detady, nE * nqVol * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&gpu.d_faceNx,    nF * nq1d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_faceNy,    nF * nq1d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_faceJac,   nF * nq1d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_faceXPhys, nF * nq1d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_faceYPhys, nF * nq1d * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&gpu.d_Bmat, (P+1) * nq1d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_Dmat, (P+1) * nq1d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_blr,  (P+1) * 2 * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&gpu.d_Minv, nE * nmodes * nmodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_wq,   nq1d * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&gpu.d_elem2face,   nE * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_face_elemL,  nF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_face_elemR,  nF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_face_faceL,  nF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_face_faceR,  nF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_face_bcType, nF * sizeof(int)));

    int solSize = NVAR_GPU * totalDOF;
    CUDA_CHECK(cudaMalloc(&gpu.d_U,    solSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_R,    solSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_k1,   solSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_k2,   solSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_k3,   solSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_k4,   solSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_Utmp, solSize * sizeof(double)));

    int coeffSize = NVAR_GPU * nE * nmodes;
    CUDA_CHECK(cudaMalloc(&gpu.d_Ucoeff,   coeffSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_rhsCoeff, coeffSize * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&gpu.d_dtMin,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_dtLocal, nE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_nanFlag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_normBuf, 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_epsilon, nE * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_epsilon, 0, nE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_sensor, nE * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_sensor, 0, nE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_Uprev, solSize * sizeof(double)));
    int qcoeffSize = NVAR_GPU * 2 * nE * nmodes;
    CUDA_CHECK(cudaMalloc(&gpu.d_Qcoeff, qcoeffSize * sizeof(double)));
}

void gpuFree(GPUSolverData& gpu)
{
    cudaFree(gpu.d_detJ);   cudaFree(gpu.d_dxidx);
    cudaFree(gpu.d_dxidy);  cudaFree(gpu.d_detadx);
    cudaFree(gpu.d_detady);
    cudaFree(gpu.d_faceNx); cudaFree(gpu.d_faceNy);
    cudaFree(gpu.d_faceJac); cudaFree(gpu.d_faceXPhys);
    cudaFree(gpu.d_faceYPhys);
    cudaFree(gpu.d_Bmat);   cudaFree(gpu.d_Dmat);
    cudaFree(gpu.d_blr);    cudaFree(gpu.d_Minv);
    cudaFree(gpu.d_wq);
    cudaFree(gpu.d_elem2face);
    cudaFree(gpu.d_face_elemL); cudaFree(gpu.d_face_elemR);
    cudaFree(gpu.d_face_faceL); cudaFree(gpu.d_face_faceR);
    cudaFree(gpu.d_face_bcType);
    cudaFree(gpu.d_U);     cudaFree(gpu.d_R);
    cudaFree(gpu.d_k1);    cudaFree(gpu.d_k2);
    cudaFree(gpu.d_k3);    cudaFree(gpu.d_k4);
    cudaFree(gpu.d_Utmp);
    cudaFree(gpu.d_Ucoeff);  cudaFree(gpu.d_rhsCoeff);
    cudaFree(gpu.d_dtMin);   cudaFree(gpu.d_dtLocal);
    cudaFree(gpu.d_nanFlag); cudaFree(gpu.d_normBuf);
    cudaFree(gpu.d_epsilon); cudaFree(gpu.d_sensor);
    cudaFree(gpu.d_Uprev);   cudaFree(gpu.d_Qcoeff);
}

// ============================================================================
// Host wrapper: copy static data to GPU
// ============================================================================

void gpuCopyStaticData(
    GPUSolverData& gpu,
    const double* detJ, const double* dxidx, const double* dxidy,
    const double* detadx, const double* detady,
    const double* faceNx, const double* faceNy, const double* faceJac,
    const double* faceXPhys, const double* faceYPhys,
    const double* Bmat_flat, const double* Dmat_flat, const double* blr_flat,
    const double* Minv, const double* wq,
    const int* elem2face_flat,
    const int* face_elemL, const int* face_elemR,
    const int* face_faceL, const int* face_faceR,
    const int* face_bcType)
{
    int nE = gpu.nE, nF = gpu.nF, nq1d = gpu.nq1d;
    int nqVol = gpu.nqVol, nmodes = gpu.nmodes, P1 = gpu.P1;

    CUDA_CHECK(cudaMemcpy(gpu.d_detJ,   detJ,   nE*nqVol*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_dxidx,  dxidx,  nE*nqVol*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_dxidy,  dxidy,  nE*nqVol*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_detadx, detadx, nE*nqVol*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_detady, detady, nE*nqVol*sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu.d_faceNx,    faceNx,    nF*nq1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_faceNy,    faceNy,    nF*nq1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_faceJac,   faceJac,   nF*nq1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_faceXPhys, faceXPhys, nF*nq1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_faceYPhys, faceYPhys, nF*nq1d*sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu.d_Bmat, Bmat_flat, P1*nq1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_Dmat, Dmat_flat, P1*nq1d*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_blr,  blr_flat,  P1*2*sizeof(double),    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu.d_Minv, Minv, nE*nmodes*nmodes*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_wq,   wq,   nq1d*sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(c_Bmat, Bmat_flat, P1*nq1d*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Dmat, Dmat_flat, P1*nq1d*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_blr,  blr_flat,  P1*2*sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wq,   wq,        nq1d*sizeof(double)));

    CUDA_CHECK(cudaMemcpy(gpu.d_elem2face,   elem2face_flat, nE*4*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_elemL,  face_elemL,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_elemR,  face_elemR,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_faceL,  face_faceL,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_faceR,  face_faceR,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_bcType, face_bcType, nF*sizeof(int), cudaMemcpyHostToDevice));
}

// ============================================================================
// Host wrapper: upload nodal-to-modal transform
// ============================================================================

void gpuSetNodalToModal(const double* T, int P1)
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_NodalToModal, T, P1 * P1 * sizeof(double)));
}

// ============================================================================
// Host wrapper: copy solution to/from GPU
// ============================================================================

void gpuCopySolutionToDevice(GPUSolverData& gpu, const double* U_flat)
{
    CUDA_CHECK(cudaMemcpy(gpu.d_U, U_flat,
               NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyHostToDevice));
}

void gpuCopySolutionToHost(GPUSolverData& gpu, double* U_flat)
{
    CUDA_CHECK(cudaMemcpy(U_flat, gpu.d_U,
               NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Host wrapper: compute DG RHS on GPU
// ============================================================================

void gpuComputeDGRHS(GPUSolverData& gpu, bool useUtmp, double time)
{
    const double* d_Uin = useUtmp ? gpu.d_Utmp : gpu.d_U;
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes, nqVol = gpu.nqVol, totalDOF = gpu.totalDOF;
    int nqFace = gpu.nqFace;

    // --- Kernel 1: Forward transform ---
    int blockDim1 = max(64, NVAR_GPU * nmodes);
    if (blockDim1 % 32 != 0) blockDim1 = ((blockDim1 + 31) / 32) * 32;
    int smem1 = NVAR_GPU * nmodes * sizeof(double);
    forwardTransformKernel<<<nE, blockDim1, smem1>>>(
        d_Uin, gpu.d_Ucoeff, gpu.d_detJ, gpu.d_Minv,
        nE, P1, nq1d, nmodes, nqVol, totalDOF);

    // --- Kernel 1.5: Shock sensor ---
    if (gpu.useAV) {
        int blockS = 256;
        int gridS  = (nE + blockS - 1) / blockS;
        double uMag = sqrt(gpu.uInf*gpu.uInf + gpu.vInf*gpu.vInf);
        double cInf = sqrt(GAMMA_GPU * gpu.pInf / gpu.rhoInf);
        double smaxRef = uMag + cInf;
        shockSensorKernel<<<gridS, blockS>>>(
            gpu.d_Ucoeff,
            gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            gpu.d_epsilon, gpu.d_sensor,
            nE, P1, nmodes, nqVol,
            gpu.AVs0, gpu.AVkappa, smaxRef, gpu.AVscale);
    }

    // --- Kernel 2: Volume + surface integral (1 block per element) ---
    int nwork    = NVAR_GPU * nmodes;
    int nFaceQP  = 4 * nqFace;
    int blockDim2 = ((nwork + nFaceQP + 31) / 32) * 32;
    int smem2 = (9 * nqVol + 4 * nmodes + nFaceQP * 4) * sizeof(double);
    const double* d_eps_ptr = gpu.useAV ? gpu.d_epsilon : nullptr;
    volumeSurfaceKernel<<<nE, blockDim2, smem2>>>(
        d_Uin, gpu.d_Ucoeff, gpu.d_rhsCoeff,
        gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
        nE, P1, nq1d, nmodes, nqVol, totalDOF, nqFace,
        gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf,
        gpu.fluxType,
        d_eps_ptr);

    // --- Kernel 2.5: AV volume diffusion ---
    if (gpu.useAV) {
        int blockAV = max(64, NVAR_GPU * nmodes);
        if (blockAV % 32 != 0) blockAV = ((blockAV + 31) / 32) * 32;
        blockAV = max(blockAV, NVAR_GPU * nqVol);
        if (blockAV % 32 != 0) blockAV = ((blockAV + 31) / 32) * 32;
        int smemAV = (NVAR_GPU * nmodes + 2 * NVAR_GPU * nqVol) * sizeof(double);
        avVolumeKernel<<<nE, blockAV, smemAV>>>(
            gpu.d_Ucoeff, gpu.d_rhsCoeff,
            gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
            gpu.d_epsilon,
            nE, P1, nq1d, nmodes, nqVol);
    }

    // --- Kernel 3: Mass solve + backward transform ---
    int blockDim3 = max(64, NVAR_GPU * nmodes);
    if (blockDim3 % 32 != 0) blockDim3 = ((blockDim3 + 31) / 32) * 32;
    int smem3 = NVAR_GPU * nmodes * sizeof(double);
    massSolveBackwardKernel<<<nE, blockDim3, smem3>>>(
        gpu.d_rhsCoeff, gpu.d_R, gpu.d_Minv,
        nE, P1, nq1d, nmodes, nqVol, totalDOF);
}

// ============================================================================
// Host wrapper: RK4 stage update
// ============================================================================

void gpuRK4Stage(GPUSolverData& gpu, double dt, int stage)
{
    int N = NVAR_GPU * gpu.totalDOF;
    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;

    double* d_k;
    switch (stage) {
        case 1: d_k = gpu.d_k1; break;
        case 2: d_k = gpu.d_k2; break;
        case 3: d_k = gpu.d_k3; break;
        default: d_k = gpu.d_k4; break;
    }

    rk4StageKernel<<<gridDim, blockDim>>>(
        gpu.d_U, gpu.d_Utmp, d_k, gpu.d_R,
        gpu.d_k1, gpu.d_k2, gpu.d_k3,
        dt, stage, N);

    // Enforce positivity on the updated solution
    int bk = 256;
    int gd = (gpu.totalDOF + bk - 1) / bk;
    double rhoMin = 1e-6, pMin = 1e-6;
    double* d_target = (stage < 4) ? gpu.d_Utmp : gpu.d_U;
    positivityKernel<<<gd, bk>>>(d_target, gpu.totalDOF, rhoMin, pMin);
}

// ============================================================================
// Host wrapper: CFL time step
// ============================================================================

double gpuComputeCFL(GPUSolverData& gpu, double CFL, int P)
{
    double huge = 1e20;
    CUDA_CHECK(cudaMemcpy(gpu.d_dtMin, &huge, sizeof(double), cudaMemcpyHostToDevice));

    int totalPts = gpu.nE * gpu.nqVol;
    int blockDim = 256;
    int gridDim = min((totalPts + blockDim - 1) / blockDim, 1024);
    int smem = blockDim * sizeof(double);
    cflKernel<<<gridDim, blockDim, smem>>>(
        gpu.d_U, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        gpu.d_dtMin, gpu.nE, gpu.nqVol, gpu.totalDOF, CFL, P);

    double dtMin;
    CUDA_CHECK(cudaMemcpy(&dtMin, gpu.d_dtMin, sizeof(double), cudaMemcpyDeviceToHost));
    return dtMin;
}

void gpuComputeElementCFL(GPUSolverData& gpu, double CFL, int P)
{
    double huge = 1e20;
    CUDA_CHECK(cudaMemcpy(gpu.d_dtMin, &huge, sizeof(double), cudaMemcpyHostToDevice));
    int blockDim = 256;
    int gridDim = (gpu.nE + blockDim - 1) / blockDim;
    elementCflKernel<<<gridDim, blockDim>>>(
        gpu.d_U, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        gpu.d_dtLocal, gpu.d_dtMin,
        gpu.nE, gpu.nqVol, gpu.totalDOF, CFL, P);
}

// ============================================================================
// Host wrapper: NaN check
// ============================================================================

bool gpuCheckNaN(GPUSolverData& gpu)
{
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(gpu.d_nanFlag, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int N = NVAR_GPU * gpu.totalDOF;
    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;
    nanCheckKernel<<<gridDim, blockDim>>>(gpu.d_U, N, gpu.d_nanFlag);

    int flag;
    CUDA_CHECK(cudaMemcpy(&flag, gpu.d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost));
    return (flag != 0);
}

void gpuCopyEpsilonToHost(GPUSolverData& gpu, double* eps_host)
{
    CUDA_CHECK(cudaMemcpy(eps_host, gpu.d_epsilon,
               gpu.nE * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpuCopySensorToHost(GPUSolverData& gpu, double* sensor_host)
{
    CUDA_CHECK(cudaMemcpy(sensor_host, gpu.d_sensor,
               gpu.nE * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpuSnapshotSolution(GPUSolverData& gpu)
{
    CUDA_CHECK(cudaMemcpy(gpu.d_Uprev, gpu.d_U,
               NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyDeviceToDevice));
}

void gpuCopyPrevSolutionToHost(GPUSolverData& gpu, double* U_flat)
{
    CUDA_CHECK(cudaMemcpy(U_flat, gpu.d_Uprev,
               NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpuRestoreSnapshot(GPUSolverData& gpu)
{
    CUDA_CHECK(cudaMemcpy(gpu.d_U, gpu.d_Uprev,
               NVAR_GPU * gpu.totalDOF * sizeof(double), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// Host wrapper: fused per-variable residual norm (1 H2D + 1 kernel + 1 D2H)
// ============================================================================

void gpuResidualNormPerVarFused(GPUSolverData& gpu, double norms[4])
{
    CUDA_CHECK(cudaMemset(gpu.d_normBuf, 0, 4 * sizeof(double)));

    int bk = 256;
    int gd = min((gpu.totalDOF + bk - 1) / bk, 1024);
    residualNormPerVarKernel<<<gd, bk, 4 * bk * sizeof(double)>>>(
        gpu.d_R, gpu.d_normBuf, gpu.totalDOF);

    CUDA_CHECK(cudaMemcpy(norms, gpu.d_normBuf, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int v = 0; v < 4; ++v)
        norms[v] = sqrt(norms[v]);
}

// ============================================================================
// Implicit solver: GPU reduction and BLAS-like kernels
// ============================================================================

__global__ void dotProductKernel(const double* __restrict__ x,
                                 const double* __restrict__ y,
                                 double* __restrict__ result, int N)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x)
        sum += x[i] * y[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

__global__ void axpyKernel(double a, const double* __restrict__ x,
                           double* __restrict__ y, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] += a * x[i];
}

__global__ void scaleVecKernel(double a, double* __restrict__ x, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= a;
}

__global__ void implicitMatvecKernel(const double* __restrict__ Rpert,
                                     const double* __restrict__ R0,
                                     const double* __restrict__ v,
                                     double* __restrict__ result,
                                     const double* __restrict__ d_dtLocal,
                                     double invEps, int totalDOF, int nqVol, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int e = (i % totalDOF) / nqVol;
        double invDt = 1.0 / d_dtLocal[e];
        double Jv = (Rpert[i] - R0[i]) * invEps;
        result[i] = v[i] * invDt - Jv;
    }
}

// ============================================================================
// Implicit solver: Block-Jacobi preconditioner assembly
// ============================================================================

__device__ void elementRhsCoeff_d(
    const double Ucoeff_loc[NVAR_GPU * MAX_P1 * MAX_P1],
    const double* __restrict__ d_Ucoeff, int e,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_dxidx, const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx, const double* __restrict__ d_detady,
    const double* __restrict__ d_faceNx, const double* __restrict__ d_faceNy,
    const double* __restrict__ d_faceJac,
    const int* __restrict__ d_elem2face,
    const int* __restrict__ d_face_elemL, const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_faceL, const int* __restrict__ d_face_faceR,
    const int* __restrict__ d_face_bcType,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf, int fluxType,
    const double* __restrict__ d_epsilon,
    double rhsOut[NVAR_GPU * MAX_P1 * MAX_P1])
{
    for (int k = 0; k < NVAR_GPU * nmodes; ++k)
        rhsOut[k] = 0.0;

    double Unodal[NVAR_GPU * MAX_NQ1D * MAX_NQ1D];
    for (int v = 0; v < NVAR_GPU; ++v)
        for (int qx = 0; qx < nq1d; ++qx)
            for (int qe = 0; qe < nq1d; ++qe) {
                double val = 0.0;
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        val += Ucoeff_loc[v * nmodes + ii * P1 + jj]
                             * c_Bmat[ii * nq1d + qx] * c_Bmat[jj * nq1d + qe];
                Unodal[v * nqVol + qx * nq1d + qe] = val;
            }

    int baseQ = e * nqVol;
    for (int mi = 0; mi < P1; ++mi)
        for (int mj = 0; mj < P1; ++mj) {
            int m = mi * P1 + mj;
            double volR[NVAR_GPU] = {0,0,0,0};
            for (int qx = 0; qx < nq1d; ++qx) {
                double Bi = c_Bmat[mi * nq1d + qx];
                double Di = c_Dmat[mi * nq1d + qx];
                for (int qe = 0; qe < nq1d; ++qe) {
                    int qLocal = qx * nq1d + qe;
                    int gIdx = baseQ + qLocal;
                    double w = c_wq[qx] * c_wq[qe] * d_detJ[gIdx];
                    double rho  = Unodal[0 * nqVol + qLocal];
                    double rhou = Unodal[1 * nqVol + qLocal];
                    double rhov = Unodal[2 * nqVol + qLocal];
                    double rhoE = Unodal[3 * nqVol + qLocal];
                    double uu = rhou/rho, vv = rhov/rho;
                    double p = (GAMMA_GPU-1.0)*(rhoE - 0.5*rho*(uu*uu+vv*vv));
                    double fx[4] = {rhou, rhou*uu+p, rhou*vv, (rhoE+p)*uu};
                    double fy[4] = {rhov, rhov*uu, rhov*vv+p, (rhoE+p)*vv};
                    double Bj = c_Bmat[mj * nq1d + qe];
                    double Dj = c_Dmat[mj * nq1d + qe];
                    double dphidxi  = Di * Bj;
                    double dphideta = Bi * Dj;
                    double dphidx = d_dxidx[gIdx]*dphidxi + d_detadx[gIdx]*dphideta;
                    double dphidy = d_dxidy[gIdx]*dphidxi + d_detady[gIdx]*dphideta;
                    for (int vv2 = 0; vv2 < 4; ++vv2)
                        volR[vv2] += w * (fx[vv2]*dphidx + fy[vv2]*dphidy);
                }
            }
            for (int v = 0; v < 4; ++v)
                rhsOut[v * nmodes + m] = volR[v];
        }

    for (int lf = 0; lf < 4; ++lf) {
        int f = d_elem2face[e * 4 + lf];
        int eL = d_face_elemL[f];
        int eR = d_face_elemR[f];
        bool is_left = (e == eL);
        int eN = is_left ? eR : eL;
        int lfN = is_left ? d_face_faceR[f] : d_face_faceL[f];
        bool is_boundary = (eN < 0);

        for (int q = 0; q < nqFace; ++q) {
            int q_face = is_left ? q : (nqFace - 1 - q);
            int fIdx = f * nqFace + q_face;
            double nx = d_faceNx[fIdx];
            double ny = d_faceNy[fIdx];
            double wf = c_wq[q] * d_faceJac[fIdx];

            double UMe[4] = {0,0,0,0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        UMe[vv] += Ucoeff_loc[vv*nmodes + ii*P1+jj]
                                 * evalPhiFace(lf, ii, jj, q, P1, nq1d);

            double UNbr[4];
            if (!is_boundary) {
                int qN = nqFace - 1 - q;
                for (int vv = 0; vv < 4; ++vv) {
                    UNbr[vv] = 0.0;
                    for (int ii = 0; ii < P1; ++ii)
                        for (int jj = 0; jj < P1; ++jj)
                            UNbr[vv] += d_Ucoeff[vv*nE*nmodes + eN*nmodes + ii*P1+jj]
                                      * evalPhiFace(lfN, ii, jj, qN, P1, nq1d);
                }
            } else {
                int bcType = d_face_bcType[f];
                if (bcType == 1)      slipWallBC_d(UMe, nx, ny, UNbr);
                else if (bcType == 2) riemannBC_d(UMe, nx, ny, rhoInf, uInf, vInf, pInf, UNbr);
                else                  for (int vv=0;vv<4;++vv) UNbr[vv] = UMe[vv];
            }

            double Fnum[4];
            if (fluxType == 1) {
                if (is_left) hllc_d(UMe, UNbr, nx, ny, Fnum);
                else         hllc_d(UNbr, UMe, nx, ny, Fnum);
            } else {
                if (is_left) laxFriedrichs_d(UMe, UNbr, nx, ny, Fnum);
                else         laxFriedrichs_d(UNbr, UMe, nx, ny, Fnum);
            }

            if (d_epsilon != nullptr) {
                double eps_me = d_epsilon[e];
                double eps_nb = is_boundary ? eps_me : d_epsilon[eN];
                double eps_f  = fmax(eps_me, eps_nb);
                if (eps_f > 0.0) {
                    double sigma = eps_f * (double)(P1*P1) / fmax(2.0*d_faceJac[fIdx], 1e-30);
                    if (is_left)
                        for (int vv=0;vv<4;++vv) Fnum[vv] += sigma*(UMe[vv]-UNbr[vv]);
                    else
                        for (int vv=0;vv<4;++vv) Fnum[vv] += sigma*(UNbr[vv]-UMe[vv]);
                }
            }

            double sign = is_left ? -1.0 : 1.0;
            for (int mi = 0; mi < P1; ++mi)
                for (int mj = 0; mj < P1; ++mj) {
                    double phi = evalPhiFace(lf, mi, mj, q, P1, nq1d);
                    for (int vv = 0; vv < 4; ++vv)
                        rhsOut[vv*nmodes + mi*P1+mj] += sign * wf * Fnum[vv] * phi;
                }
        }
    }
}

__global__ void blockJacobiAssemblyKernel(
    const double* __restrict__ d_Ucoeff,
    const double* __restrict__ d_Minv,
    const double* __restrict__ d_dtLocal,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_dxidx, const double* __restrict__ d_dxidy,
    const double* __restrict__ d_detadx, const double* __restrict__ d_detady,
    const double* __restrict__ d_faceNx, const double* __restrict__ d_faceNy,
    const double* __restrict__ d_faceJac,
    const int* __restrict__ d_elem2face,
    const int* __restrict__ d_face_elemL, const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_faceL, const int* __restrict__ d_face_faceR,
    const int* __restrict__ d_face_bcType,
    const double* __restrict__ d_epsilon,
    double* __restrict__ d_Jac,
    int* __restrict__ d_JacPiv,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf, int fluxType)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int blockSz = NVAR_GPU * nmodes;
    double dt_e = d_dtLocal[e];

    double Ucoeff_loc[NVAR_GPU * MAX_P1 * MAX_P1];
    for (int v = 0; v < NVAR_GPU; ++v)
        for (int mm = 0; mm < nmodes; ++mm)
            Ucoeff_loc[v * nmodes + mm] = d_Ucoeff[v * nE * nmodes + e * nmodes + mm];

    double rhsBase[NVAR_GPU * MAX_P1 * MAX_P1];
    elementRhsCoeff_d(Ucoeff_loc, d_Ucoeff, e,
        d_detJ, d_dxidx, d_dxidy, d_detadx, d_detady,
        d_faceNx, d_faceNy, d_faceJac,
        d_elem2face, d_face_elemL, d_face_elemR,
        d_face_faceL, d_face_faceR, d_face_bcType,
        nE, P1, nq1d, nmodes, nqVol, nqFace,
        rhoInf, uInf, vInf, pInf, fluxType, d_epsilon, rhsBase);

    double* Jblock = &d_Jac[e * blockSz * blockSz];

    for (int j = threadIdx.x; j < blockSz; j += blockDim.x) {
        double Ucoeff_pert[NVAR_GPU * MAX_P1 * MAX_P1];
        for (int k = 0; k < blockSz; ++k) Ucoeff_pert[k] = Ucoeff_loc[k];

        double h = 1e-7 * fmax(fabs(Ucoeff_pert[j]), 1.0);
        Ucoeff_pert[j] += h;

        double rhsPert[NVAR_GPU * MAX_P1 * MAX_P1];
        elementRhsCoeff_d(Ucoeff_pert, d_Ucoeff, e,
            d_detJ, d_dxidx, d_dxidy, d_detadx, d_detady,
            d_faceNx, d_faceNy, d_faceJac,
            d_elem2face, d_face_elemL, d_face_elemR,
            d_face_faceL, d_face_faceR, d_face_bcType,
            nE, P1, nq1d, nmodes, nqVol, nqFace,
            rhoInf, uInf, vInf, pInf, fluxType, d_epsilon, rhsPert);

        for (int i = 0; i < blockSz; ++i) {
            double Minv_rhs_base = 0.0, Minv_rhs_pert = 0.0;
            int iv = i / nmodes, im = i % nmodes;
            for (int mp = 0; mp < nmodes; ++mp) {
                double Mval = d_Minv[e * nmodes * nmodes + im * nmodes + mp];
                Minv_rhs_base += Mval * rhsBase[iv * nmodes + mp];
                Minv_rhs_pert += Mval * rhsPert[iv * nmodes + mp];
            }
            double dRdU = (Minv_rhs_pert - Minv_rhs_base) / h;
            Jblock[i * blockSz + j] = -dRdU;
            if (i == j)
                Jblock[i * blockSz + j] += 1.0 / dt_e;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < blockSz; ++k) d_JacPiv[e * blockSz + k] = k;

        for (int col = 0; col < blockSz; ++col) {
            int pivot = col;
            double maxVal = fabs(Jblock[col * blockSz + col]);
            for (int r = col + 1; r < blockSz; ++r) {
                double v = fabs(Jblock[r * blockSz + col]);
                if (v > maxVal) { maxVal = v; pivot = r; }
            }
            if (pivot != col) {
                int tmp = d_JacPiv[e * blockSz + col];
                d_JacPiv[e * blockSz + col] = d_JacPiv[e * blockSz + pivot];
                d_JacPiv[e * blockSz + pivot] = tmp;
                for (int c = 0; c < blockSz; ++c) {
                    double t = Jblock[col * blockSz + c];
                    Jblock[col * blockSz + c] = Jblock[pivot * blockSz + c];
                    Jblock[pivot * blockSz + c] = t;
                }
            }
            double diag = Jblock[col * blockSz + col];
            if (fabs(diag) < 1e-30) diag = 1e-30;
            for (int r = col + 1; r < blockSz; ++r) {
                Jblock[r * blockSz + col] /= diag;
                for (int c = col + 1; c < blockSz; ++c)
                    Jblock[r * blockSz + c] -= Jblock[r * blockSz + col] * Jblock[col * blockSz + c];
            }
        }
    }
}

// ============================================================================
// Implicit solver: Block-Jacobi preconditioner apply
// ============================================================================

__global__ void precondApplyKernel(
    const double* __restrict__ d_input,
    double* __restrict__ d_output,
    const double* __restrict__ d_Ucoeff,
    const double* __restrict__ d_Minv,
    const double* __restrict__ d_detJ,
    const double* __restrict__ d_Jac,
    const int* __restrict__ d_JacPiv,
    int nE, int P1, int nq1d, int nmodes, int nqVol)
{
    int e = blockIdx.x;
    if (e >= nE) return;

    int blockSz = NVAR_GPU * nmodes;
    int totalDOF = nE * nqVol;

    double v_coeff[NVAR_GPU * MAX_P1 * MAX_P1];
    for (int v = 0; v < NVAR_GPU; ++v)
        for (int mm = 0; mm < nmodes; ++mm) {
            int mi = mm / P1, mj = mm % P1;
            double val = 0.0;
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    int qIdx = e * nqVol + qx * nq1d + qe;
                    double w = c_wq[qx] * c_wq[qe] * d_detJ[qIdx];
                    val += w * c_Bmat[mi*nq1d+qx] * c_Bmat[mj*nq1d+qe]
                         * d_input[v * totalDOF + qIdx];
                }
            double proj = val;
            val = 0.0;
            for (int mp = 0; mp < nmodes; ++mp)
                val += d_Minv[e*nmodes*nmodes + mm*nmodes + mp] * proj;
            v_coeff[v * nmodes + mm] = val;
        }

    double rhs[NVAR_GPU * MAX_P1 * MAX_P1];
    const double* Jblock = &d_Jac[e * blockSz * blockSz];
    const int* piv = &d_JacPiv[e * blockSz];

    for (int i = 0; i < blockSz; ++i)
        rhs[i] = v_coeff[piv[i]];

    for (int i = 0; i < blockSz; ++i)
        for (int k = 0; k < i; ++k)
            rhs[i] -= Jblock[i * blockSz + k] * rhs[k];

    for (int i = blockSz - 1; i >= 0; --i) {
        for (int k = i + 1; k < blockSz; ++k)
            rhs[i] -= Jblock[i * blockSz + k] * rhs[k];
        rhs[i] /= Jblock[i * blockSz + i];
    }

    for (int v = 0; v < NVAR_GPU; ++v)
        for (int qx = 0; qx < nq1d; ++qx)
            for (int qe = 0; qe < nq1d; ++qe) {
                double val = 0.0;
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        val += rhs[v*nmodes + ii*P1+jj]
                             * c_Bmat[ii*nq1d+qx] * c_Bmat[jj*nq1d+qe];
                d_output[v * totalDOF + e * nqVol + qx * nq1d + qe] = val;
            }
}

// ============================================================================
// Implicit solver: Host BLAS helpers
// ============================================================================

static double gpuDot(const double* d_x, const double* d_y, double* d_buf, int N)
{
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_buf, &zero, sizeof(double), cudaMemcpyHostToDevice));
    int bk = 256;
    int gd = min((N + bk - 1) / bk, 1024);
    dotProductKernel<<<gd, bk, bk * sizeof(double)>>>(d_x, d_y, d_buf, N);
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_buf, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

static double gpuNorm2(const double* d_x, double* d_buf, int N)
{
    return sqrt(gpuDot(d_x, d_x, d_buf, N));
}

static void gpuAxpy(double a, const double* d_x, double* d_y, int N)
{
    int bk = 256;
    int gd = (N + bk - 1) / bk;
    axpyKernel<<<gd, bk>>>(a, d_x, d_y, N);
}

static void gpuScaleVec(double a, double* d_x, int N)
{
    int bk = 256;
    int gd = (N + bk - 1) / bk;
    scaleVecKernel<<<gd, bk>>>(a, d_x, N);
}

// ============================================================================
// Implicit solver: Allocate / Free
// ============================================================================

void gpuImplicitAllocate(ImplicitGPUData& imp, const GPUSolverData& gpu, int maxKrylov)
{
    imp.N = NVAR_GPU * gpu.totalDOF;
    imp.maxKrylov = maxKrylov;
    imp.Unorm = 0.0;
    imp.blockSz = NVAR_GPU * gpu.nmodes;

    CUDA_CHECK(cudaMalloc(&imp.d_R0, imp.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&imp.d_V, (size_t)(maxKrylov + 1) * imp.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&imp.d_w, imp.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&imp.d_dotBuf, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&imp.d_Jac, (size_t)gpu.nE * imp.blockSz * imp.blockSz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&imp.d_JacPiv, gpu.nE * imp.blockSz * sizeof(int)));

    double jacMB = (double)gpu.nE * imp.blockSz * imp.blockSz * sizeof(double) / (1024.0*1024.0);
    double kryMB = (double)(maxKrylov + 1) * imp.N * sizeof(double) / (1024.0*1024.0);
    printf("Implicit solver allocated: %d Krylov vectors (%.1f MB), Jacobian blocks (%.1f MB)\n",
           maxKrylov + 1, kryMB, jacMB);
}

void gpuImplicitFree(ImplicitGPUData& imp)
{
    cudaFree(imp.d_R0);
    cudaFree(imp.d_V);
    cudaFree(imp.d_w);
    cudaFree(imp.d_dotBuf);
    cudaFree(imp.d_Jac);
    cudaFree(imp.d_JacPiv);
}

// ============================================================================
// Implicit solver: Residual norm
// ============================================================================

double gpuResidualNorm(GPUSolverData& gpu, ImplicitGPUData& imp)
{
    return gpuNorm2(gpu.d_R, imp.d_dotBuf, imp.N);
}

void gpuResidualNormPerVar(GPUSolverData& gpu, double norms[4])
{
    for (int v = 0; v < NVAR_GPU; v++) {
        const double* ptr = gpu.d_R + (size_t)v * gpu.totalDOF;
        norms[v] = gpuNorm2(ptr, gpu.d_dtMin, gpu.totalDOF);
    }
}

// ============================================================================
// Implicit solver: Assemble block-Jacobi preconditioner
// ============================================================================

void gpuAssembleBlockJacobi(GPUSolverData& gpu, ImplicitGPUData& imp)
{
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes, nqVol = gpu.nqVol, nqFace = gpu.nqFace;

    int blockDim = min(64, imp.blockSz);
    const double* d_eps_ptr = gpu.useAV ? gpu.d_epsilon : nullptr;
    blockJacobiAssemblyKernel<<<nE, blockDim>>>(
        gpu.d_Ucoeff, gpu.d_Minv, gpu.d_dtLocal,
        gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
        d_eps_ptr,
        imp.d_Jac, imp.d_JacPiv,
        nE, P1, nq1d, nmodes, nqVol, nqFace,
        gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf, gpu.fluxType);
}

// ============================================================================
// Implicit solver: One backward Euler step with right-preconditioned GMRES
//
// Solves (I/dt_e - dR/dU) * dU = R(U) via restarted GMRES with
// matrix-free Jacobian-vector products and block-Jacobi preconditioner,
// then updates U += dU.
// Returns the number of GMRES iterations used.
// ============================================================================

int gpuImplicitStep(GPUSolverData& gpu, ImplicitGPUData& imp,
                    double gmresTol, int gmresMaxIter, double time,
                    double& residualNorm, double* perVarNorms)
{
    int N = imp.N;
    int m = min(gmresMaxIter, imp.maxKrylov);
    int nE = gpu.nE, P1 = gpu.P1, nq1d = gpu.nq1d;
    int nmodes = gpu.nmodes, nqVol = gpu.nqVol;

    gpuComputeDGRHS(gpu, false, time);
    CUDA_CHECK(cudaMemcpy(imp.d_R0, gpu.d_R, N * sizeof(double), cudaMemcpyDeviceToDevice));

    imp.Unorm = gpuNorm2(gpu.d_U, imp.d_dotBuf, N);
    double beta = gpuNorm2(imp.d_R0, imp.d_dotBuf, N);
    residualNorm = beta;

    if (perVarNorms) {
        for (int v = 0; v < NVAR_GPU; v++) {
            const double* ptr = imp.d_R0 + (size_t)v * gpu.totalDOF;
            perVarNorms[v] = gpuNorm2(ptr, imp.d_dotBuf, gpu.totalDOF);
        }
    }

    if (beta < 1e-15) return 0;

    gpuAssembleBlockJacobi(gpu, imp);

    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> g(m + 1, 0.0);
    std::vector<double> cs(m, 0.0), sn(m, 0.0);
    std::vector<double> y(m, 0.0);

    CUDA_CHECK(cudaMemcpy(imp.d_V, imp.d_R0, N * sizeof(double), cudaMemcpyDeviceToDevice));
    gpuScaleVec(1.0 / beta, imp.d_V, N);
    g[0] = beta;

    double absTol = gmresTol * beta;
    int j_final = -1;

    for (int j = 0; j < m; ++j) {
        double* vj = imp.d_V + (size_t)j * N;
        double* w  = imp.d_V + (size_t)(j + 1) * N;

        precondApplyKernel<<<nE, 1>>>(
            vj, imp.d_w, gpu.d_Ucoeff, gpu.d_Minv, gpu.d_detJ,
            imp.d_Jac, imp.d_JacPiv, nE, P1, nq1d, nmodes, nqVol);

        double znorm = gpuNorm2(imp.d_w, imp.d_dotBuf, N);
        double eps = 1.49e-8 * (1.0 + imp.Unorm) / fmax(znorm, 1e-30);

        CUDA_CHECK(cudaMemcpy(gpu.d_Utmp, gpu.d_U, N * sizeof(double), cudaMemcpyDeviceToDevice));
        gpuAxpy(eps, imp.d_w, gpu.d_Utmp, N);

        gpuComputeDGRHS(gpu, true, time);

        {
            int bk = 256;
            int gd = (N + bk - 1) / bk;
            implicitMatvecKernel<<<gd, bk>>>(gpu.d_R, imp.d_R0, imp.d_w, w,
                gpu.d_dtLocal, 1.0/eps, gpu.totalDOF, gpu.nqVol, N);
        }

        for (int i = 0; i <= j; ++i) {
            double* vi = imp.d_V + (size_t)i * N;
            double hij = gpuDot(w, vi, imp.d_dotBuf, N);
            H[i * m + j] = hij;
            gpuAxpy(-hij, vi, w, N);
        }

        double wnorm = gpuNorm2(w, imp.d_dotBuf, N);
        H[(j + 1) * m + j] = wnorm;

        if (wnorm > 1e-14)
            gpuScaleVec(1.0 / wnorm, w, N);

        for (int i = 0; i < j; ++i) {
            double temp       =  cs[i] * H[i * m + j] + sn[i] * H[(i + 1) * m + j];
            H[(i + 1) * m + j] = -sn[i] * H[i * m + j] + cs[i] * H[(i + 1) * m + j];
            H[i * m + j]       = temp;
        }

        double r = sqrt(H[j * m + j] * H[j * m + j] +
                        H[(j + 1) * m + j] * H[(j + 1) * m + j]);
        if (r < 1e-30) r = 1e-30;
        cs[j] = H[j * m + j] / r;
        sn[j] = H[(j + 1) * m + j] / r;
        H[j * m + j] = r;
        H[(j + 1) * m + j] = 0.0;

        double temp = cs[j] * g[j];
        g[j + 1] = -sn[j] * g[j];
        g[j] = temp;

        j_final = j;

        if (fabs(g[j + 1]) < absTol || wnorm < 1e-14)
            break;
    }

    if (j_final < 0) return 0;

    for (int i = j_final; i >= 0; --i) {
        y[i] = g[i];
        for (int k = i + 1; k <= j_final; ++k)
            y[i] -= H[i * m + k] * y[k];
        y[i] /= H[i * m + i];
    }

    CUDA_CHECK(cudaMemset(imp.d_w, 0, N * sizeof(double)));
    for (int i = 0; i <= j_final; ++i) {
        double* vi = imp.d_V + (size_t)i * N;
        gpuAxpy(y[i], vi, imp.d_w, N);
    }

    precondApplyKernel<<<nE, 1>>>(
        imp.d_w, gpu.d_Utmp, gpu.d_Ucoeff, gpu.d_Minv, gpu.d_detJ,
        imp.d_Jac, imp.d_JacPiv, nE, P1, nq1d, nmodes, nqVol);

    gpuAxpy(1.0, gpu.d_Utmp, gpu.d_U, N);

    {
        int bk = 256;
        int gd = (gpu.totalDOF + bk - 1) / bk;
        positivityKernel<<<gd, bk>>>(gpu.d_U, gpu.totalDOF, 1e-6, 1e-6);
    }

    return j_final + 1;
}
