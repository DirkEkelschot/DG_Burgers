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
                                     const double* __restrict__ Bmat,
                                     const double* __restrict__ blr,
                                     int P1, int nq1d)
{
    int nqF = nq1d;
    double phiXi, phiEta;
    switch (lf) {
        case 0: phiXi = Bmat[i * nq1d + q];               phiEta = blr[j * 2 + 0]; break;
        case 1: phiXi = blr[i * 2 + 1];                   phiEta = Bmat[j * nq1d + q]; break;
        case 2: phiXi = Bmat[i * nq1d + (nqF - 1 - q)];   phiEta = blr[j * 2 + 1]; break;
        case 3: phiXi = blr[i * 2 + 0];                   phiEta = Bmat[j * nq1d + (nqF - 1 - q)]; break;
        default: phiXi = 0; phiEta = 0;
    }
    return phiXi * phiEta;
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
    const double* __restrict__ d_Bmat,
    const double* __restrict__ d_Minv,
    const double* __restrict__ d_wq,
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
            double Bi = d_Bmat[mi * nq1d + qx];
            for (int qe = 0; qe < nq1d; ++qe) {
                int qIdx = e * nqVol + qx * nq1d + qe;
                double weight = d_wq[qx] * d_wq[qe] * d_detJ[qIdx];
                val += weight * Bi * d_Bmat[mj * nq1d + qe]
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
// Kernel 2: Volume + surface integral (flat, 1 thread per (e,v,m))
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
    const double* __restrict__ d_Bmat,
    const double* __restrict__ d_Dmat,
    const double* __restrict__ d_blr,
    const double* __restrict__ d_wq,
    const int* __restrict__ d_elem2face,
    const int* __restrict__ d_face_elemL,
    const int* __restrict__ d_face_elemR,
    const int* __restrict__ d_face_faceL,
    const int* __restrict__ d_face_faceR,
    const int* __restrict__ d_face_bcType,
    int nE, int P1, int nq1d, int nmodes, int nqVol, int totalDOF, int nqFace,
    double rhoInf, double uInf, double vInf, double pInf)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nE * NVAR_GPU * nmodes;
    if (tid >= total) return;

    int m = tid % nmodes;
    int v = (tid / nmodes) % NVAR_GPU;
    int e = tid / (NVAR_GPU * nmodes);

    int mi = m / P1;
    int mj = m % P1;

    // ===== Volume integral =====
    double vol = 0.0;
    for (int qx = 0; qx < nq1d; ++qx) {
        double Bi  = d_Bmat[mi * nq1d + qx];
        double Di  = d_Dmat[mi * nq1d + qx];
        for (int qe = 0; qe < nq1d; ++qe) {
            int qIdx = e * nqVol + qx * nq1d + qe;
            double w = d_wq[qx] * d_wq[qe] * d_detJ[qIdx];

            double rho  = d_U[0 * totalDOF + qIdx];
            double rhou = d_U[1 * totalDOF + qIdx];
            double rhov = d_U[2 * totalDOF + qIdx];
            double rhoE = d_U[3 * totalDOF + qIdx];

            double uv = rhou / rho;
            double vv = rhov / rho;
            double p  = (GAMMA_GPU - 1.0) * (rhoE - 0.5 * rho * (uv*uv + vv*vv));

            double fx, fy;
            switch (v) {
                case 0: fx = rhou;           fy = rhov;           break;
                case 1: fx = rhou*uv + p;    fy = rhov*uv;        break;
                case 2: fx = rhou*vv;        fy = rhov*vv + p;    break;
                default: fx = (rhoE+p)*uv;   fy = (rhoE+p)*vv;    break;
            }

            double Bj  = d_Bmat[mj * nq1d + qe];
            double Dj  = d_Dmat[mj * nq1d + qe];
            double dphidxi  = Di * Bj;
            double dphideta = Bi * Dj;
            double dphidx = d_dxidx[qIdx]  * dphidxi + d_detadx[qIdx] * dphideta;
            double dphidy = d_dxidy[qIdx]  * dphidxi + d_detady[qIdx] * dphideta;

            vol += w * (fx * dphidx + fy * dphidy);
        }
    }

    // ===== Surface integral (element-centric, no atomics) =====
    double surf = 0.0;
    for (int lf = 0; lf < 4; ++lf) {
        int f  = d_elem2face[e * 4 + lf];
        int eL = d_face_elemL[f];
        int eR = d_face_elemR[f];
        bool is_left = (e == eL);
        int eN  = is_left ? eR : eL;
        int lfN = is_left ? d_face_faceR[f] : d_face_faceL[f];
        bool is_boundary = (eN < 0);

        for (int q = 0; q < nqFace; ++q) {
            int q_face = is_left ? q : (nqFace - 1 - q);
            int fIdx = f * nqFace + q_face;
            double nx = d_faceNx[fIdx];
            double ny = d_faceNy[fIdx];
            double wf = d_wq[q] * d_faceJac[fIdx];

            // Evaluate own trace
            double UMe[4] = {0, 0, 0, 0};
            for (int vv = 0; vv < 4; ++vv)
                for (int ii = 0; ii < P1; ++ii)
                    for (int jj = 0; jj < P1; ++jj)
                        UMe[vv] += d_Ucoeff[vv * nE * nmodes + e * nmodes + ii*P1+jj]
                                 * evalPhiFace(lf, ii, jj, q, d_Bmat, d_blr, P1, nq1d);

            // Evaluate neighbor trace or boundary ghost
            double UNbr[4];
            if (!is_boundary) {
                int qN = nqFace - 1 - q;
                for (int vv = 0; vv < 4; ++vv) {
                    UNbr[vv] = 0.0;
                    for (int ii = 0; ii < P1; ++ii)
                        for (int jj = 0; jj < P1; ++jj)
                            UNbr[vv] += d_Ucoeff[vv * nE * nmodes + eN * nmodes + ii*P1+jj]
                                      * evalPhiFace(lfN, ii, jj, qN, d_Bmat, d_blr, P1, nq1d);
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
            if (is_left)
                laxFriedrichs_d(UMe, UNbr, nx, ny, Fnum);
            else
                laxFriedrichs_d(UNbr, UMe, nx, ny, Fnum);

            double sign = is_left ? -1.0 : 1.0;
            double phi = evalPhiFace(lf, mi, mj, q, d_Bmat, d_blr, P1, nq1d);
            surf += sign * wf * Fnum[v] * phi;
        }
    }

    d_rhsCoeff[v * nE * nmodes + e * nmodes + m] = vol + surf;
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
    const double* __restrict__ d_Bmat,
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
            double Bi = d_Bmat[i * nq1d + qx];
            for (int j = 0; j < P1; ++j)
                val += dUdt[v * nmodes + i * P1 + j] * Bi * d_Bmat[j * nq1d + qe];
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
    const double* __restrict__ d_detJ,
    double* __restrict__ d_dtMin,
    int nE, int nqVol, int totalDOF, double CFL, int P)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double myMin = 1e20;

    for (int i = idx; i < nE * nqVol; i += gridDim.x * blockDim.x) {
        int e = i / nqVol;
        double det = fabs(d_detJ[e * nqVol]);
        double h   = sqrt(det);

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
    CUDA_CHECK(cudaMalloc(&gpu.d_nanFlag, sizeof(int)));
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
    cudaFree(gpu.d_dtMin);   cudaFree(gpu.d_nanFlag);
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

    CUDA_CHECK(cudaMemcpy(gpu.d_elem2face,   elem2face_flat, nE*4*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_elemL,  face_elemL,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_elemR,  face_elemR,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_faceL,  face_faceL,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_faceR,  face_faceR,  nF*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_face_bcType, face_bcType, nF*sizeof(int), cudaMemcpyHostToDevice));
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

    int blockDim1 = 64;
    int smem1 = NVAR_GPU * nmodes * sizeof(double);
    forwardTransformKernel<<<nE, blockDim1, smem1>>>(
        d_Uin, gpu.d_Ucoeff, gpu.d_detJ, gpu.d_Bmat, gpu.d_Minv, gpu.d_wq,
        nE, P1, nq1d, nmodes, nqVol, totalDOF);

    int totalThreads2 = nE * NVAR_GPU * nmodes;
    int blockDim2 = 256;
    int gridDim2 = (totalThreads2 + blockDim2 - 1) / blockDim2;
    volumeSurfaceKernel<<<gridDim2, blockDim2>>>(
        d_Uin, gpu.d_Ucoeff, gpu.d_rhsCoeff,
        gpu.d_detJ, gpu.d_dxidx, gpu.d_dxidy, gpu.d_detadx, gpu.d_detady,
        gpu.d_faceNx, gpu.d_faceNy, gpu.d_faceJac,
        gpu.d_Bmat, gpu.d_Dmat, gpu.d_blr, gpu.d_wq,
        gpu.d_elem2face, gpu.d_face_elemL, gpu.d_face_elemR,
        gpu.d_face_faceL, gpu.d_face_faceR, gpu.d_face_bcType,
        nE, P1, nq1d, nmodes, nqVol, totalDOF, nqFace,
        gpu.rhoInf, gpu.uInf, gpu.vInf, gpu.pInf);

    int blockDim3 = 64;
    int smem3 = NVAR_GPU * nmodes * sizeof(double);
    massSolveBackwardKernel<<<nE, blockDim3, smem3>>>(
        gpu.d_rhsCoeff, gpu.d_R, gpu.d_Minv, gpu.d_Bmat,
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
        gpu.d_U, gpu.d_detJ, gpu.d_dtMin,
        gpu.nE, gpu.nqVol, gpu.totalDOF, CFL, P);

    double dtMin;
    CUDA_CHECK(cudaMemcpy(&dtMin, gpu.d_dtMin, sizeof(double), cudaMemcpyDeviceToHost));
    return dtMin;
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
