#ifndef EULER2D_GPU_H
#define EULER2D_GPU_H

#include <vector>

#define NVAR_GPU 4

struct GPUSolverData {
    int nE, nF, P, nq1d, nqVol, nqFace, nmodes, totalDOF;
    int P1; // P + 1

    // Geometry volume [nE * nqVol]
    double* d_detJ;
    double* d_dxidx;
    double* d_dxidy;
    double* d_detadx;
    double* d_detady;

    // Geometry face [nF * nqFace]
    double* d_faceNx;
    double* d_faceNy;
    double* d_faceJac;
    double* d_faceXPhys;
    double* d_faceYPhys;

    // Basis [(P+1) * nq1d] and [(P+1) * 2]
    double* d_Bmat;
    double* d_Dmat;
    double* d_blr;

    // Mass matrix inverse [nE * nmodes * nmodes]
    double* d_Minv;

    // Quadrature weights [nq1d]
    double* d_wq;

    // Face connectivity
    int* d_elem2face;   // [nE * 4]
    int* d_face_elemL;  // [nF]
    int* d_face_elemR;  // [nF]
    int* d_face_faceL;  // [nF]
    int* d_face_faceR;  // [nF]
    int* d_face_bcType; // [nF]: 0=interior, 1=slipwall, 2=farfield

    // Solution arrays [NVAR * totalDOF]
    double* d_U;
    double* d_R;
    double* d_k1;
    double* d_k2;
    double* d_k3;
    double* d_k4;
    double* d_Utmp;

    // Working arrays
    double* d_Ucoeff;   // [NVAR * nE * nmodes]
    double* d_rhsCoeff; // [NVAR * nE * nmodes]

    // CFL reduction scratch
    double* d_dtMin;    // [1]
    double* d_dtLocal;  // [nE] per-element dt (used by implicit solver)

    // NaN flag
    int* d_nanFlag;     // [1]

    // Freestream state (for boundary conditions)
    double rhoInf, uInf, vInf, pInf;

    // Flux type: 0 = Lax-Friedrichs, 1 = HLLC
    int fluxType;

    // Previous solution snapshot for NaN recovery [NVAR * totalDOF]
    double* d_Uprev;

    // Artificial viscosity (Persson-Peraire + LDG)
    double* d_epsilon;  // [nE] per-element viscosity coefficient
    double* d_sensor;   // [nE] per-element raw sensor value (log10)
    double* d_Qcoeff;   // [NVAR * 2 * nE * nmodes] LDG auxiliary gradient
    bool useAV;
    double AVs0, AVkappa, AVscale;
};

struct ImplicitGPUData {
    int N;
    int maxKrylov;
    double* d_R0;       // saved base residual [NVAR * totalDOF]
    double* d_V;        // Krylov basis [(maxKrylov+1) * N]
    double* d_w;        // preconditioner work vector [N]
    double* d_dotBuf;   // reduction scratch [1]
    double* d_Jac;      // block-Jacobi LU factors [nE * blockSz * blockSz]
    int*    d_JacPiv;   // block-Jacobi pivots [nE * blockSz]
    double Unorm;
    int blockSz;        // NVAR * nmodes
};

void gpuAllocate(GPUSolverData& gpu, int nE, int nF, int P, int nq1d);
void gpuFree(GPUSolverData& gpu);

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
    const int* face_bcType);

void gpuCopySolutionToDevice(GPUSolverData& gpu, const double* U_flat);
void gpuCopySolutionToHost(GPUSolverData& gpu, double* U_flat);

void gpuComputeDGRHS(GPUSolverData& gpu, bool useUtmp, double time);
void gpuRK4Stage(GPUSolverData& gpu, double dt, int stage);
double gpuComputeCFL(GPUSolverData& gpu, double CFL, int P);
void gpuComputeElementCFL(GPUSolverData& gpu, double CFL, int P);
bool gpuCheckNaN(GPUSolverData& gpu);
void gpuSetNodalToModal(const double* T, int P1);
void gpuCopyEpsilonToHost(GPUSolverData& gpu, double* eps_host);
void gpuCopySensorToHost(GPUSolverData& gpu, double* sensor_host);
void gpuSnapshotSolution(GPUSolverData& gpu);
void gpuCopyPrevSolutionToHost(GPUSolverData& gpu, double* U_flat);
void gpuRestoreSnapshot(GPUSolverData& gpu);

void gpuImplicitAllocate(ImplicitGPUData& imp, const GPUSolverData& gpu, int maxKrylov);
void gpuImplicitFree(ImplicitGPUData& imp);
double gpuResidualNorm(GPUSolverData& gpu, ImplicitGPUData& imp);
void gpuResidualNormPerVar(GPUSolverData& gpu, double norms[4]);
void gpuAssembleBlockJacobi(GPUSolverData& gpu, ImplicitGPUData& imp);
int gpuImplicitStep(GPUSolverData& gpu, ImplicitGPUData& imp,
                    double gmresTol, int gmresMaxIter, double time,
                    double& residualNorm,
                    double* perVarNorms = nullptr);

#endif
