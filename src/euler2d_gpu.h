#ifndef EULER2D_GPU_H
#define EULER2D_GPU_H

#include <vector>
#include <map>

struct Mesh2D;

#define NVAR_GPU 4

struct cublasContext;
typedef cublasContext* cublasHandle_t;

// Per-P-group coefficient arrays and basis data (used with variable-P)
struct PGroupGPU {
    int P, P1, nmodes;
    int nEGroup;
    int*    d_elemIdx;   // [nEGroup] maps local -> global element index
    double* d_Ucoeff;    // [NVAR * nEGroup * nmodes]
    double* d_rhsCoeff;  // [NVAR * nEGroup * nmodes]
    double* d_Minv;      // [nEGroup * nmodes * nmodes]
    double* d_Qcoeff;    // [NVAR * 2 * nEGroup * nmodes]
    // Host-side basis data (uploaded to constant memory before each group launch)
    std::vector<double> h_Bmat, h_Dmat, h_blr, h_wq, h_faceInterp, h_NodalToModal;
};

// p-Multigrid level data
struct pMGLevel {
    int P, P1, nmodes;
    int blockSize;              // NVAR * nmodes

    double* d_coeff   = nullptr; // [NVAR * nE * nmodes]
    double* d_rhs     = nullptr;
    double* d_tmp     = nullptr;
    double* d_Rbase_c = nullptr;

    double*  d_diagBlocks = nullptr;
    int*     d_pivots     = nullptr;
    int*     d_bjInfo     = nullptr;
    double** d_blockPtrs  = nullptr;
    double*  d_bjPacked   = nullptr;
    double** d_bjRhsPtrs  = nullptr;
    bool     assembled    = false;

    PGroupGPU group;
};

struct GPUSolverData {
    int nE, nF, P, nq1d, nqVol, nqFace, nmodes, totalDOF;
    int P1; // P + 1

    // Modal coefficient-space mode: when true, d_U/d_Utmp/d_k*/d_R/d_Uprev
    // store expansion coefficients (size NVAR * totalCoeff) and d_U_quad
    // holds quadrature-point values as a working buffer.
    bool modalMode;
    int totalCoeff;  // nE * nmodes
    int primaryDOF;  // totalCoeff (modal) or totalDOF (nodal)

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

    // Solution arrays [NVAR * primaryDOF]
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
    double* d_U_quad;   // [NVAR * totalDOF] quad-point values (modal mode only)

    // CFL reduction scratch
    double* d_dtMin;    // [1]

    // NaN flag
    int* d_nanFlag;     // [1]

    // Fused per-variable residual norm scratch
    double* d_normBuf;  // [4]

    // Freestream state (for boundary conditions)
    double rhoInf, uInf, vInf, pInf;

    // Flux type: 0 = Lax-Friedrichs, 1 = HLLC
    int fluxType;

    // Previous solution snapshot for NaN recovery [NVAR * primaryDOF]
    double* d_Uprev;

    // Artificial viscosity (Persson-Peraire + LDG)
    double* d_epsilon;  // [nE] per-element viscosity coefficient
    double* d_sensor;   // [nE] per-element raw sensor value (log10)
    double* d_Qcoeff;   // [NVAR * 2 * nE * nmodes] LDG auxiliary gradient
    bool useAV;
    bool freezeAV = false; // when true, skip shockSensorKernel and reuse existing d_epsilon
    double AVs0, AVkappa, AVscale;

    // Implicit solver (JFNK + block-Jacobi)
    bool   useImplicit = false;
    double* d_Un       = nullptr;  // [NVAR * primaryDOF] U^n snapshot
    double* d_Rbase    = nullptr;  // [NVAR * primaryDOF] cached R~(U^k)
    double* d_newton_rhs = nullptr; // [NVAR * primaryDOF] Newton RHS

    // GMRES Krylov basis: (restart+1) device vectors, each [NVAR * primaryDOF]
    int     gmres_restart = 0;
    int     gmres_nvec    = 0;
    double** d_V          = nullptr;  // host array of device pointers
    double*  d_V_pool     = nullptr;  // contiguous device storage for all Krylov vectors

    // Block-Jacobi preconditioner
    int     jacBlockSize  = 0;     // NVAR * nmodes (64 for P=3)
    double* d_diagBlocks  = nullptr; // [nE * jacBlockSize * jacBlockSize]
    int*    d_pivots      = nullptr; // [nE * jacBlockSize]
    int*    d_bjInfo       = nullptr; // [nE] batched LU info
    double** d_blockPtrs  = nullptr; // [nE] device array of pointers into d_diagBlocks
    double*  d_bjPacked   = nullptr; // [nE * jacBlockSize] packed RHS for batched solve
    double** d_bjRhsPtrs  = nullptr; // [nE] device pointers into d_bjPacked
    bool    bjAssembled   = false;
    bool    useBlockJacobi = true;

    // Element graph coloring for clean block-Jacobi assembly
    int     nColors        = 0;
    std::vector<std::vector<int>> colorGroups; // colorGroups[c] = list of element indices
    int*    d_colorElems   = nullptr; // [nE] flattened color group indices on device
    int*    d_colorOffsets = nullptr; // [nColors+1] offsets into d_colorElems

    cublasHandle_t cublasH = nullptr;

    // p-Multigrid preconditioner levels
    int pmg_nLevels  = 0;
    int pmg_nuPre    = 2;
    int pmg_nuPost   = 2;
    int pmg_nuCoarse = 10;
    std::vector<pMGLevel> pmg_levels;
    PGroupGPU fineBasisGroup; // fine-level basis for restoring constant memory after coarse ops
};

void gpuAllocate(GPUSolverData& gpu, int nE, int nF, int P, int nq1d,
                 bool modalMode = false);
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
int gpuCFLDiagnostic(GPUSolverData& gpu, double CFL, int P, const Mesh2D& mesh);
bool gpuCheckNaN(GPUSolverData& gpu);
void gpuSetNodalToModal(const double* T, int P1);
void gpuSetFaceInterp(const double* interpL, const double* interpR, int nq1d);
void gpuBackwardTransform(GPUSolverData& gpu, const double* d_coeffs,
                          double* d_quad);
void gpuCopyQuadPointsToHost(GPUSolverData& gpu, double* U_flat);
void gpuCopyPrevQuadPointsToHost(GPUSolverData& gpu, double* U_flat);
void gpuCopyEpsilonToHost(GPUSolverData& gpu, double* eps_host);
void gpuCopySensorToHost(GPUSolverData& gpu, double* sensor_host);
void gpuSnapshotSolution(GPUSolverData& gpu);
void gpuCopyPrevSolutionToHost(GPUSolverData& gpu, double* U_flat);
void gpuRestoreSnapshot(GPUSolverData& gpu);
void gpuResidualNormPerVarFused(GPUSolverData& gpu, double norms[4]);
void gpuSyncUcoeff(GPUSolverData& gpu);

// Force coefficients (wall pressure integration on GPU)
void gpuComputeForceCoeff(GPUSolverData& gpu,
                          double chordRef, double AoA_deg,
                          double& Cl, double& Cd);

// Update geometry arrays on GPU (for mesh deformation)
void gpuUpdateGeometry(GPUSolverData& gpu,
                       const double* detJ, const double* dxidx, const double* dxidy,
                       const double* detadx, const double* detady,
                       const double* faceNx, const double* faceNy, const double* faceJac,
                       const double* faceXPhys, const double* faceYPhys);

// Update mass matrix inverse on GPU (for mesh deformation)
void gpuUpdateMinv(GPUSolverData& gpu, const double* Minv);

// Variable-P support
void gpuAllocateGroup(PGroupGPU& grp, int P, int nEGroup);
void gpuFreeGroup(PGroupGPU& grp);
void gpuUploadGroupElemIdx(PGroupGPU& grp, const int* elemIdx);
void gpuUploadGroupMinv(PGroupGPU& grp, const double* Minv);
void gpuUploadGroupBasis(const PGroupGPU& grp);
void gpuComputeDGRHS_group(GPUSolverData& gpu, PGroupGPU& grp,
                           bool useUtmp, double time);

// Kernel wrappers for p-multigrid coarse-level DG operator
void gpuLaunchBackwardTransform(GPUSolverData& gpu,
                                const double* d_coeffs, double* d_quad,
                                int P1, int nmodes);
void gpuLaunchVolumeSurfaceAV(GPUSolverData& gpu,
                              const double* d_quad, const double* d_coeffs,
                              double* d_rhsCoeff,
                              int P1, int nmodes);
void gpuLaunchMassSolveModal(const double* d_rhsCoeff, double* d_R,
                             const double* d_Minv,
                             int nE, int nmodes);

// Implicit solver (JFNK + block-Jacobi) -- implemented in implicit_solver.cu
void gpuImplicitAllocate(GPUSolverData& gpu, int gmres_restart);
void gpuBuildElementColoring(GPUSolverData& gpu,
                             const std::vector<std::array<int,2>>& faceElems);
void gpuImplicitFree(GPUSolverData& gpu);

void jfnkMatvec(GPUSolverData& gpu, const double* d_v, double* d_w,
                double dt, double time);

int gmresSolve(GPUSolverData& gpu, double* d_x, const double* d_b,
               double dt, double time, double tol, int maxRestarts);

void assembleBlockJacobi(GPUSolverData& gpu, double dt, double time);

void applyBlockJacobi(GPUSolverData& gpu, const double* d_r, double* d_z);

int backwardEulerStep(GPUSolverData& gpu, double dt, double time,
                      int maxNewton, double newtonTol,
                      double gmresTol, int maxGMRESRestarts);

// p-Multigrid preconditioner
void pmgAllocateLevel(pMGLevel& level, int P, int nE, cublasHandle_t cublasH);
void pmgFreeLevel(pMGLevel& level);
void pmgApplyPrecond(GPUSolverData& gpu, std::vector<pMGLevel>& levels,
                     const double* d_r, double* d_z, double dt, double time);
void pmgAssembleSmootherLevel(GPUSolverData& gpu, pMGLevel& level,
                              double dt, double time);

#endif
