#ifndef EULER2D_GPU_H
#define EULER2D_GPU_H

#include <vector>
#include <map>

#define NVAR_GPU 4

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
    double AVs0, AVkappa, AVscale;
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
int gpuCFLDiagnostic(GPUSolverData& gpu, double CFL, int P);
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

#endif
