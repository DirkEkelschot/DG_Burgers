#ifndef ADJOINT2D_GPU_H
#define ADJOINT2D_GPU_H

#include "euler2d_gpu.h"

// Per-P-group metadata for variable-P adjoint.
// Coefficient arrays are GLOBAL (nE_total * nmodes_max stride).
// Each group only processes its elements via d_elemIdx.
struct AdjointPGroup {
    int P, P1, nmodes;
    int nEGroup;
    int* d_elemIdx;   // [nEGroup] maps local block index → global element index
    // Host-side basis data for constant memory upload
    std::vector<double> h_Bmat, h_Dmat, h_blr, h_wq, h_NodalToModal;
};

struct AdjointGPUData {
    // Modal/nodal branching (mirrors GPUSolverData)
    bool modalMode;
    int  primaryDOF;   // totalCoeff (modal) or totalDOF (nodal)
    int  totalCoeff;   // nE * nmodes

    // Solution arrays [NVAR * primaryDOF]
    double* d_psi;
    double* d_psiTmp;
    double* d_pk1;
    double* d_pk2;
    double* d_pk3;
    double* d_pk4;
    double* d_adjR;

    // Coefficient-space working arrays [NVAR * nE * nmodes]
    double* d_psiCoeff;     // nodal path: projected coefficients; modal: alias for psi
    double* d_adjVolCoeff;
    double* d_adjRhsCoeff;
    double* d_adjSurfQuad;

    // Quad-point working buffer (modal mode only) [NVAR * totalDOF]
    double* d_psi_quad;

    // Objective gradient
    double* d_dJdU;         // nodal: quad-point dJ/dU; modal: unused
    double* d_dJdUcoeff;    // modal: coefficient-space dJ/dU; nodal: unused

    double* d_alphaFace;

    double* d_dtMin;
    int*    d_nanFlag;
    double* d_liftBuf;
    double* d_normBuf;  // [4] scratch for per-variable residual norms

    double* d_adjEpsilon;  // [nE] adjoint-specific AV coefficient
    double* d_adjSensor;   // [nE] adjoint sensor value

    double chordRef;
    bool   fullAV;         // use forward AV scale without clamping (modal basis)
    double maxAVscale;     // CFL-derived cap: P / (2 * CFL * (2P+1))
    double forceNx;
    double forceNy;
};

void adjointGpuAllocate(AdjointGPUData& adj, const GPUSolverData& gpu);
void adjointGpuFree(AdjointGPUData& adj);

void adjointGpuSetBasisData(const double* Bmat, const double* Dmat,
                            const double* blr, const double* wq,
                            int P1, int nq1d);
void adjointGpuSetNodalToModal(const double* T, int P1);

void adjointComputeFrozenAlpha(AdjointGPUData& adj, const GPUSolverData& gpu);

double adjointComputeForceCoeff(AdjointGPUData& adj, const GPUSolverData& gpu,
                                double chordRef, double forceNx, double forceNy);

void adjointComputeObjectiveGradient(AdjointGPUData& adj, const GPUSolverData& gpu,
                                     double chordRef, double forceNx, double forceNy);

void adjointComputeRHS(AdjointGPUData& adj, const GPUSolverData& gpu, bool usePsiTmp);

void adjointRK4Stage(AdjointGPUData& adj, double dt, int stage, int N);

bool adjointCheckNaN(AdjointGPUData& adj, int N);

double adjointResidualL2(AdjointGPUData& adj, int N);
void   adjointResidualNormPerVar(AdjointGPUData& adj, int totalDOF, double norms[4]);

void adjointCopySolutionToHost(AdjointGPUData& adj, double* psi_flat, int N);
void adjointCopyQuadPointsToHost(AdjointGPUData& adj, const GPUSolverData& gpu,
                                 double* psi_quad_flat);
void adjointCopySolutionToDevice(AdjointGPUData& adj, const double* psi_flat, int N);

// Variable-P adjoint support
void adjointAllocatePGroup(AdjointPGroup& grp, int P, int nEGroup);
void adjointFreePGroup(AdjointPGroup& grp);
void adjointUploadPGroupElemIdx(AdjointPGroup& grp, const int* elemIdx);
void adjointUploadPGroupBasis(const AdjointPGroup& grp);
void adjointZeroCoeffArrays(AdjointGPUData& adj, const GPUSolverData& gpu);
void adjointAddObjectiveGradient(AdjointGPUData& adj, const GPUSolverData& gpu);
void adjointComputeRHS_group(AdjointGPUData& adj, const GPUSolverData& gpu,
                             const AdjointPGroup& grp, bool usePsiTmp);

#endif
