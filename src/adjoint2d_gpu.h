#ifndef ADJOINT2D_GPU_H
#define ADJOINT2D_GPU_H

#include "euler2d_gpu.h"

struct AdjointGPUData {
    double* d_psi;
    double* d_psiTmp;
    double* d_pk1;
    double* d_pk2;
    double* d_pk3;
    double* d_pk4;
    double* d_adjR;

    double* d_psiCoeff;
    double* d_adjVolCoeff;
    double* d_adjRhsCoeff;
    double* d_adjSurfQuad;

    double* d_dJdU;

    double* d_alphaFace;

    double* d_dtMin;
    int*    d_nanFlag;
    double* d_liftBuf;
    double* d_normBuf;  // [4] scratch for per-variable residual norms

    double* d_adjEpsilon;  // [nE] adjoint-specific AV coefficient
    double* d_adjSensor;   // [nE] adjoint sensor value

    double chordRef;
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
void adjointCopySolutionToDevice(AdjointGPUData& adj, const double* psi_flat, int N);

#endif
