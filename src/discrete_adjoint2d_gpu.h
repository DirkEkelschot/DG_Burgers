#ifndef DISCRETE_ADJOINT2D_GPU_H
#define DISCRETE_ADJOINT2D_GPU_H

#include "euler2d_gpu.h"

struct DiscreteAdjointGPUData {
    bool modalMode;
    int  primaryDOF;
    int  totalCoeff;

    // Solution arrays [NVAR * primaryDOF]
    double* d_psi;
    double* d_psiTmp;
    double* d_pk1;
    double* d_pk2;
    double* d_pk3;
    double* d_pk4;
    double* d_adjR;

    // Coefficient-space working arrays [NVAR * nE * nmodes]
    double* d_psiCoeff;
    double* d_adjRhsCoeff;

    // Quad-point working buffer (modal mode) [NVAR * totalDOF]
    double* d_psi_quad;

    // Objective gradient in coefficient space [NVAR * nE * nmodes]
    double* d_dJdUcoeff;

    double* d_dtMin;
    int*    d_nanFlag;
    double* d_liftBuf;
    double* d_normBuf;

    double chordRef;
    double forceNx;
    double forceNy;
};

void discreteAdjointAllocate(DiscreteAdjointGPUData& da, const GPUSolverData& gpu);
void discreteAdjointFree(DiscreteAdjointGPUData& da);

void discreteAdjointSetBasisData(const double* Bmat, const double* Dmat,
                                 const double* blr, const double* wq,
                                 int P1, int nq1d);
void discreteAdjointSetNodalToModal(const double* T, int P1);

double discreteAdjointComputeForceCoeff(DiscreteAdjointGPUData& da,
                                        const GPUSolverData& gpu,
                                        double chordRef,
                                        double forceNx, double forceNy);

void discreteAdjointComputeObjectiveGradient(DiscreteAdjointGPUData& da,
                                             const GPUSolverData& gpu,
                                             double chordRef,
                                             double forceNx, double forceNy);

void discreteAdjointComputeRHS(DiscreteAdjointGPUData& da,
                               const GPUSolverData& gpu,
                               bool usePsiTmp);

void discreteAdjointRK4Stage(DiscreteAdjointGPUData& da, double dt,
                             int stage, int N);

bool discreteAdjointCheckNaN(DiscreteAdjointGPUData& da, int N);

double discreteAdjointResidualL2(DiscreteAdjointGPUData& da, int N);
void   discreteAdjointResidualNormPerVar(DiscreteAdjointGPUData& da,
                                         int totalDOF, double norms[4]);

void discreteAdjointCopySolutionToHost(DiscreteAdjointGPUData& da,
                                       double* psi_flat, int N);
void discreteAdjointCopyQuadPointsToHost(DiscreteAdjointGPUData& da,
                                         const GPUSolverData& gpu,
                                         double* psi_quad_flat);
void discreteAdjointCopySolutionToDevice(DiscreteAdjointGPUData& da,
                                         const double* psi_flat, int N);

#endif
