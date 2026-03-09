#ifndef SHAPE_SENSITIVITY_H
#define SHAPE_SENSITIVITY_H

#include "euler2d_gpu.h"
#include "discrete_adjoint2d_gpu.h"
#include "hicks_henne.h"
#include "mesh_deform.h"
#include "geom2d.h"
#include <vector>

// Compute dJ/d(alpha_k) for all design variables via finite differences.
// psi^T * dR/dalpha is computed on GPU using a dot product kernel.
std::vector<double> computeShapeGradient(
    GPUSolverData& gpu,
    DiscreteAdjointGPUData& da,
    Mesh2D& mesh,
    const std::vector<std::array<double, 2>>& baselineNodes,
    const GeomData2D& baselineGeom,
    const HicksHenneParam& hh,
    const std::vector<double>& alpha,
    const std::vector<WallNodeInfo>& wallNodes,
    MeshDeformer& deformer,
    const std::vector<double>& xiVol,
    const std::vector<double>& etaVol,
    int nqVol,
    const std::vector<double>& zFace,
    int nqFace,
    double chordRef,
    double forceNx,
    double forceNy,
    double fdEpsilon,
    double baselineJ);

// Overload for L/D objective: J = Cl/Cd.
// AoA_rad is used to derive both lift and drag force directions internally.
std::vector<double> computeShapeGradientLoverD(
    GPUSolverData& gpu,
    DiscreteAdjointGPUData& da,
    Mesh2D& mesh,
    const std::vector<std::array<double, 2>>& baselineNodes,
    const GeomData2D& baselineGeom,
    const HicksHenneParam& hh,
    const std::vector<double>& alpha,
    const std::vector<WallNodeInfo>& wallNodes,
    MeshDeformer& deformer,
    const std::vector<double>& xiVol,
    const std::vector<double>& etaVol,
    int nqVol,
    const std::vector<double>& zFace,
    int nqFace,
    double chordRef,
    double AoA_rad,
    double fdEpsilon,
    double baselineJ);

// GPU dot product: sum_i a[i] * b[i], for vectors of length N on device.
double gpuDotProduct(const double* d_a, const double* d_b, int N);

#endif
