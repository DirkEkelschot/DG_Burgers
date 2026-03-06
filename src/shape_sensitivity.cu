#include "shape_sensitivity.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
}

// Reduction kernel for dot product
__global__ void dotProductKernel(const double* a, const double* b, double* out, int N)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    double val = 0.0;
    while (i < N) {
        val += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

double gpuDotProduct(const double* d_a, const double* d_b, int N)
{
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

    int blockSize = 256;
    int gridSize = std::min((N + blockSize - 1) / blockSize, 1024);
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(
        d_a, d_b, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}

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
    double baselineJ)
{
    int nDesign = hh.nDesignVars();
    std::vector<double> gradient(nDesign, 0.0);

    // Store baseline residual on GPU (d_R after computing RHS at baseline)
    // Compute baseline residual
    gpuUpdateGeometry(gpu,
        baselineGeom.detJ.data(), baselineGeom.dxidx.data(), baselineGeom.dxidy.data(),
        baselineGeom.detadx.data(), baselineGeom.detady.data(),
        baselineGeom.faceNx.data(), baselineGeom.faceNy.data(), baselineGeom.faceJac.data(),
        baselineGeom.faceXPhys.data(), baselineGeom.faceYPhys.data());

    gpuComputeDGRHS(gpu, false, 0.0);

    // Copy baseline residual to host
    int solSize = NVAR_GPU * gpu.primaryDOF;
    std::vector<double> R_baseline(solSize);
    if (gpu.modalMode) {
        CUDA_CHECK(cudaMemcpy(R_baseline.data(), gpu.d_rhsCoeff,
                              solSize * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(R_baseline.data(), gpu.d_R,
                              solSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Copy adjoint to host
    std::vector<double> psi_host(solSize);
    discreteAdjointCopySolutionToHost(da, psi_host.data(), solSize);

    std::cout << "Computing shape gradient (" << nDesign << " design variables)..." << std::endl;

    for (int k = 0; k < nDesign; ++k) {
        // Perturb alpha_k
        std::vector<double> alpha_pert(alpha);
        alpha_pert[k] += fdEpsilon;

        // Deform mesh with perturbed alpha
        deformMesh(mesh, hh, alpha_pert, wallNodes, deformer, baselineNodes);

        // Recompute geometry
        GeomData2D geom_pert = computeGeometry(mesh, xiVol, etaVol, nqVol, zFace, nqFace);

        // Upload perturbed geometry to GPU
        gpuUpdateGeometry(gpu,
            geom_pert.detJ.data(), geom_pert.dxidx.data(), geom_pert.dxidy.data(),
            geom_pert.detadx.data(), geom_pert.detady.data(),
            geom_pert.faceNx.data(), geom_pert.faceNy.data(), geom_pert.faceJac.data(),
            geom_pert.faceXPhys.data(), geom_pert.faceYPhys.data());

        // Compute perturbed objective
        double J_pert = discreteAdjointComputeForceCoeff(da, gpu, chordRef, forceNx, forceNy);

        // Compute perturbed residual
        gpuComputeDGRHS(gpu, false, 0.0);

        std::vector<double> R_pert(solSize);
        if (gpu.modalMode) {
            CUDA_CHECK(cudaMemcpy(R_pert.data(), gpu.d_rhsCoeff,
                                  solSize * sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(R_pert.data(), gpu.d_R,
                                  solSize * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // dJ/dalpha_k = (J_pert - J_base)/eps + psi^T * (R_pert - R_base)/eps
        double dJdirect = (J_pert - baselineJ) / fdEpsilon;

        double psiTdR = 0.0;
        for (int i = 0; i < solSize; ++i)
            psiTdR += psi_host[i] * (R_pert[i] - R_baseline[i]);
        psiTdR /= fdEpsilon;

        gradient[k] = dJdirect + psiTdR;

        std::cout << "  alpha_" << k << ": dJ_direct=" << dJdirect
                  << "  psi^T*dR=" << psiTdR
                  << "  total=" << gradient[k] << std::endl;
    }

    // Restore baseline geometry
    mesh.nodes = baselineNodes;
    gpuUpdateGeometry(gpu,
        baselineGeom.detJ.data(), baselineGeom.dxidx.data(), baselineGeom.dxidy.data(),
        baselineGeom.detadx.data(), baselineGeom.detady.data(),
        baselineGeom.faceNx.data(), baselineGeom.faceNy.data(), baselineGeom.faceJac.data(),
        baselineGeom.faceXPhys.data(), baselineGeom.faceYPhys.data());

    return gradient;
}
