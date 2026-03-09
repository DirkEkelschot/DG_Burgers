#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <set>

#include "src/Polylib.h"
#include "src/basis_poly.h"
#include "src/basis_functions.h"
#include "src/io.h"
#include "src/mesh2d.h"
#include "src/geom2d.h"
#include "src/euler2d.h"
#include "src/euler2d_gpu.h"
#include "src/discrete_adjoint2d_gpu.h"
#include "src/hicks_henne.h"
#include "src/mesh_deform.h"
#include "src/shape_sensitivity.h"

using namespace polylib;

extern "C" {
    extern void dgetrf_(int*, int*, double*, int*, int[], int*);
    extern void dgetrs_(unsigned char*, int*, int*, double*, int*, int[], double[], int*, int*);
}

static double g_rhoInf, g_uInf, g_vInf, g_pInf;

static std::vector<double> readAlphaFile(const std::string& filename)
{
    std::vector<double> alpha;
    std::ifstream in(filename);
    if (!in.is_open()) return alpha;
    double val;
    while (in >> val) alpha.push_back(val);
    return alpha;
}

static bool readRestart(const std::string& filename,
                        std::vector<std::vector<double>>& U,
                        int nE, int nqVol, double& time)
{
    int totalDOF = nE * nqVol;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Error: cannot open restart file " << filename << std::endl;
        return false;
    }
    int nvar, nE_file, nqVol_file;
    in.read(reinterpret_cast<char*>(&nvar), sizeof(int));
    in.read(reinterpret_cast<char*>(&nE_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&nqVol_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&time), sizeof(double));
    if (nvar != NVAR2D || nE_file != nE || nqVol_file != nqVol) {
        std::cout << "Error: restart file mismatch" << std::endl;
        return false;
    }
    for (int v = 0; v < NVAR2D; ++v) {
        U[v].resize(totalDOF);
        in.read(reinterpret_cast<char*>(U[v].data()), totalDOF * sizeof(double));
    }
    in.close();
    return true;
}

static int readAdjointRestart(const std::string& filename,
                              double* psi_flat, int solSize,
                              int nE, int nqVol, int& iter)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) return 0;
    int nvar, nE_file, nqVol_file;
    in.read(reinterpret_cast<char*>(&nvar), sizeof(int));
    in.read(reinterpret_cast<char*>(&nE_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&nqVol_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&iter), sizeof(int));
    if (nvar != NVAR2D || nE_file != nE || nqVol_file != nqVol) return -1;
    in.read(reinterpret_cast<char*>(psi_flat), solSize * sizeof(double));
    in.close();
    return 1;
}

int main(int argc, char* argv[])
{
    std::string inputFile = "inputs2d.xml";
    std::string alphaFile = "alpha.dat";
    std::string gradientFile = "gradient.dat";
    std::string objectiveFile = "objective.dat";
    bool fdCheck = false;
    int fdCheckVar = -1;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--alpha" && i + 1 < argc)
            alphaFile = argv[++i];
        else if (std::string(argv[i]) == "--gradient" && i + 1 < argc)
            gradientFile = argv[++i];
        else if (std::string(argv[i]) == "--objective" && i + 1 < argc)
            objectiveFile = argv[++i];
        else if (std::string(argv[i]) == "--fd-check") {
            fdCheck = true;
            if (i + 1 < argc && argv[i+1][0] != '-')
                fdCheckVar = std::stoi(argv[++i]);
        }
        else if (argv[i][0] != '-')
            inputFile = argv[i];
    }

    Inputs2D* inp = ReadXmlFile2D(inputFile.c_str());
    if (!inp) return 1;

    int P    = inp->porder;
    int nq1d = inp->nquad;
    double chordRef   = inp->adjChordRef;
    std::string adjObjective = inp->adjObjective;
    double fdEpsilon = inp->optFDEpsilon;

    if (inp->restartfile.empty()) {
        std::cout << "Error: requires a converged restart file (RestartFile)." << std::endl;
        return 1;
    }

    // ========================================================================
    // Load mesh (this is the DEFORMED mesh from deform_mesh)
    // ========================================================================
    Mesh2D mesh;
    mesh.readGmsh(inp->meshfile);

    // ========================================================================
    // Setup Hicks-Henne and read design variables
    // ========================================================================
    HicksHenneParam hh;
    hh.setupUniformBumps(inp->optNBumpsUpper, inp->optNBumpsLower);
    int nDesign = hh.nDesignVars();

    std::vector<double> alpha = readAlphaFile(alphaFile);
    if (static_cast<int>(alpha.size()) != nDesign) {
        std::cout << "Warning: alpha file size mismatch, using zeros." << std::endl;
        alpha.assign(nDesign, 0.0);
    }

    // We need the BASELINE (undeformed) mesh to perform FD perturbations.
    // Read the baseline mesh from the original mesh file specified as BaseMeshFile,
    // or derive it by applying alpha=0.
    // For now, we reload the mesh and treat it as the baseline that gets deformed.
    Mesh2D baseMesh;
    if (!inp->baseMeshFile.empty()) {
        baseMesh.readGmsh(inp->baseMeshFile);
    } else {
        std::cout << "Error: BaseMeshFile must be specified for gradient computation." << std::endl;
        delete inp;
        return 1;
    }

    auto baselineNodes = baseMesh.nodes;

    // Classify wall nodes on baseline mesh
    std::vector<int> wallFaceIndices;
    for (int f = 0; f < baseMesh.nFaces; ++f) {
        if (baseMesh.faces[f].elemR < 0 && baseMesh.faces[f].bcTag == 1)
            wallFaceIndices.push_back(f);
    }
    auto wallNodes = classifyWallNodes(baseMesh, wallFaceIndices);

    // Setup RBF on baseline mesh
    std::set<int> boundaryNodeIDs;
    for (int f = 0; f < baseMesh.nFaces; ++f) {
        if (baseMesh.faces[f].elemR < 0) {
            boundaryNodeIDs.insert(baseMesh.faces[f].nodeIDs[0]);
            boundaryNodeIDs.insert(baseMesh.faces[f].nodeIDs[1]);
            if (baseMesh.geomOrder == 2) {
                int eL = baseMesh.faces[f].elemL;
                int lf = baseMesh.faces[f].faceL;
                boundaryNodeIDs.insert(baseMesh.elements[eL][4 + lf]);
            }
        }
    }
    MeshDeformer deformer;
    deformer.setup(baseMesh, boundaryNodeIDs);

    // ========================================================================
    // Quadrature and basis
    // ========================================================================
    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    if (inp->ptype == "GaussLegendre") {
        zwgl(zq.data(), wq.data(), nq1d);
    } else {
        zwgll(zq.data(), wq.data(), nq1d);
    }

    bool modalMode = (inp->btype == "Modal");
    std::unique_ptr<BasisPoly> basis1D = BasisPoly::Create(inp->btype, P, inp->ptype, zq, wq);
    basis1D->ConstructBasis();

    int nqVol  = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);
    int P1     = P + 1;

    std::vector<double> xiVol(nqVol), etaVol(nqVol);
    for (int i = 0; i < nq1d; ++i)
        for (int j = 0; j < nq1d; ++j) {
            xiVol[i * nq1d + j]  = zq[i];
            etaVol[i * nq1d + j] = zq[j];
        }

    // Compute geometry on the DEFORMED mesh (current state)
    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

    BasisPoly* gpuBasis = basis1D.get();
    auto Bmat = gpuBasis->GetB();
    auto Dmat = gpuBasis->GetD();
    auto blr  = gpuBasis->GetLeftRightBasisValues();

    std::vector<double> massLU;
    std::vector<int> massPiv;
    assembleAndFactorMassMatrices(mesh, geom, Bmat, Bmat, wq, wq,
                                 P, nq1d, massLU, massPiv);
    std::vector<double> Minv;
    computeMassInverse(massLU, massPiv, mesh.nElements, nmodes, Minv);

    // ========================================================================
    // Load forward solution
    // ========================================================================
    int nE = mesh.nElements;
    int totalDOF = nE * nqVol;

    std::vector<std::vector<double>> U(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) U[v].resize(totalDOF, 0.0);

    {
        double AoA_rad = inp->AoA * M_PI / 180.0;
        g_rhoInf = 1.0;
        g_pInf   = 1.0 / (GAMMA * inp->Mach * inp->Mach);
        double cInf = std::sqrt(GAMMA * g_pInf / g_rhoInf);
        g_uInf = inp->Mach * cInf * std::cos(AoA_rad);
        g_vInf = inp->Mach * cInf * std::sin(AoA_rad);
    }

    double time_restart = 0.0;
    if (!readRestart(inp->restartfile, U, nE, nqVol, time_restart))
        return 1;

    // Force direction
    double AoA_rad = inp->AoA * M_PI / 180.0;
    double forceNx = 0.0, forceNy = 0.0;
    if (adjObjective == "Drag") {
        forceNx =  std::cos(AoA_rad);
        forceNy =  std::sin(AoA_rad);
    } else if (adjObjective == "Lift") {
        forceNx = -std::sin(AoA_rad);
        forceNy =  std::cos(AoA_rad);
    }

    // ========================================================================
    // Flatten data for GPU
    // ========================================================================
    std::vector<double> Bmat_flat(P1 * nq1d), Dmat_flat(P1 * nq1d), blr_flat(P1 * 2);
    for (int i = 0; i < P1; ++i) {
        for (int q = 0; q < nq1d; ++q) {
            Bmat_flat[i * nq1d + q] = Bmat[i][q];
            Dmat_flat[i * nq1d + q] = Dmat[i][q];
        }
        blr_flat[i * 2 + 0] = blr[i][0];
        blr_flat[i * 2 + 1] = blr[i][1];
    }

    std::vector<int> elem2face_flat(nE * 4);
    for (int e = 0; e < nE; ++e)
        for (int lf = 0; lf < 4; ++lf)
            elem2face_flat[e * 4 + lf] = mesh.elem2face[e][lf];

    int nF = mesh.nFaces;
    std::vector<int> face_elemL(nF), face_elemR(nF);
    std::vector<int> face_faceL(nF), face_faceR(nF);
    std::vector<int> face_bcType(nF, 0);
    for (int f = 0; f < nF; ++f) {
        face_elemL[f] = mesh.faces[f].elemL;
        face_elemR[f] = mesh.faces[f].elemR;
        face_faceL[f] = mesh.faces[f].faceL;
        face_faceR[f] = mesh.faces[f].faceR;
        if (mesh.faces[f].elemR < 0) {
            int tag = mesh.faces[f].bcTag;
            if (inp->testcase == "NACA0012") {
                if (tag == 1) face_bcType[f] = 1;
                else          face_bcType[f] = 2;
            }
        }
    }

    std::vector<double> U_flat(NVAR2D * totalDOF);
    for (int v = 0; v < NVAR2D; ++v)
        std::memcpy(&U_flat[v * totalDOF], U[v].data(), totalDOF * sizeof(double));

    // ========================================================================
    // Initialize GPU solver (frozen state)
    // ========================================================================
    GPUSolverData gpu;
    gpuAllocate(gpu, nE, nF, P, nq1d, modalMode);
    gpu.rhoInf = g_rhoInf; gpu.uInf = g_uInf;
    gpu.vInf   = g_vInf;   gpu.pInf = g_pInf;
    gpu.fluxType = (inp->fluxtype == "HLLC") ? 1 : 0;
    gpu.useAV    = inp->useAV;
    gpu.AVkappa  = inp->AVkappa;
    gpu.AVs0     = (inp->AVs0 != 0.0) ? inp->AVs0
                   : -(4.25 * std::log10((double)std::max(P, 1)) + 0.5);
    gpu.AVscale  = inp->AVscale;

    gpuCopyStaticData(gpu,
        geom.detJ.data(), geom.dxidx.data(), geom.dxidy.data(),
        geom.detadx.data(), geom.detady.data(),
        geom.faceNx.data(), geom.faceNy.data(), geom.faceJac.data(),
        geom.faceXPhys.data(), geom.faceYPhys.data(),
        Bmat_flat.data(), Dmat_flat.data(), blr_flat.data(),
        Minv.data(), wq.data(),
        elem2face_flat.data(),
        face_elemL.data(), face_elemR.data(),
        face_faceL.data(), face_faceR.data(),
        face_bcType.data());

    if (modalMode) {
        std::vector<double> U_coeff(NVAR2D * nE * nmodes, 0.0);
        for (int e = 0; e < nE; ++e)
            for (int var = 0; var < NVAR2D; ++var) {
                std::vector<double> proj(nmodes, 0.0);
                for (int i = 0; i < P1; ++i)
                    for (int j = 0; j < P1; ++j) {
                        int m = i * P1 + j;
                        for (int qx = 0; qx < nq1d; ++qx)
                            for (int qe = 0; qe < nq1d; ++qe) {
                                int qIdx = e * nqVol + qx * nq1d + qe;
                                double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                                proj[m] += w * Bmat[i][qx] * Bmat[j][qe] * U[var][qIdx];
                            }
                    }
                for (int m = 0; m < nmodes; ++m) {
                    double val = 0.0;
                    for (int mp = 0; mp < nmodes; ++mp)
                        val += Minv[e * nmodes * nmodes + m * nmodes + mp] * proj[mp];
                    U_coeff[var * nE * nmodes + e * nmodes + m] = val;
                }
            }
        gpuCopySolutionToDevice(gpu, U_coeff.data());
    } else {
        gpuCopySolutionToDevice(gpu, U_flat.data());
    }

    {
        std::vector<double> interpL(nq1d), interpR(nq1d);
        std::vector<double> zq_mut(zq);
        for (int k = 0; k < nq1d; ++k) {
            interpL[k] = polylib::hgj(k, -1.0, zq_mut.data(), nq1d, 0.0, 0.0);
            interpR[k] = polylib::hgj(k,  1.0, zq_mut.data(), nq1d, 0.0, 0.0);
        }
        gpuSetFaceInterp(interpL.data(), interpR.data(), nq1d);
    }

    {
        std::vector<double> T(P1 * P1, 0.0);
        if (modalMode) {
            for (int i = 0; i < P1; ++i) T[i * P1 + i] = 1.0;
        } else {
            std::vector<double> zn = gpuBasis->GetZn();
            std::vector<double> V(P1 * P1);
            for (int n = 0; n < P1; ++n)
                for (int p = 0; p < P1; ++p) {
                    double val;
                    polylib::jacobfd(1, &zn[n], &val, NULL, p, 0.0, 0.0);
                    V[n * P1 + p] = val;
                }
            std::vector<double> aug(P1 * 2 * P1);
            for (int r = 0; r < P1; ++r)
                for (int c = 0; c < P1; ++c) {
                    aug[r * 2 * P1 + c] = V[r * P1 + c];
                    aug[r * 2 * P1 + P1 + c] = (r == c) ? 1.0 : 0.0;
                }
            for (int col = 0; col < P1; ++col) {
                int pivot = col;
                for (int r = col + 1; r < P1; ++r)
                    if (std::fabs(aug[r * 2*P1 + col]) > std::fabs(aug[pivot * 2*P1 + col]))
                        pivot = r;
                if (pivot != col)
                    for (int c = 0; c < 2 * P1; ++c)
                        std::swap(aug[col * 2*P1 + c], aug[pivot * 2*P1 + c]);
                double diagInv = 1.0 / aug[col * 2*P1 + col];
                for (int c = 0; c < 2 * P1; ++c)
                    aug[col * 2*P1 + c] *= diagInv;
                for (int r = 0; r < P1; ++r) {
                    if (r == col) continue;
                    double fv = aug[r * 2*P1 + col];
                    for (int c = 0; c < 2 * P1; ++c)
                        aug[r * 2*P1 + c] -= fv * aug[col * 2*P1 + c];
                }
            }
            for (int r = 0; r < P1; ++r)
                for (int c = 0; c < P1; ++c)
                    T[r * P1 + c] = aug[r * 2*P1 + P1 + c];
        }
        gpuSetNodalToModal(T.data(), P1);
        discreteAdjointSetNodalToModal(T.data(), P1);
    }

    gpuComputeDGRHS(gpu, false, 0.0);
    gpuSyncUcoeff(gpu);

    // ========================================================================
    // Initialize discrete adjoint and load restart
    // ========================================================================
    DiscreteAdjointGPUData da;
    discreteAdjointAllocate(da, gpu);
    da.chordRef = chordRef;
    da.forceNx = forceNx;
    da.forceNy = forceNy;

    discreteAdjointSetBasisData(Bmat_flat.data(), Dmat_flat.data(),
                                blr_flat.data(), wq.data(), P1, nq1d);

    double baselineJ;
    std::cout << std::scientific << std::setprecision(10);
    if (adjObjective == "LiftOverDrag") {
        double Cl, Cd;
        baselineJ = discreteAdjointComputeLiftOverDrag(da, gpu, chordRef, AoA_rad, Cl, Cd);
        std::cout << "Objective (L/D) = " << baselineJ
                  << "  (Cl=" << Cl << ", Cd=" << Cd << ")" << std::endl;
    } else {
        baselineJ = discreteAdjointComputeForceCoeff(da, gpu, chordRef, forceNx, forceNy);
        std::cout << "Objective (" << adjObjective << ") = " << baselineJ << std::endl;
    }

    std::string adjRestartPath = inp->adjRestartFile.empty()
                               ? "discrete_adjoint_restart.bin"
                               : inp->adjRestartFile;
    int solSize = NVAR_GPU * da.primaryDOF;
    std::vector<double> psi_raw(solSize, 0.0);
    int adjIter;
    int rstatus = readAdjointRestart(adjRestartPath, psi_raw.data(), solSize, nE, nqVol, adjIter);
    if (rstatus <= 0) {
        std::cout << "Error: could not load adjoint restart." << std::endl;
        discreteAdjointFree(da);
        gpuFree(gpu);
        delete inp;
        return 1;
    }
    discreteAdjointCopySolutionToDevice(da, psi_raw.data(), solSize);
    std::cout << "Adjoint restart loaded (iter=" << adjIter << ")" << std::endl;

    // ========================================================================
    // Compute shape gradient
    // ========================================================================

    // We need to work with the baseline (undeformed) mesh for FD perturbations
    // The current 'mesh' is the deformed mesh. We use baseMesh for perturbations.
    // But the GPU solver is initialized with the deformed mesh's geometry.
    // The sensitivity routine will perturb around the current alpha and recompute.

    std::vector<double> gradient;
    if (adjObjective == "LiftOverDrag") {
        gradient = computeShapeGradientLoverD(
            gpu, da, baseMesh, baselineNodes, geom,
            hh, alpha, wallNodes, deformer,
            xiVol, etaVol, nqVol, zq, nq1d,
            chordRef, AoA_rad,
            fdEpsilon, baselineJ);
    } else {
        gradient = computeShapeGradient(
            gpu, da, baseMesh, baselineNodes, geom,
            hh, alpha, wallNodes, deformer,
            xiVol, etaVol, nqVol, zq, nq1d,
            chordRef, forceNx, forceNy,
            fdEpsilon, baselineJ);
    }

    // Write gradient
    {
        std::ofstream gout(gradientFile);
        gout << std::scientific << std::setprecision(15);
        for (int k = 0; k < nDesign; ++k)
            gout << gradient[k] << "\n";
        gout.close();
        std::cout << "Gradient written to " << gradientFile << std::endl;
    }

    // Write objective
    {
        std::ofstream oout(objectiveFile);
        oout << std::scientific << std::setprecision(15);
        oout << baselineJ << "\n";
        oout.close();
    }

    // Print gradient summary
    double gradNorm = 0.0;
    for (double g : gradient) gradNorm += g * g;
    gradNorm = std::sqrt(gradNorm);
    std::cout << "|gradient| = " << gradNorm << std::endl;

    // ========================================================================
    // FD gradient verification (--fd-check)
    // Perturb one design variable, run a full forward solve on the perturbed
    // mesh, and compare (J_pert - J_base)/eps with the adjoint gradient.
    // ========================================================================
    if (fdCheck) {
        std::cout << "\n=== FD Gradient Verification ===" << std::endl;

        int kStart = (fdCheckVar >= 0) ? fdCheckVar : 0;
        int kEnd   = (fdCheckVar >= 0) ? fdCheckVar + 1 : nDesign;

        for (int k = kStart; k < kEnd; ++k) {
            std::vector<double> alpha_pert(alpha);
            alpha_pert[k] += fdEpsilon;

            // Deform mesh with perturbed alpha
            deformMesh(baseMesh, hh, alpha_pert, wallNodes, deformer, baselineNodes);
            GeomData2D geom_pert = computeGeometry(baseMesh, xiVol, etaVol, nqVol, zq, nq1d);

            // Recompute mass matrix for perturbed mesh
            std::vector<double> massLU_pert;
            std::vector<int> massPiv_pert;
            assembleAndFactorMassMatrices(baseMesh, geom_pert, Bmat, Bmat, wq, wq,
                                         P, nq1d, massLU_pert, massPiv_pert);
            std::vector<double> Minv_pert;
            computeMassInverse(massLU_pert, massPiv_pert, nE, nmodes, Minv_pert);

            gpuUpdateGeometry(gpu,
                geom_pert.detJ.data(), geom_pert.dxidx.data(), geom_pert.dxidy.data(),
                geom_pert.detadx.data(), geom_pert.detady.data(),
                geom_pert.faceNx.data(), geom_pert.faceNy.data(), geom_pert.faceJac.data(),
                geom_pert.faceXPhys.data(), geom_pert.faceYPhys.data());

            gpuUpdateMinv(gpu, Minv_pert.data());

            // Re-initialize solution to freestream and run forward solve
            {
                std::vector<double> U_init(NVAR2D * totalDOF);
                for (int e2 = 0; e2 < nE; ++e2)
                    for (int q = 0; q < nqVol; ++q) {
                        int idx = e2 * nqVol + q;
                        U_init[0 * totalDOF + idx] = g_rhoInf;
                        U_init[1 * totalDOF + idx] = g_rhoInf * g_uInf;
                        U_init[2 * totalDOF + idx] = g_rhoInf * g_vInf;
                        U_init[3 * totalDOF + idx] = g_pInf / (GAMMA - 1.0)
                            + 0.5 * g_rhoInf * (g_uInf * g_uInf + g_vInf * g_vInf);
                    }
                gpuCopySolutionToDevice(gpu, U_init.data());
            }

            double CFL_val = inp->CFL;
            int nt_val = inp->nt;
            for (int iter = 0; iter < nt_val; ++iter) {
                double dt_loc = (CFL_val > 0.0) ? gpuComputeCFL(gpu, CFL_val, P) : inp->dt;
                gpuComputeDGRHS(gpu, false, 0.0);
                gpuRK4Stage(gpu, dt_loc, 1);
                gpuComputeDGRHS(gpu, true, 0.0);
                gpuRK4Stage(gpu, dt_loc, 2);
                gpuComputeDGRHS(gpu, true, 0.0);
                gpuRK4Stage(gpu, dt_loc, 3);
                gpuComputeDGRHS(gpu, true, 0.0);
                gpuRK4Stage(gpu, dt_loc, 4);
            }
            gpuComputeDGRHS(gpu, false, 0.0);
            gpuSyncUcoeff(gpu);

            double J_pert;
            if (adjObjective == "LiftOverDrag") {
                double Cl_p, Cd_p;
                J_pert = discreteAdjointComputeLiftOverDrag(da, gpu, chordRef, AoA_rad, Cl_p, Cd_p);
            } else {
                J_pert = discreteAdjointComputeForceCoeff(da, gpu, chordRef, forceNx, forceNy);
            }
            double fd_grad = (J_pert - baselineJ) / fdEpsilon;

            std::cout << "  alpha_" << k
                      << "  adjoint=" << std::setw(16) << gradient[k]
                      << "  FD=" << std::setw(16) << fd_grad
                      << "  ratio=" << std::setw(12)
                      << ((std::fabs(fd_grad) > 1e-30) ? gradient[k] / fd_grad : 0.0)
                      << std::endl;
        }

        // Restore baseline
        baseMesh.nodes = baselineNodes;
        gpuUpdateGeometry(gpu,
            geom.detJ.data(), geom.dxidx.data(), geom.dxidy.data(),
            geom.detadx.data(), geom.detady.data(),
            geom.faceNx.data(), geom.faceNy.data(), geom.faceJac.data(),
            geom.faceXPhys.data(), geom.faceYPhys.data());
        gpuCopySolutionToDevice(gpu, U_flat.data());
        gpuComputeDGRHS(gpu, false, 0.0);
        gpuSyncUcoeff(gpu);
    }

    discreteAdjointFree(da);
    gpuFree(gpu);
    delete inp;
    return 0;
}
