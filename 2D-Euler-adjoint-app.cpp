#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <map>

#include "src/Polylib.h"
#include "src/basis_poly.h"
#include "src/basis_functions.h"
#include "src/io.h"
#include "src/mesh2d.h"
#include "src/geom2d.h"
#include "src/euler2d.h"
#include "src/euler2d_gpu.h"
#include "src/adjoint2d_gpu.h"
#include "src/p_adapt.h"

using namespace polylib;

extern "C" {
    extern void dgetrf_(int*, int*, double*, int*, int[], int*);
    extern void dgetrs_(unsigned char*, int*, int*, double*, int*, int[], double[], int*, int*);
}

static double g_rhoInf, g_uInf, g_vInf, g_pInf;

// ============================================================================
// Visualization basis evaluation (shared with forward driver)
// ============================================================================

static void evalModalBasis1D(int P, const std::vector<double>& zpts,
                             std::vector<std::vector<double>>& Bvis)
{
    int npts = (int)zpts.size();
    Bvis.resize(P + 1, std::vector<double>(npts));
    for (int p = 0; p <= P; ++p)
        for (int i = 0; i < npts; ++i) {
            double z = zpts[i];
            double val;
            polylib::jacobfd(1, &z, &val, NULL, p, 0.0, 0.0);
            Bvis[p][i] = val;
        }
}

static void evalNodalBasis1D(int P, const std::string& ptype,
                             const std::vector<double>& zn_in,
                             const std::vector<double>& zpts,
                             std::vector<std::vector<double>>& Bvis)
{
    std::vector<double> zn(zn_in);
    int npts = (int)zpts.size();
    Bvis.resize(P + 1, std::vector<double>(npts));
    for (int p = 0; p <= P; ++p)
        for (int i = 0; i < npts; ++i) {
            if (ptype == "GaussLegendre")
                Bvis[p][i] = polylib::hgj(p, zpts[i], zn.data(), P + 1, 0.0, 0.0);
            else
                Bvis[p][i] = polylib::hglj(p, zpts[i], zn.data(), P + 1, 0.0, 0.0);
        }
}

// ============================================================================
// Per-element error indicator (L2 norm of adjoint)
// ============================================================================

static std::vector<double> computeErrorIndicator(const double* psi_flat,
                                                  int nE, int nqVol, int nq1d,
                                                  int totalDOF,
                                                  const std::vector<double>& wq,
                                                  const GeomData2D& geom)
{
    std::vector<double> eta(nE, 0.0);
    for (int e = 0; e < nE; ++e) {
        double val = 0.0;
        for (int v = 0; v < NVAR2D; ++v)
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    int idx = e * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[idx];
                    double p = psi_flat[v * totalDOF + idx];
                    val += w * p * p;
                }
        eta[e] = std::sqrt(val);
    }
    return eta;
}

// ============================================================================
// VTK output for adjoint solution
// ============================================================================

static void writeAdjointVTK(const std::string& filename,
                            const Mesh2D& mesh,
                            const GeomData2D& geom,
                            const std::vector<std::vector<double>>& psi,
                            const std::vector<std::vector<double>>& Bmat,
                            const std::vector<std::vector<double>>& Bvis,
                            const std::vector<double>& Minv,
                            const std::vector<double>& wq,
                            const std::vector<double>& zVis,
                            int P, int nq1d,
                            const std::vector<double>& eta = {})
{
    int nE = mesh.nElements;
    int nqVol = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);
    int nVis = (int)zVis.size();
    int nVisVol = nVis * nVis;
    int totalPts = nE * nVisVol;

    std::vector<double> xVis(totalPts), yVis(totalPts);
    std::vector<std::vector<double>> PsiVis(NVAR2D, std::vector<double>(totalPts));

    for (int e = 0; e < nE; ++e) {
        std::vector<double> coeff(NVAR2D * nmodes, 0.0);
        for (int v = 0; v < NVAR2D; ++v) {
            std::vector<double> proj(nmodes, 0.0);
            for (int i = 0; i <= P; ++i)
                for (int j = 0; j <= P; ++j) {
                    int m = i * (P + 1) + j;
                    for (int qx = 0; qx < nq1d; ++qx)
                        for (int qe = 0; qe < nq1d; ++qe) {
                            int qIdx = e * nqVol + qx * nq1d + qe;
                            double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                            proj[m] += w * Bmat[i][qx] * Bmat[j][qe] * psi[v][qIdx];
                        }
                }
            for (int m = 0; m < nmodes; ++m) {
                double val = 0.0;
                for (int mp = 0; mp < nmodes; ++mp)
                    val += Minv[e * nmodes * nmodes + m * nmodes + mp] * proj[mp];
                coeff[v * nmodes + m] = val;
            }
        }

        for (int ix = 0; ix < nVis; ++ix)
            for (int iy = 0; iy < nVis; ++iy) {
                int idx = e * nVisVol + ix * nVis + iy;
                refToPhys(mesh, e, zVis[ix], zVis[iy], xVis[idx], yVis[idx]);
                for (int v = 0; v < NVAR2D; ++v) {
                    double val = 0.0;
                    for (int i = 0; i <= P; ++i)
                        for (int j = 0; j <= P; ++j)
                            val += coeff[v * nmodes + i*(P+1)+j]
                                 * Bvis[i][ix] * Bvis[j][iy];
                    PsiVis[v][idx] = val;
                }
            }
    }

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler Adjoint Solution\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << totalPts << " double\n";
    for (int i = 0; i < totalPts; ++i)
        out << xVis[i] << " " << yVis[i] << " 0.0\n";

    int nSub = nVis - 1;
    int nCells = nE * nSub * nSub;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    for (int e = 0; e < nE; ++e) {
        int base = e * nVisVol;
        for (int ix = 0; ix < nSub; ++ix)
            for (int iy = 0; iy < nSub; ++iy) {
                int p0 = base + ix * nVis + iy;
                int p1 = base + (ix+1) * nVis + iy;
                int p2 = base + (ix+1) * nVis + (iy+1);
                int p3 = base + ix * nVis + (iy+1);
                out << "4 " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
    }

    out << "CELL_TYPES " << nCells << "\n";
    for (int i = 0; i < nCells; ++i)
        out << "9\n";

    out << "POINT_DATA " << totalPts << "\n";
    const char* varNames[4] = {"lambda_rho", "lambda_rhou", "lambda_rhov", "lambda_rhoE"};
    for (int v = 0; v < NVAR2D; ++v) {
        out << "SCALARS " << varNames[v] << " double 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < totalPts; ++i)
            out << PsiVis[v][i] << "\n";
    }

    if (!eta.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        out << "SCALARS ErrorIndicator double 1\nLOOKUP_TABLE default\n";
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy)
                    out << eta[e] << "\n";
    }

    out.close();
}

// ============================================================================
// VTK output for adjoint at actual solution (quadrature) points
// ============================================================================

static void writeAdjointVTK_solpts(const std::string& filename,
                                   const Mesh2D& mesh,
                                   const GeomData2D& geom,
                                   const std::vector<std::vector<double>>& psi,
                                   const std::vector<double>& zq,
                                   int nq1d,
                                   const std::string& ptype,
                                   const std::vector<double>& eta = {})
{
    int nE    = mesh.nElements;
    int nqVol = nq1d * nq1d;

    bool needAug = (ptype == "GaussLegendre");
    int nAug    = needAug ? nq1d + 2 : nq1d;
    int nAugVol = nAug * nAug;

    std::vector<double> zAug(nAug);
    std::vector<double> Interp(nAug * nq1d, 0.0);

    if (needAug) {
        zAug[0] = -1.0;
        for (int k = 0; k < nq1d; ++k) zAug[k + 1] = zq[k];
        zAug[nAug - 1] = 1.0;

        for (int k = 0; k < nq1d; ++k)
            Interp[(k + 1) * nq1d + k] = 1.0;

        std::vector<double> zq_mut(zq);
        for (int k = 0; k < nq1d; ++k) {
            Interp[0 * nq1d + k]          = polylib::hgj(k, -1.0, zq_mut.data(), nq1d, 0.0, 0.0);
            Interp[(nAug - 1) * nq1d + k] = polylib::hgj(k,  1.0, zq_mut.data(), nq1d, 0.0, 0.0);
        }
    } else {
        for (int k = 0; k < nq1d; ++k) zAug[k] = zq[k];
        for (int k = 0; k < nq1d; ++k) Interp[k * nq1d + k] = 1.0;
    }

    int totalPts = nE * nAugVol;

    std::vector<double> xPts(totalPts), yPts(totalPts);
    std::vector<std::vector<double>> PsiAug(NVAR2D, std::vector<double>(totalPts));

    for (int e = 0; e < nE; ++e)
        for (int i = 0; i < nAug; ++i)
            for (int j = 0; j < nAug; ++j) {
                int idx = e * nAugVol + i * nAug + j;
                refToPhys(mesh, e, zAug[i], zAug[j], xPts[idx], yPts[idx]);

                for (int v = 0; v < NVAR2D; ++v) {
                    double val = 0.0;
                    for (int a = 0; a < nq1d; ++a)
                        for (int b = 0; b < nq1d; ++b)
                            val += Interp[i * nq1d + a] * Interp[j * nq1d + b]
                                 * psi[v][e * nqVol + a * nq1d + b];
                    PsiAug[v][idx] = val;
                }
            }

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler Adjoint at quadrature points\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << totalPts << " double\n";
    for (int i = 0; i < totalPts; ++i)
        out << xPts[i] << " " << yPts[i] << " 0.0\n";

    int nSub   = nAug - 1;
    int nCells = nE * nSub * nSub;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    for (int e = 0; e < nE; ++e) {
        int base = e * nAugVol;
        for (int ix = 0; ix < nSub; ++ix)
            for (int iy = 0; iy < nSub; ++iy) {
                int p0 = base + ix       * nAug + iy;
                int p1 = base + (ix + 1) * nAug + iy;
                int p2 = base + (ix + 1) * nAug + (iy + 1);
                int p3 = base + ix       * nAug + (iy + 1);
                out << "4 " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
    }

    out << "CELL_TYPES " << nCells << "\n";
    for (int i = 0; i < nCells; ++i)
        out << "9\n";

    out << "POINT_DATA " << totalPts << "\n";
    const char* varNames[4] = {"lambda_rho", "lambda_rhou", "lambda_rhov", "lambda_rhoE"};
    for (int v = 0; v < NVAR2D; ++v) {
        out << "SCALARS " << varNames[v] << " double 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < totalPts; ++i)
            out << PsiAug[v][i] << "\n";
    }

    if (!eta.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        out << "SCALARS ErrorIndicator double 1\nLOOKUP_TABLE default\n";
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy)
                    out << eta[e] << "\n";
    }

    out.close();
}

// ============================================================================
// Restart I/O (forward solution)
// ============================================================================

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
    std::cout << "Restart loaded from " << filename << " (time = " << time << ")" << std::endl;
    return true;
}

// ============================================================================
// Adjoint restart I/O
// ============================================================================

static void writeAdjointRestart(const std::string& filename,
                                const double* psi_flat, int solSize,
                                int nE, int nqVol, int iter)
{
    std::ofstream out(filename, std::ios::binary);
    int nvar = NVAR2D;
    out.write(reinterpret_cast<const char*>(&nvar), sizeof(int));
    out.write(reinterpret_cast<const char*>(&nE), sizeof(int));
    out.write(reinterpret_cast<const char*>(&nqVol), sizeof(int));
    out.write(reinterpret_cast<const char*>(&iter), sizeof(int));
    out.write(reinterpret_cast<const char*>(psi_flat), solSize * sizeof(double));
    out.close();
    std::cout << "Adjoint restart written to " << filename
              << " (iter = " << iter << ")" << std::endl;
}

static bool readAdjointRestart(const std::string& filename,
                               double* psi_flat, int solSize,
                               int nE, int nqVol, int& iter)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Error: cannot open adjoint restart file " << filename << std::endl;
        return false;
    }
    int nvar, nE_file, nqVol_file;
    in.read(reinterpret_cast<char*>(&nvar), sizeof(int));
    in.read(reinterpret_cast<char*>(&nE_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&nqVol_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&iter), sizeof(int));
    if (nvar != NVAR2D || nE_file != nE || nqVol_file != nqVol) {
        std::cout << "Error: adjoint restart file mismatch (nvar=" << nvar
                  << " nE=" << nE_file << " nqVol=" << nqVol_file
                  << "), expected (" << NVAR2D << ", " << nE << ", " << nqVol << ")" << std::endl;
        return false;
    }
    in.read(reinterpret_cast<char*>(psi_flat), solSize * sizeof(double));
    in.close();
    std::cout << "Adjoint restart loaded from " << filename
              << " (iter = " << iter << ")" << std::endl;
    return true;
}

static void printProgressBar(int current, int total, int width = 50)
{
    float progress = static_cast<float>(current) / total;
    int barWidth = static_cast<int>(width * progress);
    fprintf(stderr, "\r[");
    for (int i = 0; i < width; ++i) {
        if (i < barWidth)       fprintf(stderr, "=");
        else if (i == barWidth) fprintf(stderr, ">");
        else                    fprintf(stderr, " ");
    }
    fprintf(stderr, "] %3d%%", static_cast<int>(progress * 100.0));
    fflush(stderr);
}

// ============================================================================
// Variable-P adjoint solver
// ============================================================================

static int runVariablePAdjoint(Inputs2D* inp, Mesh2D& mesh)
{
    int nE = mesh.nElements, nF = mesh.nFaces;
    int pMin = inp->pMin, pMax = inp->pMax;
    int nq1d = pMax + 2;
    int nqVol = nq1d * nq1d;
    int totalDOF = nE * nqVol;
    int P = pMax, P1 = P + 1;
    int nmodes = P1 * P1;

    // Read P distribution
    std::vector<int> elemP;
    if (!inp->errorIndicatorFile.empty()) {
        std::vector<double> eta = readErrorIndicator(inp->errorIndicatorFile, nE);
        elemP = assignElementP(eta, pMin, pMax);
    } else {
        elemP.assign(nE, pMax);
        std::cout << "No error indicator file; using P=" << pMax << " everywhere" << std::endl;
    }
    auto groups = buildPGroups(elemP, pMin, pMax);

    // Quadrature and geometry at common nq1d
    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    polylib::zwgl(zq.data(), wq.data(), nq1d);
    std::vector<double> xiVol(nqVol), etaVol(nqVol);
    for (int i = 0; i < nq1d; ++i)
        for (int j = 0; j < nq1d; ++j) {
            xiVol[i * nq1d + j] = zq[i];
            etaVol[i * nq1d + j] = zq[j];
        }
    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

    // Freestream
    double AoA_rad = inp->AoA * M_PI / 180.0;
    g_rhoInf = 1.0;
    g_pInf = 1.0 / (GAMMA * inp->Mach * inp->Mach);
    double cInfVal = std::sqrt(GAMMA * g_pInf / g_rhoInf);
    g_uInf = inp->Mach * cInfVal * std::cos(AoA_rad);
    g_vInf = inp->Mach * cInfVal * std::sin(AoA_rad);

    // Build pMax basis for GPU static data (Nodal for the forward GPU)
    auto gpuBasisMax = BasisPoly::Create("Nodal", pMax, inp->ptype, zq, wq);
    gpuBasisMax->ConstructBasis();
    auto BmatMax = gpuBasisMax->GetB();
    auto DmatMax = gpuBasisMax->GetD();
    auto blrMax = gpuBasisMax->GetLeftRightBasisValues();

    // Per-element Minv: each element uses its own P's basis, stored in a
    // nmodes_max * nmodes_max block (zero-padded for low-P elements).
    std::vector<double> Minv(nE * nmodes * nmodes, 0.0);
    for (auto& [p, ginfo] : groups) {
        int gP1 = p + 1, gnm = gP1 * gP1;
        std::vector<double> zq_c(zq), wq_c(wq);
        auto basis_g = BasisPoly::Create("Nodal", p, inp->ptype, zq_c, wq_c);
        basis_g->ConstructBasis();
        auto Bg = basis_g->GetB();
        for (int eIdx = 0; eIdx < (int)ginfo.globalElemIdx.size(); ++eIdx) {
            int eG = ginfo.globalElemIdx[eIdx];
            std::vector<double> Mlocal(gnm * gnm, 0.0);
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    int qIdx = eG * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                    for (int i1 = 0; i1 < gP1; ++i1)
                        for (int j1 = 0; j1 < gP1; ++j1) {
                            int row = i1 * gP1 + j1;
                            double phiR = Bg[i1][qx] * Bg[j1][qe];
                            for (int i2 = 0; i2 < gP1; ++i2)
                                for (int j2 = 0; j2 < gP1; ++j2)
                                    Mlocal[row * gnm + i2 * gP1 + j2] += w * phiR * Bg[i2][qx] * Bg[j2][qe];
                        }
                }
            int INFO;
            std::vector<int> piv(gnm);
            dgetrf_(&gnm, &gnm, Mlocal.data(), &gnm, piv.data(), &INFO);
            std::vector<double> MinvLocal(gnm * gnm, 0.0);
            for (int i = 0; i < gnm; ++i) MinvLocal[i * gnm + i] = 1.0;
            unsigned char TR = 'N'; int NR = gnm;
            dgetrs_(&TR, &gnm, &NR, Mlocal.data(), &gnm, piv.data(), MinvLocal.data(), &gnm, &INFO);
            // Store in global Minv (zero-padded to nmodes * nmodes)
            for (int r = 0; r < gnm; ++r)
                for (int c = 0; c < gnm; ++c)
                    Minv[eG * nmodes * nmodes + r * nmodes + c] = MinvLocal[r * gnm + c];
        }
    }

    // Load forward solution
    std::vector<std::vector<double>> U(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) U[v].resize(totalDOF, 0.0);
    double time_restart = 0.0;
    if (!inp->restartfile.empty()) {
        int iterDummy = 0;
        std::ifstream in(inp->restartfile, std::ios::binary);
        int nvar, nE_f, nqVol_f;
        in.read(reinterpret_cast<char*>(&nvar), sizeof(int));
        in.read(reinterpret_cast<char*>(&nE_f), sizeof(int));
        in.read(reinterpret_cast<char*>(&nqVol_f), sizeof(int));
        in.read(reinterpret_cast<char*>(&time_restart), sizeof(double));
        if (nvar != NVAR2D || nE_f != nE || nqVol_f != nqVol) {
            std::cout << "Restart file mismatch" << std::endl;
            return 1;
        }
        for (int v = 0; v < NVAR2D; ++v) {
            U[v].resize(totalDOF);
            in.read(reinterpret_cast<char*>(U[v].data()), totalDOF * sizeof(double));
        }
        in.close();
        std::cout << "Forward restart loaded (time = " << time_restart << ")" << std::endl;
    } else {
        std::cout << "Error: adjoint solver requires a restart file" << std::endl;
        return 1;
    }

    // Set up GPU
    GPUSolverData gpu;
    gpuAllocate(gpu, nE, nF, pMax, nq1d);
    gpu.rhoInf = g_rhoInf; gpu.uInf = g_uInf;
    gpu.vInf = g_vInf; gpu.pInf = g_pInf;
    gpu.fluxType = (inp->fluxtype == "HLLC") ? 1 : 0;
    gpu.useAV = inp->useAV;
    gpu.AVkappa = inp->AVkappa;
    gpu.AVs0 = (inp->AVs0 != 0.0) ? inp->AVs0
               : -(4.25 * std::log10((double)std::max(pMax, 1)) + 0.5);
    gpu.AVscale = inp->AVscale;

    // Flatten and upload
    std::vector<double> Bmat_flat(P1 * nq1d), Dmat_flat(P1 * nq1d), blr_flat(P1 * 2);
    for (int i = 0; i < P1; ++i) {
        for (int q = 0; q < nq1d; ++q) {
            Bmat_flat[i * nq1d + q] = BmatMax[i][q];
            Dmat_flat[i * nq1d + q] = DmatMax[i][q];
        }
        blr_flat[i * 2] = blrMax[i][0];
        blr_flat[i * 2 + 1] = blrMax[i][1];
    }
    std::vector<int> e2f(nE * 4), feL(nF), feR(nF), ffL(nF), ffR(nF), fbc(nF, 0);
    for (int e = 0; e < nE; ++e)
        for (int lf = 0; lf < 4; ++lf) e2f[e * 4 + lf] = mesh.elem2face[e][lf];
    for (int f = 0; f < nF; ++f) {
        feL[f] = mesh.faces[f].elemL; feR[f] = mesh.faces[f].elemR;
        ffL[f] = mesh.faces[f].faceL; ffR[f] = mesh.faces[f].faceR;
        if (mesh.faces[f].elemR < 0) {
            int tag = mesh.faces[f].bcTag;
            if (inp->testcase == "NACA0012") {
                if (tag == 1) fbc[f] = 1; else fbc[f] = 2;
            }
        }
    }
    gpuCopyStaticData(gpu,
        geom.detJ.data(), geom.dxidx.data(), geom.dxidy.data(),
        geom.detadx.data(), geom.detady.data(),
        geom.faceNx.data(), geom.faceNy.data(), geom.faceJac.data(),
        geom.faceXPhys.data(), geom.faceYPhys.data(),
        Bmat_flat.data(), Dmat_flat.data(), blr_flat.data(),
        Minv.data(), wq.data(),
        e2f.data(), feL.data(), feR.data(), ffL.data(), ffR.data(), fbc.data());

    std::vector<double> U_flat(NVAR2D * totalDOF);
    for (int v = 0; v < NVAR2D; ++v)
        std::memcpy(&U_flat[v * totalDOF], U[v].data(), totalDOF * sizeof(double));
    gpuCopySolutionToDevice(gpu, U_flat.data());

    // Face interpolation weights
    {
        std::vector<double> interpL(nq1d), interpR(nq1d);
        std::vector<double> zq_m(zq);
        for (int k = 0; k < nq1d; ++k) {
            interpL[k] = polylib::hgj(k, -1.0, zq_m.data(), nq1d, 0.0, 0.0);
            interpR[k] = polylib::hgj(k, 1.0, zq_m.data(), nq1d, 0.0, 0.0);
        }
        gpuSetFaceInterp(interpL.data(), interpR.data(), nq1d);
    }

    // NodalToModal (Nodal basis, V^{-1})
    {
        std::vector<double> zn = gpuBasisMax->GetZn();
        std::vector<double> T(P1 * P1, 0.0);
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
            double di = 1.0 / aug[col * 2*P1 + col];
            for (int c = 0; c < 2 * P1; ++c) aug[col * 2*P1 + c] *= di;
            for (int r = 0; r < P1; ++r) {
                if (r == col) continue;
                double f = aug[r * 2*P1 + col];
                for (int c = 0; c < 2 * P1; ++c) aug[r * 2*P1 + c] -= f * aug[col * 2*P1 + c];
            }
        }
        for (int r = 0; r < P1; ++r)
            for (int c = 0; c < P1; ++c)
                T[r * P1 + c] = aug[r * 2*P1 + P1 + c];
        gpuSetNodalToModal(T.data(), P1);
        adjointGpuSetNodalToModal(T.data(), P1);
    }

    // Freeze forward state: one RHS per P-group to populate Ucoeff
    // Use pMax for the global forward RHS (all elements get pMax basis)
    gpuComputeDGRHS(gpu, false, 0.0);
    gpuSyncUcoeff(gpu);
    std::cout << "Forward state frozen." << std::endl;

    // Set up per-group adjoint basis data
    std::map<int, AdjointPGroup> adjGroups;
    for (auto& [p, ginfo] : groups) {
        int nEG = (int)ginfo.globalElemIdx.size();
        AdjointPGroup& ag = adjGroups[p];
        adjointAllocatePGroup(ag, p, nEG);
        adjointUploadPGroupElemIdx(ag, ginfo.globalElemIdx.data());

        std::vector<double> zq_c(zq), wq_c(wq);
        auto basis = BasisPoly::Create("Nodal", p, inp->ptype, zq_c, wq_c);
        basis->ConstructBasis();
        auto Bg = basis->GetB();
        auto Dg = basis->GetD();
        auto blrg = basis->GetLeftRightBasisValues();
        int gP1 = p + 1;

        ag.h_Bmat.resize(gP1 * nq1d);
        ag.h_Dmat.resize(gP1 * nq1d);
        ag.h_blr.resize(gP1 * 2);
        for (int i = 0; i < gP1; ++i) {
            for (int q = 0; q < nq1d; ++q) {
                ag.h_Bmat[i * nq1d + q] = Bg[i][q];
                ag.h_Dmat[i * nq1d + q] = Dg[i][q];
            }
            ag.h_blr[i * 2] = blrg[i][0];
            ag.h_blr[i * 2 + 1] = blrg[i][1];
        }
        ag.h_wq.assign(wq.begin(), wq.end());

        std::vector<double> zn = basis->GetZn();
        ag.h_NodalToModal.resize(gP1 * gP1);
        std::vector<double> Vg(gP1 * gP1);
        for (int n = 0; n < gP1; ++n)
            for (int pp = 0; pp < gP1; ++pp) {
                double val;
                polylib::jacobfd(1, &zn[n], &val, NULL, pp, 0.0, 0.0);
                Vg[n * gP1 + pp] = val;
            }
        std::vector<double> augG(gP1 * 2 * gP1);
        for (int r = 0; r < gP1; ++r)
            for (int c = 0; c < gP1; ++c) {
                augG[r * 2 * gP1 + c] = Vg[r * gP1 + c];
                augG[r * 2 * gP1 + gP1 + c] = (r == c) ? 1.0 : 0.0;
            }
        for (int col = 0; col < gP1; ++col) {
            int pivot = col;
            for (int r = col + 1; r < gP1; ++r)
                if (std::fabs(augG[r * 2*gP1 + col]) > std::fabs(augG[pivot * 2*gP1 + col]))
                    pivot = r;
            if (pivot != col)
                for (int c = 0; c < 2 * gP1; ++c)
                    std::swap(augG[col * 2*gP1 + c], augG[pivot * 2*gP1 + c]);
            double di = 1.0 / augG[col * 2*gP1 + col];
            for (int c = 0; c < 2 * gP1; ++c) augG[col * 2*gP1 + c] *= di;
            for (int r = 0; r < gP1; ++r) {
                if (r == col) continue;
                double f = augG[r * 2*gP1 + col];
                for (int c = 0; c < 2 * gP1; ++c) augG[r * 2*gP1 + c] -= f * augG[col * 2*gP1 + c];
            }
        }
        for (int r = 0; r < gP1; ++r)
            for (int c = 0; c < gP1; ++c)
                ag.h_NodalToModal[r * gP1 + c] = augG[r * 2*gP1 + gP1 + c];

        std::cout << "  Adjoint P-group P=" << p << ": " << nEG << " elements" << std::endl;
    }

    // Adjoint setup
    double chordRef = inp->adjChordRef;
    std::string adjObjective = inp->adjObjective;
    double forceNx, forceNy;
    if (adjObjective == "Drag") {
        forceNx = -std::cos(AoA_rad); forceNy = -std::sin(AoA_rad);
    } else {
        forceNx = std::sin(AoA_rad); forceNy = -std::cos(AoA_rad);
    }

    AdjointGPUData adj;
    adjointGpuAllocate(adj, gpu);
    adj.chordRef = chordRef;
    adj.fullAV = (inp->btype == "Modal");
    adjointGpuSetBasisData(Bmat_flat.data(), Dmat_flat.data(),
                           blr_flat.data(), wq.data(), P1, nq1d);
    adjointComputeFrozenAlpha(adj, gpu);

    double objVal = adjointComputeForceCoeff(adj, gpu, chordRef, forceNx, forceNy);
    std::cout << "Objective = " << adjObjective << " = " << objVal << std::endl;
    adjointComputeObjectiveGradient(adj, gpu, chordRef, forceNx, forceNy);

    int solSize = NVAR_GPU * adj.primaryDOF;
    int adjMaxIter = inp->adjMaxIter;
    double adjTol = inp->adjTol;
    double CFL = inp->CFL;
    double dt = inp->dt;

    std::cout << "Variable-P adjoint: " << nE << " elements, PMin=" << pMin
              << ", PMax=" << pMax << ", maxIter=" << adjMaxIter << std::endl;

    int iterStart = 0;
    if (!inp->adjRestartFile.empty()) {
        std::vector<double> psi_restart(solSize);
        if (!readAdjointRestart(inp->adjRestartFile, psi_restart.data(),
                                solSize, nE, nqVol, iterStart)) {
            std::cout << "Adjoint restart failed, exiting." << std::endl;
            for (auto& [p, ag] : adjGroups) adjointFreePGroup(ag);
            adjointGpuFree(adj);
            gpuFree(gpu);
            return 1;
        }
        adjointCopySolutionToDevice(adj, psi_restart.data(), solSize);
        std::cout << "Adjoint restarting from iteration " << iterStart << std::endl;
    }

    // Adjoint iteration
    auto tStart = std::chrono::high_resolution_clock::now();
    bool converged = false, nanFound = false;

    auto computeAdjRHS = [&](bool usePsiTmp) {
        adjointZeroCoeffArrays(adj, gpu);
        for (auto& [p, ag] : adjGroups)
            adjointComputeRHS_group(adj, gpu, ag, usePsiTmp);
        adjointAddObjectiveGradient(adj, gpu);
    };

    std::vector<double> psi_chk(solSize);

    std::ofstream csvFile;
    if (iterStart > 0) {
        csvFile.open("adjoint_residual_history.csv", std::ios::app);
    } else {
        csvFile.open("adjoint_residual_history.csv");
        csvFile << "iter,lambda_rho,lambda_rhou,lambda_rhov,lambda_rhoE\n";
    }

    for (int iter = iterStart; iter < adjMaxIter && !converged && !nanFound; ++iter) {
        if (CFL > 0.0) dt = gpuComputeCFL(gpu, CFL, pMax);

        computeAdjRHS(false);
        adjointRK4Stage(adj, dt, 1, solSize);
        computeAdjRHS(true);
        adjointRK4Stage(adj, dt, 2, solSize);
        computeAdjRHS(true);
        adjointRK4Stage(adj, dt, 3, solSize);
        computeAdjRHS(true);
        adjointRK4Stage(adj, dt, 4, solSize);

        nanFound = adjointCheckNaN(adj, solSize);
        if (nanFound) {
            std::cout << "\nNaN detected in adjoint at iteration " << iter << std::endl;
            break;
        }

        if ((iter + 1) % 100 == 0 || iter < 5) {
            double perVarNorms[4];
            adjointResidualNormPerVar(adj, adj.primaryDOF, perVarNorms);
            double resL2 = 0.0;
            for (int v = 0; v < 4; ++v) resL2 += perVarNorms[v] * perVarNorms[v];
            resL2 = std::sqrt(resL2);

            csvFile << (iter + 1) << "," << perVarNorms[0] << "," << perVarNorms[1]
                    << "," << perVarNorms[2] << "," << perVarNorms[3] << "\n";
            csvFile.flush();

            std::cout << "Adjoint iter " << std::setw(6) << (iter+1)
                      << "  dt=" << std::scientific << std::setprecision(4) << dt
                      << "  |res|=" << resL2 << std::endl;

            if (resL2 < adjTol) {
                converged = true;
                std::cout << "Adjoint converged at iteration " << (iter+1) << std::endl;
            }
        }

        if (inp->checkpoint > 0 && (iter + 1) % inp->checkpoint == 0 && !nanFound) {
            std::vector<double> psi_quad_chk(NVAR_GPU * totalDOF);
            adjointCopyQuadPointsToHost(adj, gpu, psi_quad_chk.data());

            std::vector<std::vector<double>> psi_vtk(NVAR2D);
            for (int v = 0; v < NVAR2D; ++v) {
                psi_vtk[v].resize(totalDOF);
                std::memcpy(psi_vtk[v].data(), &psi_quad_chk[v * totalDOF],
                            totalDOF * sizeof(double));
            }

            std::vector<double> eta_chk = computeErrorIndicator(
                psi_quad_chk.data(), nE, nqVol, nq1d, totalDOF, wq, geom);

            std::string chkVtk = "adjoint_checkpoint_" + std::to_string(iter + 1) + ".vtk";
            writeAdjointVTK_solpts(chkVtk, mesh, geom, psi_vtk, zq, nq1d, inp->ptype, eta_chk);

            std::vector<double> psi_raw(solSize);
            adjointCopySolutionToHost(adj, psi_raw.data(), solSize);
            std::string chkBin = "adjoint_restart_" + std::to_string(iter + 1) + ".bin";
            writeAdjointRestart(chkBin, psi_raw.data(), solSize, nE, nqVol, iter + 1);

            std::cout << "\nAdjoint checkpoint at iteration " << (iter + 1) << std::endl;
        }

        printProgressBar(iter + 1, adjMaxIter);
    }
    csvFile.close();

    auto tEnd = std::chrono::high_resolution_clock::now();
    std::cout << "\nAdjoint wall-clock time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count()
              << " ms" << std::endl;

    // Write adjoint output at quad points
    std::vector<double> psi_flat(NVAR_GPU * totalDOF);
    adjointCopyQuadPointsToHost(adj, gpu, psi_flat.data());

    std::vector<std::vector<double>> psi(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) {
        psi[v].resize(totalDOF);
        std::memcpy(psi[v].data(), &psi_flat[v * totalDOF], totalDOF * sizeof(double));
    }

    writeAdjointVTK_solpts("adjoint2d_final_solpts.vtk", mesh, geom, psi, zq, nq1d, inp->ptype);
    std::cout << "Adjoint output written to adjoint2d_final_solpts.vtk" << std::endl;

    // Error indicator
    std::vector<double> eta = computeErrorIndicator(
        psi_flat.data(), nE, nqVol, nq1d, totalDOF, wq, geom);

    if (!nanFound) {
        std::vector<double> psi_raw(solSize);
        adjointCopySolutionToHost(adj, psi_raw.data(), solSize);
        writeAdjointRestart("adjoint_restart.bin", psi_raw.data(), solSize, nE, nqVol, adjMaxIter);
    }

    {
        double eta_max = *std::max_element(eta.begin(), eta.end());
        double eta_sum = 0.0;
        for (auto& v : eta) eta_sum += v;
        std::cout << "\nSensitivity: max(eta)=" << eta_max << "  sum(eta)=" << eta_sum << std::endl;

        std::ofstream efile("error_indicator.dat");
        efile << "# element   eta\n";
        for (int e = 0; e < nE; ++e)
            efile << e << " " << eta[e] << "\n";
        efile.close();
        std::cout << "Sensitivity indicators written to error_indicator.dat" << std::endl;
    }

    for (auto& [p, ag] : adjGroups) adjointFreePGroup(ag);
    adjointGpuFree(adj);
    gpuFree(gpu);
    return 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    setvbuf(stderr, NULL, _IONBF, 0);

    std::string inputFile = "inputs2d.xml";
    if (argc > 1) inputFile = argv[1];

    Inputs2D* inp = ReadXmlFile2D(inputFile.c_str());
    if (!inp) return 1;

    int    P    = inp->porder;
    int    nq1d = inp->nquad;
    double CFL  = inp->CFL;
    int    adjMaxIter = inp->adjMaxIter;
    double adjTol     = inp->adjTol;
    double chordRef   = inp->adjChordRef;
    std::string adjObjective = inp->adjObjective;

    if (inp->restartfile.empty()) {
        std::cout << "Error: adjoint solver requires a converged restart file (RestartFile)." << std::endl;
        return 1;
    }

    // ========================================================================
    // Read mesh
    // ========================================================================
    Mesh2D mesh;
    mesh.readGmsh(inp->meshfile);
    std::cout << "Mesh: " << mesh.nElements << " elements, "
              << mesh.nFaces << " faces" << std::endl;

    // Variable-P mode
    if (inp->pMin > 0 && inp->pMax > 0) {
        int ret = runVariablePAdjoint(inp, mesh);
        delete inp;
        return ret;
    }

    // ========================================================================
    // Set up quadrature and basis
    // ========================================================================
    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    if (inp->ptype == "GaussLegendre") {
        zwgl(zq.data(), wq.data(), nq1d);
    } else {
        zwgll(zq.data(), wq.data(), nq1d);
    }

    std::unique_ptr<BasisPoly> basis1D = BasisPoly::Create(inp->btype, P, inp->ptype, zq, wq);
    basis1D->ConstructBasis();

    int nqVol  = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);

    std::vector<double> xiVol(nqVol), etaVol(nqVol);
    for (int i = 0; i < nq1d; ++i)
        for (int j = 0; j < nq1d; ++j) {
            xiVol[i * nq1d + j]  = zq[i];
            etaVol[i * nq1d + j] = zq[j];
        }

    // ========================================================================
    // Compute geometry
    // ========================================================================
    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

    // ========================================================================
    // Mass matrices
    // ========================================================================
    bool modalMode = (inp->btype == "Modal");
    BasisPoly* gpuBasis = basis1D.get();

    std::vector<std::vector<double>> Bmat = gpuBasis->GetB();
    std::vector<std::vector<double>> Dmat = gpuBasis->GetD();
    std::vector<std::vector<double>> blr  = gpuBasis->GetLeftRightBasisValues();

    std::vector<double> massLU;
    std::vector<int>    massPiv;
    assembleAndFactorMassMatrices(mesh, geom, Bmat, Bmat, wq, wq,
                                 P, nq1d, massLU, massPiv);

    std::vector<double> Minv;
    computeMassInverse(massLU, massPiv, mesh.nElements, nmodes, Minv);

    // ========================================================================
    // Visualization basis
    // ========================================================================
    int nVis = std::max(nq1d, 15);
    std::vector<double> zVis(nVis);
    for (int i = 0; i < nVis; ++i)
        zVis[i] = -1.0 + 2.0 * i / (nVis - 1);

    std::vector<std::vector<double>> Bvis;
    {
        std::vector<double> zn = gpuBasis->GetZn();
        evalNodalBasis1D(P, inp->ptype, zn, zVis, Bvis);
    }

    // ========================================================================
    // Load converged forward solution
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

    // ========================================================================
    // Force direction for lift/drag objective
    // ========================================================================
    double AoA_rad = inp->AoA * M_PI / 180.0;
    double forceNx, forceNy;
    if (adjObjective == "Drag") {
        forceNx = -std::cos(AoA_rad);
        forceNy = -std::sin(AoA_rad);
    } else {
        forceNx =  std::sin(AoA_rad);
        forceNy = -std::cos(AoA_rad);
    }
    std::cout << "Adjoint objective  = " << adjObjective << std::endl;
    std::cout << "Force direction    = (" << forceNx << ", " << forceNy << ")" << std::endl;

    // ========================================================================
    // Flatten data for GPU
    // ========================================================================
    std::vector<double> Bmat_flat((P+1) * nq1d);
    std::vector<double> Dmat_flat((P+1) * nq1d);
    std::vector<double> blr_flat((P+1) * 2);
    for (int i = 0; i < P + 1; ++i) {
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
    // Initialize forward GPU solver (for frozen state)
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

    // In modal mode, project quad-point restart to modal coefficients
    if (modalMode) {
        std::vector<double> U_coeff(NVAR2D * nE * nmodes, 0.0);
        for (int e = 0; e < nE; ++e)
            for (int var = 0; var < NVAR2D; ++var) {
                std::vector<double> proj(nmodes, 0.0);
                for (int i = 0; i < P + 1; ++i)
                    for (int j = 0; j < P + 1; ++j) {
                        int m = i * (P + 1) + j;
                        for (int qx = 0; qx < nq1d; ++qx)
                            for (int qe = 0; qe < nq1d; ++qe) {
                                int qIdx = e * nqVol + qx * nq1d + qe;
                                double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                                proj[m] += w * Bmat[i][qx] * Bmat[j][qe]
                                         * U[var][qIdx];
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
        std::cout << "IC projected to " << nE * nmodes << " modal coefficients" << std::endl;
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
        int P1 = P + 1;
        std::vector<double> T(P1 * P1, 0.0);
        if (modalMode) {
            for (int i = 0; i < P1; ++i)
                T[i * P1 + i] = 1.0;
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
        adjointGpuSetNodalToModal(T.data(), P1);
    }

    // Run one forward RHS to populate Ucoeff and epsilon
    gpuComputeDGRHS(gpu, false, 0.0);
    gpuSyncUcoeff(gpu);

    std::cout << "Forward solution loaded and Ucoeff/epsilon frozen." << std::endl;

    // ========================================================================
    // Initialize adjoint solver
    // ========================================================================
    AdjointGPUData adj;
    adjointGpuAllocate(adj, gpu);
    adj.chordRef = chordRef;
    adj.fullAV = (inp->btype == "Modal");

    adjointGpuSetBasisData(Bmat_flat.data(), Dmat_flat.data(),
                           blr_flat.data(), wq.data(), P + 1, nq1d);

    adjointComputeFrozenAlpha(adj, gpu);

    double objVal = adjointComputeForceCoeff(adj, gpu, chordRef, forceNx, forceNy);
    std::cout << std::scientific << std::setprecision(10);
    if (adjObjective == "Drag")
        std::cout << "Drag coefficient (Cd) = " << objVal << std::endl;
    else
        std::cout << "Lift coefficient (Cl) = " << objVal << std::endl;

    adjointComputeObjectiveGradient(adj, gpu, chordRef, forceNx, forceNy);
    std::cout << "Objective gradient (dJ/dU) computed." << std::endl;

    int solSize = NVAR_GPU * adj.primaryDOF;

    // ========================================================================
    // Load adjoint restart if specified
    // ========================================================================
    int iterStart = 0;
    if (!inp->adjRestartFile.empty()) {
        std::vector<double> psi_restart(solSize);
        if (!readAdjointRestart(inp->adjRestartFile, psi_restart.data(),
                                solSize, nE, nqVol, iterStart)) {
            std::cout << "Adjoint restart failed, exiting." << std::endl;
            adjointGpuFree(adj);
            gpuFree(gpu);
            delete inp;
            return 1;
        }
        adjointCopySolutionToDevice(adj, psi_restart.data(), solSize);
        std::cout << "Adjoint restarting from iteration " << iterStart << std::endl;
    }

    std::cout << "Adjoint solver initialized: " << nE << " elements, P=" << P
              << ", nq1d=" << nq1d << ", totalDOF=" << totalDOF << std::endl;
    std::cout << "Max adjoint iterations = " << adjMaxIter << std::endl;
    std::cout << "Adjoint tolerance      = " << adjTol << std::endl;

    // ========================================================================
    // Adjoint RK4 pseudo-time iteration
    // ========================================================================
    auto tStart = std::chrono::high_resolution_clock::now();
    bool converged = false;
    bool nanFound = false;
    double dt = 0.0;

    std::vector<double> psi_chk(solSize);

    std::ofstream adjCsvFile;
    if (iterStart > 0) {
        adjCsvFile.open("adjoint_residual_history.csv", std::ios::app);
    } else {
        adjCsvFile.open("adjoint_residual_history.csv");
        adjCsvFile << "iter,lambda_rho,lambda_rhou,lambda_rhov,lambda_rhoE\n";
    }

    for (int iter = iterStart; iter < adjMaxIter && !converged && !nanFound; ++iter)
    {
        if (CFL > 0.0)
            dt = gpuComputeCFL(gpu, CFL, P);

        adjointComputeRHS(adj, gpu, false);
        adjointRK4Stage(adj, dt, 1, solSize);

        adjointComputeRHS(adj, gpu, true);
        adjointRK4Stage(adj, dt, 2, solSize);

        adjointComputeRHS(adj, gpu, true);
        adjointRK4Stage(adj, dt, 3, solSize);

        adjointComputeRHS(adj, gpu, true);
        adjointRK4Stage(adj, dt, 4, solSize);

        nanFound = adjointCheckNaN(adj, solSize);
        if (nanFound) {
            std::cout << "\nNaN detected in adjoint at iteration " << iter << std::endl;
            break;
        }

        if ((iter + 1) % 100 == 0 || iter < 5) {
            double perVarNorms[4];
            adjointResidualNormPerVar(adj, adj.primaryDOF, perVarNorms);
            double resL2 = 0.0;
            for (int v = 0; v < 4; ++v) resL2 += perVarNorms[v] * perVarNorms[v];
            resL2 = std::sqrt(resL2);

            adjCsvFile << (iter + 1)
                       << "," << perVarNorms[0]
                       << "," << perVarNorms[1]
                       << "," << perVarNorms[2]
                       << "," << perVarNorms[3] << "\n";
            adjCsvFile.flush();

            std::cout << "Adjoint iter " << std::setw(6) << (iter+1)
                      << "  dt=" << std::scientific << std::setprecision(4) << dt
                      << "  |res|=" << resL2 << std::endl;

            if (resL2 < adjTol) {
                converged = true;
                std::cout << "Adjoint converged at iteration " << (iter+1) << std::endl;
            }
        }

        if (inp->checkpoint > 0 && (iter + 1) % inp->checkpoint == 0 && !nanFound) {
            std::vector<double> psi_quad_chk(NVAR_GPU * totalDOF);
            adjointCopyQuadPointsToHost(adj, gpu, psi_quad_chk.data());

            std::vector<std::vector<double>> psi_vtk(NVAR2D);
            for (int v = 0; v < NVAR2D; ++v) {
                psi_vtk[v].resize(totalDOF);
                std::memcpy(psi_vtk[v].data(), &psi_quad_chk[v * totalDOF],
                            totalDOF * sizeof(double));
            }

            std::vector<double> eta_chk = computeErrorIndicator(
                psi_quad_chk.data(), nE, nqVol, nq1d, totalDOF, wq, geom);

            std::string chkVtk = "adjoint_checkpoint_" + std::to_string(iter + 1) + ".vtk";
            writeAdjointVTK(chkVtk, mesh, geom, psi_vtk, Bmat, Bvis,
                            Minv, wq, zVis, P, nq1d, eta_chk);

            adjointCopySolutionToHost(adj, psi_chk.data(), solSize);
            std::string chkBin = "adjoint_restart_" + std::to_string(iter + 1) + ".bin";
            writeAdjointRestart(chkBin, psi_chk.data(), solSize, nE, nqVol, iter + 1);

            std::cout << "\nAdjoint checkpoint at iteration " << (iter + 1) << std::endl;
        }

        printProgressBar(iter + 1, adjMaxIter);
    }

    adjCsvFile.close();

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    std::cout << "\nAdjoint wall-clock time: " << elapsed.count() << " ms" << std::endl;

    // ========================================================================
    // Copy adjoint solution back and write VTK
    // ========================================================================
    std::vector<double> psi_flat(NVAR_GPU * totalDOF);
    adjointCopyQuadPointsToHost(adj, gpu, psi_flat.data());

    std::vector<std::vector<double>> psi(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) {
        psi[v].resize(totalDOF);
        std::memcpy(psi[v].data(), &psi_flat[v * totalDOF], totalDOF * sizeof(double));
    }

    std::vector<double> eta = computeErrorIndicator(
        psi_flat.data(), nE, nqVol, nq1d, totalDOF, wq, geom);

    writeAdjointVTK("adjoint2d_final.vtk", mesh, geom, psi, Bmat, Bvis,
                    Minv, wq, zVis, P, nq1d, eta);
    writeAdjointVTK_solpts("adjoint2d_final_solpts.vtk", mesh, geom, psi, zq, nq1d, inp->ptype, eta);
    std::cout << "Adjoint solution written to adjoint2d_final.vtk" << std::endl;
    std::cout << "Adjoint solution-point output written to adjoint2d_final_solpts.vtk" << std::endl;

    if (!nanFound) {
        std::vector<double> psi_raw(solSize);
        adjointCopySolutionToHost(adj, psi_raw.data(), solSize);
        writeAdjointRestart("adjoint_restart.bin", psi_raw.data(), solSize, nE, nqVol, adjMaxIter);
    }

    // ========================================================================
    // Finite-difference gradient check (optional, on a few DOFs)
    // ========================================================================
    if (inp->adjFDCheck) {
        std::cout << "\n=== Finite-difference gradient check ===" << std::endl;
        const double fd_eps = 1e-5;
        const int nCheck = std::min(20, totalDOF);

        for (int idx = 0; idx < nCheck; ++idx) {
            int dof = (idx * 137) % totalDOF;
            int var = 0;

            double psi_val = psi_flat[var * totalDOF + dof];

            std::vector<double> U_pert(U_flat);
            U_pert[var * totalDOF + dof] += fd_eps;
            gpuCopySolutionToDevice(gpu, U_pert.data());
            gpuComputeDGRHS(gpu, false, 0.0);
            gpuSyncUcoeff(gpu);
            double Jp = adjointComputeForceCoeff(adj, gpu, chordRef, forceNx, forceNy);

            U_pert[var * totalDOF + dof] -= 2.0 * fd_eps;
            gpuCopySolutionToDevice(gpu, U_pert.data());
            gpuComputeDGRHS(gpu, false, 0.0);
            gpuSyncUcoeff(gpu);
            double Jm = adjointComputeForceCoeff(adj, gpu, chordRef, forceNx, forceNy);

            double fd_grad = (Jp - Jm) / (2.0 * fd_eps);

            std::cout << "  DOF " << std::setw(6) << dof
                      << "  adjoint=" << std::setw(14) << psi_val
                      << "  FD=" << std::setw(14) << fd_grad
                      << "  ratio=" << std::setw(10)
                      << ((std::fabs(fd_grad) > 1e-30) ? psi_val/fd_grad : 0.0)
                      << std::endl;
        }

        gpuCopySolutionToDevice(gpu, U_flat.data());
        gpuComputeDGRHS(gpu, false, 0.0);
        gpuSyncUcoeff(gpu);
    }

    // ========================================================================
    // Write sensitivity indicator text file (reuses eta computed above)
    // ========================================================================
    {
        double eta_max = *std::max_element(eta.begin(), eta.end());
        double eta_sum = 0.0;
        for (auto& v : eta) eta_sum += v;
        std::cout << "\nSensitivity: max(eta)=" << eta_max << "  sum(eta)=" << eta_sum << std::endl;

        std::ofstream efile("error_indicator.dat");
        efile << "# element   eta\n";
        for (int e = 0; e < nE; ++e)
            efile << e << " " << eta[e] << "\n";
        efile.close();
        std::cout << "Sensitivity indicators written to error_indicator.dat" << std::endl;
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    adjointGpuFree(adj);
    gpuFree(gpu);
    delete inp;
    return 0;
}
