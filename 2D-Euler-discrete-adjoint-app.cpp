#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

#include "src/Polylib.h"
#include "src/basis_poly.h"
#include "src/basis_functions.h"
#include "src/io.h"
#include "src/mesh2d.h"
#include "src/geom2d.h"
#include "src/euler2d.h"
#include "src/euler2d_gpu.h"
#include "src/discrete_adjoint2d_gpu.h"

using namespace polylib;

extern "C" {
    extern void dgetrf_(int*, int*, double*, int*, int[], int*);
    extern void dgetrs_(unsigned char*, int*, int*, double*, int*, int[], double[], int*, int*);
}

static double g_rhoInf, g_uInf, g_vInf, g_pInf;

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
// Truncation error estimate: ||u_P - u_{P-1}||_L2
// ============================================================================

static std::vector<double> computeTruncationError(
    const std::vector<std::vector<double>>& U,
    int nE, int nqVol, int nq1d,
    const std::vector<double>& zq,
    const std::vector<double>& wq,
    const GeomData2D& geom,
    int P,
    std::vector<double>* diff_flat = nullptr)
{
    int P1 = P + 1;
    int nmodes = P1 * P1;
    int totalDOF = nE * nqVol;

    std::vector<double> zq_mut(zq);
    std::vector<std::vector<double>> Bleg(P1, std::vector<double>(nq1d));
    for (int n = 0; n < P1; ++n)
        polylib::jacobfd(nq1d, zq_mut.data(), Bleg[n].data(), NULL, n, 0.0, 0.0);

    if (diff_flat)
        diff_flat->assign(NVAR2D * totalDOF, 0.0);

    std::vector<double> tau(nE, 0.0);

    for (int e = 0; e < nE; ++e) {
        std::vector<double> Mloc(nmodes * nmodes, 0.0);
        for (int qx = 0; qx < nq1d; ++qx)
            for (int qe = 0; qe < nq1d; ++qe) {
                int qIdx = e * nqVol + qx * nq1d + qe;
                double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                for (int i1 = 0; i1 < P1; ++i1)
                    for (int j1 = 0; j1 < P1; ++j1) {
                        double phi1 = Bleg[i1][qx] * Bleg[j1][qe];
                        int m1 = i1 * P1 + j1;
                        for (int i2 = 0; i2 < P1; ++i2)
                            for (int j2 = 0; j2 < P1; ++j2)
                                Mloc[m1 * nmodes + i2 * P1 + j2]
                                    += w * phi1 * Bleg[i2][qx] * Bleg[j2][qe];
                    }
            }

        std::vector<double> MinvLoc(nmodes * nmodes, 0.0);
        for (int i = 0; i < nmodes; ++i) MinvLoc[i * nmodes + i] = 1.0;
        int INFO;
        std::vector<int> piv(nmodes);
        dgetrf_(&nmodes, &nmodes, Mloc.data(), &nmodes, piv.data(), &INFO);
        unsigned char TR = 'N';
        dgetrs_(&TR, &nmodes, &nmodes, Mloc.data(), &nmodes, piv.data(),
                MinvLoc.data(), &nmodes, &INFO);

        double val = 0.0;
        for (int v = 0; v < NVAR2D; ++v) {
            std::vector<double> proj(nmodes, 0.0);
            for (int i = 0; i < P1; ++i)
                for (int j = 0; j < P1; ++j) {
                    int m = i * P1 + j;
                    for (int qx = 0; qx < nq1d; ++qx)
                        for (int qe = 0; qe < nq1d; ++qe) {
                            int qIdx = e * nqVol + qx * nq1d + qe;
                            double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                            proj[m] += w * Bleg[i][qx] * Bleg[j][qe]
                                     * U[v][qIdx];
                        }
                }

            std::vector<double> coeff(nmodes, 0.0);
            for (int m = 0; m < nmodes; ++m) {
                double c = 0.0;
                for (int mp = 0; mp < nmodes; ++mp)
                    c += MinvLoc[m * nmodes + mp] * proj[mp];
                coeff[m] = c;
            }

            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    double diff = 0.0;
                    for (int i = 0; i < P1; ++i)
                        for (int j = 0; j < P1; ++j) {
                            if (i < P && j < P) continue;
                            diff += coeff[i * P1 + j]
                                  * Bleg[i][qx] * Bleg[j][qe];
                        }
                    int qIdx = e * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                    val += w * diff * diff;

                    if (diff_flat)
                        (*diff_flat)[v * totalDOF + qIdx] = diff;
                }
        }
        tau[e] = std::sqrt(val);
    }

    return tau;
}

// ============================================================================
// DWR error indicator
// ============================================================================

static std::vector<double> computeDWR(
    const double* psi_flat,
    const std::vector<double>& diff_flat,
    int nE, int nqVol, int nq1d,
    int totalDOF,
    const std::vector<double>& wq,
    const GeomData2D& geom)
{
    std::vector<double> dwr(nE, 0.0);
    for (int e = 0; e < nE; ++e) {
        double val = 0.0;
        for (int v = 0; v < NVAR2D; ++v)
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    int idx = e * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[idx];
                    val += w * psi_flat[v * totalDOF + idx]
                             * diff_flat[v * totalDOF + idx];
                }
        dwr[e] = std::fabs(val);
    }
    return dwr;
}

// ============================================================================
// VTK output at quadrature points
// ============================================================================

static void writeDiscreteAdjointVTK_solpts(const std::string& filename,
                                            const Mesh2D& mesh,
                                            const GeomData2D& geom,
                                            const std::vector<std::vector<double>>& psi,
                                            const std::vector<double>& zq,
                                            int nq1d,
                                            const std::string& ptype,
                                            const std::vector<double>& eta = {},
                                            const std::vector<double>& tau = {},
                                            const std::vector<double>& dwr = {},
                                            const std::vector<std::vector<double>>& U = {})
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
    bool hasU = !U.empty();
    std::vector<double> xPts(totalPts), yPts(totalPts);
    std::vector<std::vector<double>> PsiAug(NVAR2D, std::vector<double>(totalPts));
    std::vector<std::vector<double>> UAug;
    if (hasU)
        UAug.assign(NVAR2D, std::vector<double>(totalPts));

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
                    if (hasU) {
                        double uval = 0.0;
                        for (int a = 0; a < nq1d; ++a)
                            for (int b = 0; b < nq1d; ++b)
                                uval += Interp[i * nq1d + a] * Interp[j * nq1d + b]
                                      * U[v][e * nqVol + a * nq1d + b];
                        UAug[v][idx] = uval;
                    }
                }
            }

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler Discrete Adjoint at quadrature points\n";
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

    if (hasU) {
        const char* uNames[4] = {"rho", "rhou", "rhov", "rhoE"};
        for (int v = 0; v < NVAR2D; ++v) {
            out << "SCALARS " << uNames[v] << " double 1\nLOOKUP_TABLE default\n";
            for (int i = 0; i < totalPts; ++i)
                out << UAug[v][i] << "\n";
        }
    }

    if (!eta.empty() || !tau.empty() || !dwr.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        auto writeCellScalar = [&](const char* name, const std::vector<double>& v) {
            if (v.empty()) return;
            out << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
            for (int e = 0; e < nE; ++e)
                for (int ix = 0; ix < nSub; ++ix)
                    for (int iy = 0; iy < nSub; ++iy)
                        out << v[e] << "\n";
        };
        writeCellScalar("TruncationError", tau);
        writeCellScalar("ErrorIndicator", eta);
        writeCellScalar("DWR", dwr);
    }

    out.close();
}

// ============================================================================
// Restart I/O
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
    std::cout << "Discrete adjoint restart written to " << filename
              << " (iter = " << iter << ")" << std::endl;
}

static int readAdjointRestart(const std::string& filename,
                              double* psi_flat, int solSize,
                              int nE, int nqVol, int& iter)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Warning: restart file " << filename
                  << " not found, starting from scratch." << std::endl;
        return 0;
    }
    int nvar, nE_file, nqVol_file;
    in.read(reinterpret_cast<char*>(&nvar), sizeof(int));
    in.read(reinterpret_cast<char*>(&nE_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&nqVol_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&iter), sizeof(int));
    if (nvar != NVAR2D || nE_file != nE || nqVol_file != nqVol) {
        std::cout << "Error: restart mismatch" << std::endl;
        return -1;
    }
    in.read(reinterpret_cast<char*>(psi_flat), solSize * sizeof(double));
    in.close();
    std::cout << "Discrete adjoint restart loaded from " << filename
              << " (iter = " << iter << ")" << std::endl;
    return 1;
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
        std::cout << "Error: discrete adjoint requires a converged restart file (RestartFile)." << std::endl;
        return 1;
    }

    Mesh2D mesh;
    mesh.readGmsh(inp->meshfile);
    std::cout << "Mesh: " << mesh.nElements << " elements, "
              << mesh.nFaces << " faces" << std::endl;

    // ========================================================================
    // Quadrature and basis (modal mode for discrete adjoint)
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

    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

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
    // Truncation error
    // ========================================================================
    std::vector<double> trunc_diff;
    std::vector<double> tau = computeTruncationError(
        U, nE, nqVol, nq1d, zq, wq, geom, P, &trunc_diff);
    {
        double tau_max = *std::max_element(tau.begin(), tau.end());
        double tau_sum = 0.0;
        for (auto& v : tau) tau_sum += v;
        std::cout << "Truncation error (P-to-P-1): max=" << tau_max
                  << "  sum=" << tau_sum << std::endl;
    }

    // ========================================================================
    // Force direction
    // ========================================================================
    double AoA_rad = inp->AoA * M_PI / 180.0;
    double forceNx = 0.0, forceNy = 0.0;
    if (adjObjective == "Drag") {
        forceNx =  std::cos(AoA_rad);
        forceNy =  std::sin(AoA_rad);
    } else if (adjObjective == "Lift") {
        forceNx = -std::sin(AoA_rad);
        forceNy =  std::cos(AoA_rad);
    }
    std::cout << "Discrete adjoint objective = " << adjObjective << std::endl;
    if (adjObjective != "LiftOverDrag")
        std::cout << "Force direction            = (" << forceNx << ", " << forceNy << ")" << std::endl;

    // ========================================================================
    // Flatten data for GPU
    // ========================================================================
    std::vector<double> Bmat_flat(P1 * nq1d);
    std::vector<double> Dmat_flat(P1 * nq1d);
    std::vector<double> blr_flat(P1 * 2);
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
    // Initialize forward GPU solver (frozen state)
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
        discreteAdjointSetNodalToModal(T.data(), P1);
    }

    gpuComputeDGRHS(gpu, false, 0.0);
    gpuSyncUcoeff(gpu);
    std::cout << "Forward solution loaded and Ucoeff/epsilon frozen." << std::endl;

    // ========================================================================
    // Initialize discrete adjoint solver
    // ========================================================================
    DiscreteAdjointGPUData da;
    discreteAdjointAllocate(da, gpu);
    da.chordRef = chordRef;
    da.forceNx = forceNx;
    da.forceNy = forceNy;

    discreteAdjointSetBasisData(Bmat_flat.data(), Dmat_flat.data(),
                                blr_flat.data(), wq.data(), P1, nq1d);

    double objVal;
    std::cout << std::scientific << std::setprecision(10);
    if (adjObjective == "LiftOverDrag") {
        double Cl, Cd;
        objVal = discreteAdjointComputeLiftOverDrag(da, gpu, chordRef, AoA_rad, Cl, Cd);
        std::cout << "L/D = " << objVal
                  << "  (Cl=" << Cl << ", Cd=" << Cd << ")" << std::endl;
        discreteAdjointComputeLiftOverDragGradient(da, gpu, chordRef, AoA_rad);
    } else {
        objVal = discreteAdjointComputeForceCoeff(da, gpu, chordRef, forceNx, forceNy);
        if (adjObjective == "Drag")
            std::cout << "Drag coefficient (Cd) = " << objVal << std::endl;
        else
            std::cout << "Lift coefficient (Cl) = " << objVal << std::endl;
        discreteAdjointComputeObjectiveGradient(da, gpu, chordRef, forceNx, forceNy);
    }
    std::cout << "Objective gradient (dJ/dU) computed." << std::endl;

    int solSize = NVAR_GPU * da.primaryDOF;

    // ========================================================================
    // Load adjoint restart if specified
    // ========================================================================
    int iterStart = 0;
    if (!inp->adjRestartFile.empty()) {
        std::vector<double> psi_restart(solSize);
        int rstatus = readAdjointRestart(inp->adjRestartFile, psi_restart.data(),
                                         solSize, nE, nqVol, iterStart);
        if (rstatus < 0) {
            discreteAdjointFree(da);
            gpuFree(gpu);
            delete inp;
            return 1;
        }
        if (rstatus > 0) {
            discreteAdjointCopySolutionToDevice(da, psi_restart.data(), solSize);
            std::cout << "Restarting from iteration " << iterStart << std::endl;
        }
    }

    std::cout << "Discrete adjoint solver initialized: " << nE << " elements, P=" << P
              << ", nq1d=" << nq1d << ", totalDOF=" << totalDOF << std::endl;
    std::cout << "Max iterations = " << adjMaxIter << std::endl;
    std::cout << "Tolerance      = " << adjTol << std::endl;

    // ========================================================================
    // Discrete adjoint RK4 pseudo-time iteration
    // ========================================================================
    auto tStart = std::chrono::high_resolution_clock::now();
    bool converged = false;
    bool nanFound = false;
    double dt = 0.0;

    std::ofstream csvFile;
    if (iterStart > 0) {
        csvFile.open("discrete_adjoint_residual_history.csv", std::ios::app);
    } else {
        csvFile.open("discrete_adjoint_residual_history.csv");
        csvFile << "iter,lambda_rho,lambda_rhou,lambda_rhov,lambda_rhoE\n";
    }

    for (int iter = iterStart; iter < adjMaxIter && !converged && !nanFound; ++iter)
    {
        if (CFL > 0.0)
            dt = gpuComputeCFL(gpu, CFL, P);

        discreteAdjointComputeRHS(da, gpu, false);
        discreteAdjointRK4Stage(da, dt, 1, solSize);

        discreteAdjointComputeRHS(da, gpu, true);
        discreteAdjointRK4Stage(da, dt, 2, solSize);

        discreteAdjointComputeRHS(da, gpu, true);
        discreteAdjointRK4Stage(da, dt, 3, solSize);

        discreteAdjointComputeRHS(da, gpu, true);
        discreteAdjointRK4Stage(da, dt, 4, solSize);

        nanFound = discreteAdjointCheckNaN(da, solSize);
        if (nanFound) {
            std::cout << "\nNaN detected at iteration " << iter << std::endl;
            break;
        }

        if ((iter + 1) % 100 == 0 || iter < 5) {
            double perVarNorms[4];
            discreteAdjointResidualNormPerVar(da, da.primaryDOF, perVarNorms);
            double resL2 = 0.0;
            for (int v = 0; v < 4; ++v) resL2 += perVarNorms[v] * perVarNorms[v];
            resL2 = std::sqrt(resL2);

            csvFile << (iter + 1)
                    << "," << perVarNorms[0]
                    << "," << perVarNorms[1]
                    << "," << perVarNorms[2]
                    << "," << perVarNorms[3] << "\n";
            csvFile.flush();

            std::cout << "Discrete adj iter " << std::setw(6) << (iter+1)
                      << "  dt=" << std::scientific << std::setprecision(4) << dt
                      << "  |res|=" << resL2 << std::endl;

            if (resL2 < adjTol) {
                converged = true;
                std::cout << "Discrete adjoint converged at iteration " << (iter+1) << std::endl;
            }
        }

        if (inp->checkpoint > 0 && (iter + 1) % inp->checkpoint == 0 && !nanFound) {
            std::vector<double> psi_raw(solSize);
            discreteAdjointCopySolutionToHost(da, psi_raw.data(), solSize);
            std::string chkBin = "discrete_adjoint_restart_" + std::to_string(iter + 1) + ".bin";
            writeAdjointRestart(chkBin, psi_raw.data(), solSize, nE, nqVol, iter + 1);

            std::vector<double> psi_chk_flat(NVAR_GPU * totalDOF);
            discreteAdjointCopyQuadPointsToHost(da, gpu, psi_chk_flat.data());
            std::vector<std::vector<double>> psi_chk(NVAR2D);
            for (int v = 0; v < NVAR2D; ++v) {
                psi_chk[v].resize(totalDOF);
                std::memcpy(psi_chk[v].data(), &psi_chk_flat[v * totalDOF], totalDOF * sizeof(double));
            }
            std::string chkVtk = "discrete_adjoint_checkpoint_" + std::to_string(iter + 1) + "_solpts.vtk";
            writeDiscreteAdjointVTK_solpts(chkVtk, mesh, geom, psi_chk, zq, nq1d, inp->ptype,
                                           {}, tau, {}, U);
            std::cout << "\nCheckpoint VTK written to " << chkVtk << std::endl;
        }

        printProgressBar(iter + 1, adjMaxIter);
    }

    csvFile.close();

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    std::cout << "\nDiscrete adjoint wall-clock time: " << elapsed.count() << " ms" << std::endl;

    // ========================================================================
    // Copy solution and write output
    // ========================================================================
    std::vector<double> psi_flat(NVAR_GPU * totalDOF);
    discreteAdjointCopyQuadPointsToHost(da, gpu, psi_flat.data());

    std::vector<std::vector<double>> psi(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) {
        psi[v].resize(totalDOF);
        std::memcpy(psi[v].data(), &psi_flat[v * totalDOF], totalDOF * sizeof(double));
    }

    std::vector<double> eta = computeErrorIndicator(
        psi_flat.data(), nE, nqVol, nq1d, totalDOF, wq, geom);
    std::vector<double> dwr = computeDWR(
        psi_flat.data(), trunc_diff, nE, nqVol, nq1d, totalDOF, wq, geom);

    writeDiscreteAdjointVTK_solpts("discrete_adjoint2d_final_solpts.vtk",
                                    mesh, geom, psi, zq, nq1d, inp->ptype,
                                    eta, tau, dwr, U);
    std::cout << "Discrete adjoint output written to discrete_adjoint2d_final_solpts.vtk" << std::endl;

    if (!nanFound) {
        std::vector<double> psi_raw(solSize);
        discreteAdjointCopySolutionToHost(da, psi_raw.data(), solSize);
        writeAdjointRestart("discrete_adjoint_restart.bin", psi_raw.data(), solSize, nE, nqVol, adjMaxIter);
    }

    // ========================================================================
    // Finite-difference gradient check
    // ========================================================================
    if (inp->adjFDCheck) {
        std::cout << "\n=== Discrete adjoint FD gradient check ===" << std::endl;
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
            double Jp;
            if (adjObjective == "LiftOverDrag") {
                double Cl_p, Cd_p;
                Jp = discreteAdjointComputeLiftOverDrag(da, gpu, chordRef, AoA_rad, Cl_p, Cd_p);
            } else {
                Jp = discreteAdjointComputeForceCoeff(da, gpu, chordRef, forceNx, forceNy);
            }

            U_pert[var * totalDOF + dof] -= 2.0 * fd_eps;
            gpuCopySolutionToDevice(gpu, U_pert.data());
            gpuComputeDGRHS(gpu, false, 0.0);
            gpuSyncUcoeff(gpu);
            double Jm;
            if (adjObjective == "LiftOverDrag") {
                double Cl_m, Cd_m;
                Jm = discreteAdjointComputeLiftOverDrag(da, gpu, chordRef, AoA_rad, Cl_m, Cd_m);
            } else {
                Jm = discreteAdjointComputeForceCoeff(da, gpu, chordRef, forceNx, forceNy);
            }

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
    // Write error indicators
    // ========================================================================
    {
        double eta_max = *std::max_element(eta.begin(), eta.end());
        double eta_sum = 0.0;
        for (auto& v : eta) eta_sum += v;
        double dwr_max = *std::max_element(dwr.begin(), dwr.end());
        double dwr_sum = 0.0;
        for (auto& v : dwr) dwr_sum += v;
        std::cout << "\nSensitivity: max(eta)=" << eta_max << "  sum(eta)=" << eta_sum << std::endl;
        std::cout << "DWR:         max(dwr)=" << dwr_max << "  sum(dwr)=" << dwr_sum << std::endl;

        std::ofstream efile("discrete_adjoint_error_indicator.dat");
        efile << "# element   eta   tau   dwr\n";
        for (int e = 0; e < nE; ++e)
            efile << e << " " << eta[e] << " " << tau[e] << " " << dwr[e] << "\n";
        efile.close();
        std::cout << "Error indicators written to discrete_adjoint_error_indicator.dat" << std::endl;
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    discreteAdjointFree(da);
    gpuFree(gpu);
    delete inp;
    return 0;
}
