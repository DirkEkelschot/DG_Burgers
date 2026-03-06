// Truncation error & DWR utility
//
// Evaluates the coarse (P2) DG residual operator on a fine (P3) solution
// loaded from a restart file, and outputs per-element L2 norms to a VTK file.
// Optionally accepts an adjoint restart to compute the full DWR error indicator.
//
// Usage: ./truncerr <inputs_coarse.xml> <fine_restart.bin> [--adjoint adj.bin] [output.vtk]

#include "src/io.h"
#include "src/mesh2d.h"
#include "src/geom2d.h"
#include "src/basis_poly.h"
#include "src/euler2d.h"
#include "src/Polylib.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <memory>

// ============================================================================
// Freestream globals (needed by BC functions)
// ============================================================================

static double g_rhoInf, g_uInf, g_vInf, g_pInf;

// ============================================================================
// CPU boundary condition functions (matching the GPU versions)
// ============================================================================

static void slipWallBC(const double UL[NVAR2D], double nx, double ny,
                       double /*x*/, double /*y*/, double /*t*/,
                       double UR[NVAR2D])
{
    double rho = UL[0];
    double u   = UL[1] / rho;
    double v   = UL[2] / rho;
    double Vn  = u * nx + v * ny;
    UR[0] = rho;
    UR[1] = rho * (u - 2.0 * Vn * nx);
    UR[2] = rho * (v - 2.0 * Vn * ny);
    UR[3] = UL[3];
}

static void riemannInvariantBC(const double UL[NVAR2D], double nx, double ny,
                               double /*x*/, double /*y*/, double /*t*/,
                               double UR[NVAR2D])
{
    const double gm1 = GAMMA - 1.0;

    double rhoI = UL[0];
    double uI   = UL[1] / rhoI;
    double vI   = UL[2] / rhoI;
    double pI   = pressure2D(UL[0], UL[1], UL[2], UL[3]);
    double cI   = std::sqrt(GAMMA * pI / rhoI);
    double VnI  = uI * nx + vI * ny;

    double cInfVal = std::sqrt(GAMMA * g_pInf / g_rhoInf);
    double VnInf   = g_uInf * nx + g_vInf * ny;

    double Rplus  = VnI   + 2.0 * cI / gm1;
    double Rminus = VnInf - 2.0 * cInfVal / gm1;
    double sB, VtB_x, VtB_y;

    if (VnI < 0.0) {
        sB    = g_pInf / std::pow(g_rhoInf, GAMMA);
        VtB_x = g_uInf - VnInf * nx;
        VtB_y = g_vInf - VnInf * ny;
    } else {
        sB    = pI / std::pow(rhoI, GAMMA);
        VtB_x = uI - VnI * nx;
        VtB_y = vI - VnI * ny;
    }

    double VnB = 0.5 * (Rplus + Rminus);
    double cB  = 0.25 * gm1 * (Rplus - Rminus);
    if (cB < 1e-14) cB = 1e-14;

    double rhoB = std::pow(cB * cB / (GAMMA * sB), 1.0 / gm1);
    double uB   = VtB_x + VnB * nx;
    double vB   = VtB_y + VnB * ny;
    double pB   = rhoB * cB * cB / GAMMA;

    UR[0] = rhoB;
    UR[1] = rhoB * uB;
    UR[2] = rhoB * vB;
    UR[3] = pB / gm1 + 0.5 * rhoB * (uB * uB + vB * vB);
}

// ============================================================================
// Read restart with interpolation to a (possibly different) quadrature grid
// ============================================================================

static bool readRestartInterp(const std::string& filename,
                              std::vector<std::vector<double>>& U,
                              int nE, int nqVol_new, int nq1d_new,
                              double& time)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open restart file " << filename << std::endl;
        return false;
    }

    int nvar, nE_file, nqVol_file;
    in.read(reinterpret_cast<char*>(&nvar),      sizeof(int));
    in.read(reinterpret_cast<char*>(&nE_file),   sizeof(int));
    in.read(reinterpret_cast<char*>(&nqVol_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&time),       sizeof(double));

    if (nvar != NVAR2D || nE_file != nE) {
        std::cerr << "Error: restart file mismatch (nvar=" << nvar
                  << " nE=" << nE_file << "), expected (" << NVAR2D
                  << ", " << nE << ")" << std::endl;
        return false;
    }

    if (nqVol_file == nqVol_new) {
        int totalDOF = nE * nqVol_new;
        for (int v = 0; v < NVAR2D; ++v) {
            U[v].resize(totalDOF);
            in.read(reinterpret_cast<char*>(U[v].data()), totalDOF * sizeof(double));
        }
        in.close();
        std::cout << "Fine restart loaded from " << filename
                  << " (time = " << time << ", nqVol = " << nqVol_file << ")" << std::endl;
        return true;
    }

    int nq1d_old = static_cast<int>(std::round(std::sqrt((double)nqVol_file)));
    if (nq1d_old * nq1d_old != nqVol_file) {
        std::cerr << "Error: nqVol_file=" << nqVol_file
                  << " is not a perfect square" << std::endl;
        return false;
    }

    std::cout << "Quadrature mismatch (file nq1d=" << nq1d_old
              << ", coarse nq1d=" << nq1d_new
              << "); interpolating..." << std::endl;

    int totalDOF_old = nE * nqVol_file;
    std::vector<std::vector<double>> U_old(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) {
        U_old[v].resize(totalDOF_old);
        in.read(reinterpret_cast<char*>(U_old[v].data()),
                totalDOF_old * sizeof(double));
    }
    in.close();

    std::vector<double> zq_old(nq1d_old), wq_old(nq1d_old);
    polylib::zwgl(zq_old.data(), wq_old.data(), nq1d_old);

    std::vector<double> zq_new(nq1d_new), wq_new(nq1d_new);
    polylib::zwgl(zq_new.data(), wq_new.data(), nq1d_new);

    // 1D Lagrange interpolation matrix from old GL grid to new GL grid
    std::vector<double> Imat(nq1d_new * nq1d_old);
    for (int i = 0; i < nq1d_new; ++i)
        for (int j = 0; j < nq1d_old; ++j)
            Imat[i * nq1d_old + j] = polylib::hgj(j, zq_new[i],
                                                   zq_old.data(), nq1d_old,
                                                   0.0, 0.0);

    int totalDOF_new = nE * nqVol_new;
    for (int v = 0; v < NVAR2D; ++v) {
        U[v].resize(totalDOF_new);
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nq1d_new; ++ix)
                for (int jy = 0; jy < nq1d_new; ++jy) {
                    double val = 0.0;
                    for (int a = 0; a < nq1d_old; ++a)
                        for (int b = 0; b < nq1d_old; ++b)
                            val += Imat[ix * nq1d_old + a]
                                 * Imat[jy * nq1d_old + b]
                                 * U_old[v][e * nqVol_file + a * nq1d_old + b];
                    U[v][e * nqVol_new + ix * nq1d_new + jy] = val;
                }
    }

    std::cout << "Fine restart loaded and interpolated from " << filename
              << " (time = " << time << ")" << std::endl;
    return true;
}

// ============================================================================
// Read adjoint restart with interpolation to a (possibly different) quadrature
// grid.  The adjoint file format differs from the forward restart: the header
// stores  nvar | nE | nqVol | iter  (iter is int, not double), and the data
// is flat:  psi_flat[v * totalDOF + e*nqVol + q].
// On output, psi is reshaped to psi[v][e*nqVol + q] for convenience.
// ============================================================================

static bool readAdjointRestartInterp(const std::string& filename,
                                     std::vector<std::vector<double>>& psi,
                                     int nE, int nqVol_new, int nq1d_new)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open adjoint restart file " << filename << std::endl;
        return false;
    }

    int nvar, nE_file, nqVol_file, iter;
    in.read(reinterpret_cast<char*>(&nvar),       sizeof(int));
    in.read(reinterpret_cast<char*>(&nE_file),    sizeof(int));
    in.read(reinterpret_cast<char*>(&nqVol_file), sizeof(int));
    in.read(reinterpret_cast<char*>(&iter),        sizeof(int));

    if (nvar != NVAR2D || nE_file != nE) {
        std::cerr << "Error: adjoint restart mismatch (nvar=" << nvar
                  << " nE=" << nE_file << "), expected (" << NVAR2D
                  << ", " << nE << ")" << std::endl;
        return false;
    }

    int totalDOF_file = nE * nqVol_file;
    int solSize_file  = NVAR2D * totalDOF_file;

    std::vector<double> psi_flat(solSize_file);
    in.read(reinterpret_cast<char*>(psi_flat.data()), solSize_file * sizeof(double));
    in.close();

    // Reshape flat [v*totalDOF + idx] -> psi_old[v][idx]
    std::vector<std::vector<double>> psi_old(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) {
        psi_old[v].resize(totalDOF_file);
        std::memcpy(psi_old[v].data(),
                    &psi_flat[v * totalDOF_file],
                    totalDOF_file * sizeof(double));
    }

    if (nqVol_file == nqVol_new) {
        psi = std::move(psi_old);
        std::cout << "Adjoint restart loaded from " << filename
                  << " (iter = " << iter << ", nqVol = " << nqVol_file << ")" << std::endl;
        return true;
    }

    // Quadrature mismatch: interpolate
    int nq1d_old = static_cast<int>(std::round(std::sqrt((double)nqVol_file)));
    if (nq1d_old * nq1d_old != nqVol_file) {
        std::cerr << "Error: adjoint nqVol_file=" << nqVol_file
                  << " is not a perfect square" << std::endl;
        return false;
    }

    std::cout << "Adjoint quadrature mismatch (file nq1d=" << nq1d_old
              << ", coarse nq1d=" << nq1d_new
              << "); interpolating..." << std::endl;

    std::vector<double> zq_old(nq1d_old), wq_old(nq1d_old);
    polylib::zwgl(zq_old.data(), wq_old.data(), nq1d_old);

    std::vector<double> zq_new(nq1d_new), wq_new(nq1d_new);
    polylib::zwgl(zq_new.data(), wq_new.data(), nq1d_new);

    std::vector<double> Imat(nq1d_new * nq1d_old);
    for (int i = 0; i < nq1d_new; ++i)
        for (int j = 0; j < nq1d_old; ++j)
            Imat[i * nq1d_old + j] = polylib::hgj(j, zq_new[i],
                                                   zq_old.data(), nq1d_old,
                                                   0.0, 0.0);

    psi.resize(NVAR2D);
    int nqVol_old = nqVol_file;
    for (int v = 0; v < NVAR2D; ++v) {
        psi[v].resize(nE * nqVol_new);
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nq1d_new; ++ix)
                for (int jy = 0; jy < nq1d_new; ++jy) {
                    double val = 0.0;
                    for (int a = 0; a < nq1d_old; ++a)
                        for (int b = 0; b < nq1d_old; ++b)
                            val += Imat[ix * nq1d_old + a]
                                 * Imat[jy * nq1d_old + b]
                                 * psi_old[v][e * nqVol_old + a * nq1d_old + b];
                    psi[v][e * nqVol_new + ix * nq1d_new + jy] = val;
                }
    }

    std::cout << "Adjoint restart loaded and interpolated from " << filename
              << " (iter = " << iter << ")" << std::endl;
    return true;
}

// ============================================================================
// VTK helpers (big-endian binary for legacy VTK format)
// ============================================================================

static void toBigEndian(void* data, size_t count, int elemSize) {
    char* p = static_cast<char*>(data);
    for (size_t i = 0; i < count; ++i, p += elemSize)
        std::reverse(p, p + elemSize);
}

static void writeBinFloats(std::ofstream& out, const double* src, int n) {
    std::vector<float> buf(n);
    for (int i = 0; i < n; ++i) buf[i] = static_cast<float>(src[i]);
    toBigEndian(buf.data(), n, sizeof(float));
    out.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(float));
}

static void writeBinInts(std::ofstream& out, const int* src, int n) {
    std::vector<int> buf(src, src + n);
    toBigEndian(buf.data(), n, sizeof(int));
    out.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(int));
}

// ============================================================================
// Write per-element truncation error to VTK (cell data on the mesh quads)
// ============================================================================

static void writeTruncErrorVTK(const std::string& filename,
                               const Mesh2D& mesh,
                               const std::vector<double>& tau,
                               const std::vector<double>& dwr = {})
{
    int nE = mesh.nElements;
    int nN = mesh.nNodes;

    std::ofstream out(filename, std::ios::binary);
    out << "# vtk DataFile Version 3.0\n"
        << "Truncation Error\n"
        << "BINARY\n"
        << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << nN << " float\n";
    {
        std::vector<float> pts(nN * 3);
        for (int i = 0; i < nN; ++i) {
            pts[i * 3 + 0] = static_cast<float>(mesh.nodes[i][0]);
            pts[i * 3 + 1] = static_cast<float>(mesh.nodes[i][1]);
            pts[i * 3 + 2] = 0.0f;
        }
        toBigEndian(pts.data(), nN * 3, sizeof(float));
        out.write(reinterpret_cast<const char*>(pts.data()), nN * 3 * sizeof(float));
        out.put('\n');
    }

    out << "CELLS " << nE << " " << nE * 5 << "\n";
    {
        std::vector<int> cells(nE * 5);
        for (int e = 0; e < nE; ++e) {
            cells[e * 5 + 0] = 4;
            cells[e * 5 + 1] = mesh.elements[e][0];
            cells[e * 5 + 2] = mesh.elements[e][1];
            cells[e * 5 + 3] = mesh.elements[e][2];
            cells[e * 5 + 4] = mesh.elements[e][3];
        }
        writeBinInts(out, cells.data(), nE * 5);
        out.put('\n');
    }

    out << "CELL_TYPES " << nE << "\n";
    {
        std::vector<int> types(nE, 9);
        writeBinInts(out, types.data(), nE);
        out.put('\n');
    }

    out << "CELL_DATA " << nE << "\n";

    out << "SCALARS TruncationError float 1\nLOOKUP_TABLE default\n";
    writeBinFloats(out, tau.data(), nE);
    out.put('\n');

    {
        std::vector<double> logTau(nE);
        for (int e = 0; e < nE; ++e)
            logTau[e] = (tau[e] > 0.0) ? std::log10(tau[e]) : -16.0;
        out << "SCALARS Log10_TruncationError float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, logTau.data(), nE);
        out.put('\n');
    }

    if (!dwr.empty()) {
        out << "SCALARS DWR_Indicator float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, dwr.data(), nE);
        out.put('\n');

        std::vector<double> logDwr(nE);
        for (int e = 0; e < nE; ++e)
            logDwr[e] = (dwr[e] > 0.0) ? std::log10(dwr[e]) : -16.0;
        out << "SCALARS Log10_DWR_Indicator float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, logDwr.data(), nE);
        out.put('\n');
    }

    out.close();
    std::cout << "VTK written to " << filename << std::endl;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <inputs_coarse.xml> <fine_restart.bin>"
                  << " [--adjoint adj_restart.bin] [output.vtk]"
                  << std::endl;
        return 1;
    }

    std::string xmlFile      = argv[1];
    std::string fineRestart  = argv[2];
    std::string adjointFile;
    std::string outputVTK    = "truncation_error.vtk";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--adjoint" && i + 1 < argc) {
            adjointFile = argv[++i];
        } else {
            outputVTK = arg;
        }
    }

    // ------------------------------------------------------------------
    // 1. Parse coarse-discretization XML
    // ------------------------------------------------------------------
    Inputs2D* inp = ReadXmlFile2D(xmlFile.c_str());
    if (!inp) return 1;

    int P    = inp->porder;
    int nq1d = inp->nquad;

    std::cout << "Coarse discretization: P=" << P
              << ", nq1d=" << nq1d
              << ", basis=" << inp->btype
              << ", points=" << inp->ptype << std::endl;

    // ------------------------------------------------------------------
    // 2. Read mesh
    // ------------------------------------------------------------------
    Mesh2D mesh;
    mesh.readGmsh(inp->meshfile);
    std::cout << "Mesh: " << mesh.nElements << " elements, "
              << mesh.nFaces << " faces, "
              << mesh.nNodes << " nodes" << std::endl;

    int nE    = mesh.nElements;
    int nqVol = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);

    // ------------------------------------------------------------------
    // 3. Set up quadrature and basis
    // ------------------------------------------------------------------
    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    if (inp->ptype == "GaussLegendre") {
        polylib::zwgl(zq.data(), wq.data(), nq1d);
    } else {
        polylib::zwgll(zq.data(), wq.data(), nq1d);
    }

    std::unique_ptr<BasisPoly> basis1D = BasisPoly::Create(inp->btype, P, inp->ptype, zq, wq);
    basis1D->ConstructBasis();

    std::vector<std::vector<double>> Bmat = basis1D->GetB();

    // ------------------------------------------------------------------
    // 4. Compute geometry at coarse quadrature points
    // ------------------------------------------------------------------
    std::vector<double> xiVol(nqVol), etaVol(nqVol);
    for (int i = 0; i < nq1d; ++i)
        for (int j = 0; j < nq1d; ++j) {
            xiVol[i * nq1d + j]  = zq[i];
            etaVol[i * nq1d + j] = zq[j];
        }

    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

    // ------------------------------------------------------------------
    // 5. Assemble and factor mass matrices (coarse discretization)
    // ------------------------------------------------------------------
    std::vector<double> massLU;
    std::vector<int>    massPiv;
    assembleAndFactorMassMatrices(mesh, geom, Bmat, Bmat, wq, wq,
                                 P, nq1d, massLU, massPiv);

    // ------------------------------------------------------------------
    // 6. Set up freestream state and boundary condition map
    // ------------------------------------------------------------------
    double AoA_rad = inp->AoA * M_PI / 180.0;
    g_rhoInf = 1.0;
    g_pInf   = 1.0 / (GAMMA * inp->Mach * inp->Mach);
    double cInf = std::sqrt(GAMMA * g_pInf / g_rhoInf);
    g_uInf = inp->Mach * cInf * std::cos(AoA_rad);
    g_vInf = inp->Mach * cInf * std::sin(AoA_rad);

    std::map<int, BoundaryStateFunc> bcMap;
    if (inp->testcase == "NACA0012") {
        bcMap[1] = slipWallBC;
        bcMap[2] = riemannInvariantBC;
        bcMap[3] = riemannInvariantBC;
    }

    // ------------------------------------------------------------------
    // 7. Read fine restart and interpolate to coarse quadrature points
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> U(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) U[v].resize(nE * nqVol, 0.0);

    double time_fine = 0.0;
    if (!readRestartInterp(fineRestart, U, nE, nqVol, nq1d, time_fine)) {
        delete inp;
        return 1;
    }

    // ------------------------------------------------------------------
    // 8. Evaluate the coarse DG residual: R = M^{-1} * (flux terms)
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> R(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v)
        R[v].resize(nE * nqVol, 0.0);

    computeDGRHS2D(mesh, geom, basis1D.get(),
                   P, nq1d, nq1d,
                   wq, wq, zq, zq,
                   U, R,
                   massLU, massPiv, nmodes,
                   time_fine, bcMap);

    // ------------------------------------------------------------------
    // 9. Compute per-element L2 norm of the residual (truncation error)
    // ------------------------------------------------------------------
    std::vector<double> tau(nE, 0.0);
    for (int e = 0; e < nE; ++e) {
        double val = 0.0;
        for (int v = 0; v < NVAR2D; ++v)
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    int qIdx = e * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                    double r = R[v][qIdx];
                    val += w * r * r;
                }
        tau[e] = std::sqrt(val);
    }

    // ------------------------------------------------------------------
    // 10. (Optional) Load adjoint and compute DWR indicator
    // ------------------------------------------------------------------
    std::vector<double> dwr;

    if (!adjointFile.empty()) {
        std::vector<std::vector<double>> psi(NVAR2D);
        if (!readAdjointRestartInterp(adjointFile, psi, nE, nqVol, nq1d)) {
            delete inp;
            return 1;
        }

        // DWR indicator: eta_e = |sum_v sum_q w_q * detJ_q * psi_v(q) * R_v(q)|
        dwr.resize(nE, 0.0);
        for (int e = 0; e < nE; ++e) {
            double val = 0.0;
            for (int v = 0; v < NVAR2D; ++v)
                for (int qx = 0; qx < nq1d; ++qx)
                    for (int qe = 0; qe < nq1d; ++qe) {
                        int qIdx = e * nqVol + qx * nq1d + qe;
                        double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                        val += w * psi[v][qIdx] * R[v][qIdx];
                    }
            dwr[e] = std::fabs(val);
        }
    }

    // ------------------------------------------------------------------
    // 11. Print summary statistics
    // ------------------------------------------------------------------
    double tau_max = *std::max_element(tau.begin(), tau.end());
    double tau_sum = std::accumulate(tau.begin(), tau.end(), 0.0);
    double tau_mean = tau_sum / nE;
    double tau_l2 = 0.0;
    for (auto& t : tau) tau_l2 += t * t;
    tau_l2 = std::sqrt(tau_l2);

    std::cout << "\nTruncation error  tau = R_P" << P << "(u_fine):"
              << "\n  max  = " << tau_max
              << "\n  mean = " << tau_mean
              << "\n  sum  = " << tau_sum
              << "\n  L2   = " << tau_l2
              << std::endl;

    if (!dwr.empty()) {
        double dwr_max = *std::max_element(dwr.begin(), dwr.end());
        double dwr_sum = std::accumulate(dwr.begin(), dwr.end(), 0.0);
        double dwr_mean = dwr_sum / nE;

        std::cout << "\nDWR indicator  eta_e = |<psi, tau>|_e:"
                  << "\n  max  = " << dwr_max
                  << "\n  mean = " << dwr_mean
                  << "\n  sum  = " << dwr_sum
                  << "  (approx functional error J(u)-J(u_h))"
                  << std::endl;
    }

    // ------------------------------------------------------------------
    // 12. Write VTK output
    // ------------------------------------------------------------------
    writeTruncErrorVTK(outputVTK, mesh, tau, dwr);

    delete inp;
    return 0;
}
