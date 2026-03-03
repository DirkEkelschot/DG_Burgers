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

using namespace polylib;

// ============================================================================
// Freestream state (globals set from input file before the time loop)
// ============================================================================

static double g_rhoInf, g_uInf, g_vInf, g_pInf;

// ============================================================================
// Evaluate 1D modal basis at arbitrary points
// ============================================================================

static void evalModalBasis1D(int P, const std::vector<double>& zpts,
                             std::vector<std::vector<double>>& Bvis)
{
    int npts = (int)zpts.size();
    Bvis.resize(P + 1, std::vector<double>(npts));
    for (int p = 0; p <= P; ++p) {
        for (int i = 0; i < npts; ++i) {
            double z = zpts[i];
            if (p == 0)
                Bvis[p][i] = (1.0 - z) / 2.0;
            else if (p == P)
                Bvis[p][i] = (1.0 + z) / 2.0;
            else {
                double val;
                polylib::jacobfd(1, &z, &val, NULL, p - 1, 1.0, 1.0);
                Bvis[p][i] = (1.0 - z) / 2.0 * (1.0 + z) / 2.0 * val;
            }
        }
    }
}

// ============================================================================
// Evaluate 1D nodal basis at arbitrary points
// ============================================================================

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
// Binary VTK helpers (big-endian for VTK legacy format)
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
// VTK output with interpolation to equispaced points (no gaps)
// ============================================================================

static void writeVTK(const std::string& filename,
                     const Mesh2D& mesh,
                     const GeomData2D& geom,
                     const std::vector<std::vector<double>>& U,
                     const std::vector<std::vector<double>>& Bmat,
                     const std::vector<std::vector<double>>& Bvis,
                     const std::vector<double>& Minv,
                     const std::vector<double>& wq,
                     const std::vector<double>& zVis,
                     int P, int nq1d,
                     const std::vector<double>& epsilon = {},
                     const std::vector<double>& sensor = {})
{
    int nE = mesh.nElements;
    int nqVol = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);
    int nVis = (int)zVis.size();
    int nVisVol = nVis * nVis;
    int totalPts = nE * nVisVol;

    // Compute physical coordinates and interpolated solution at vis points
    std::vector<double> xVis(totalPts), yVis(totalPts);
    std::vector<std::vector<double>> Uvis(NVAR2D, std::vector<double>(totalPts));

    for (int e = 0; e < nE; ++e)
    {
        // Forward transform: project to modal coefficients
        std::vector<double> coeff(NVAR2D * nmodes, 0.0);
        for (int v = 0; v < NVAR2D; ++v)
        {
            std::vector<double> proj(nmodes, 0.0);
            for (int i = 0; i <= P; ++i)
                for (int j = 0; j <= P; ++j) {
                    int m = i * (P + 1) + j;
                    for (int qx = 0; qx < nq1d; ++qx)
                        for (int qe = 0; qe < nq1d; ++qe) {
                            int qIdx = e * nqVol + qx * nq1d + qe;
                            double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                            proj[m] += w * Bmat[i][qx] * Bmat[j][qe] * U[v][qIdx];
                        }
                }
            for (int m = 0; m < nmodes; ++m) {
                double val = 0.0;
                for (int mp = 0; mp < nmodes; ++mp)
                    val += Minv[e * nmodes * nmodes + m * nmodes + mp] * proj[mp];
                coeff[v * nmodes + m] = val;
            }
        }

        // Evaluate at vis points: physical coords + backward transform
        for (int ix = 0; ix < nVis; ++ix)
            for (int iy = 0; iy < nVis; ++iy)
            {
                int idx = e * nVisVol + ix * nVis + iy;
                refToPhys(mesh, e, zVis[ix], zVis[iy], xVis[idx], yVis[idx]);

                for (int v = 0; v < NVAR2D; ++v) {
                    double val = 0.0;
                    for (int i = 0; i <= P; ++i)
                        for (int j = 0; j <= P; ++j)
                            val += coeff[v * nmodes + i*(P+1)+j]
                                 * Bvis[i][ix] * Bvis[j][iy];
                    Uvis[v][idx] = val;
                }
            }
    }

    // Write binary VTK file
    std::ofstream out(filename, std::ios::binary);
    out << "# vtk DataFile Version 3.0\n"
        << "2D Euler DG Solution\n"
        << "BINARY\n"
        << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << totalPts << " float\n";
    {
        std::vector<float> pts(totalPts * 3);
        for (int i = 0; i < totalPts; ++i) {
            pts[i*3]   = static_cast<float>(xVis[i]);
            pts[i*3+1] = static_cast<float>(yVis[i]);
            pts[i*3+2] = 0.0f;
        }
        toBigEndian(pts.data(), totalPts * 3, sizeof(float));
        out.write(reinterpret_cast<const char*>(pts.data()), totalPts * 3 * sizeof(float));
        out.put('\n');
    }

    int nSub = nVis - 1;
    int nCells = nE * nSub * nSub;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    {
        std::vector<int> cells(nCells * 5);
        int ci = 0;
        for (int e = 0; e < nE; ++e) {
            int base = e * nVisVol;
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy) {
                    cells[ci++] = 4;
                    cells[ci++] = base + ix * nVis + iy;
                    cells[ci++] = base + (ix+1) * nVis + iy;
                    cells[ci++] = base + (ix+1) * nVis + (iy+1);
                    cells[ci++] = base + ix * nVis + (iy+1);
                }
        }
        writeBinInts(out, cells.data(), nCells * 5);
        out.put('\n');
    }

    out << "CELL_TYPES " << nCells << "\n";
    {
        std::vector<int> types(nCells, 9);
        writeBinInts(out, types.data(), nCells);
        out.put('\n');
    }

    out << "POINT_DATA " << totalPts << "\n";

    const char* varNames[4] = {"rho", "rhou", "rhov", "rhoE"};
    for (int v = 0; v < NVAR2D; ++v) {
        out << "SCALARS " << varNames[v] << " float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, Uvis[v].data(), totalPts);
        out.put('\n');
    }

    {
        std::vector<double> tmp(totalPts);
        for (int i = 0; i < totalPts; ++i) {
            double rho = std::max(Uvis[0][i], 1e-14);
            double ke  = 0.5 * (Uvis[1][i]*Uvis[1][i] + Uvis[2][i]*Uvis[2][i]) / rho;
            tmp[i] = std::max((GAMMA - 1.0) * (Uvis[3][i] - ke), 1e-14);
        }
        out << "SCALARS Pressure float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, tmp.data(), totalPts);
        out.put('\n');
    }

    {
        std::vector<double> tmp(totalPts);
        for (int i = 0; i < totalPts; ++i) {
            double rho = std::max(Uvis[0][i], 1e-14);
            double u = Uvis[1][i] / rho;
            double v = Uvis[2][i] / rho;
            double ke = 0.5 * rho * (u * u + v * v);
            double p  = std::max((GAMMA - 1.0) * (Uvis[3][i] - ke), 1e-14);
            double c  = std::sqrt(GAMMA * p / rho);
            tmp[i] = std::sqrt(u * u + v * v) / c;
        }
        out << "SCALARS Mach float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, tmp.data(), totalPts);
        out.put('\n');
    }

    if (!epsilon.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        std::vector<double> cellData(nCells);
        int ci = 0;
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy)
                    cellData[ci++] = epsilon[e];
        out << "SCALARS ArtificialViscosity float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, cellData.data(), nCells);
        out.put('\n');
        if (!sensor.empty()) {
            ci = 0;
            for (int e = 0; e < nE; ++e)
                for (int ix = 0; ix < nSub; ++ix)
                    for (int iy = 0; iy < nSub; ++iy)
                        cellData[ci++] = sensor[e];
            out << "SCALARS ShockSensor float 1\nLOOKUP_TABLE default\n";
            writeBinFloats(out, cellData.data(), nCells);
            out.put('\n');
        }
    }

    out.close();
}

// ============================================================================
// VTK output at actual solution (quadrature) points
// ============================================================================

static void writeVTK_solpts(const std::string& filename,
                            const Mesh2D& mesh,
                            const GeomData2D& geom,
                            const std::vector<std::vector<double>>& U,
                            const std::vector<double>& zq,
                            int nq1d,
                            const std::string& ptype,
                            const std::vector<double>& epsilon = {},
                            const std::vector<double>& sensor = {})
{
    int nE    = mesh.nElements;
    int nqVol = nq1d * nq1d;

    bool needAug = (ptype == "GaussLegendre");
    int nAug    = needAug ? nq1d + 2 : nq1d;
    int nAugVol = nAug * nAug;

    // Build augmented 1D point set and interpolation matrix Interp[nAug x nq1d]
    std::vector<double> zAug(nAug);
    std::vector<double> Interp(nAug * nq1d, 0.0);

    if (needAug) {
        zAug[0] = -1.0;
        for (int k = 0; k < nq1d; ++k) zAug[k + 1] = zq[k];
        zAug[nAug - 1] = 1.0;

        // Interior rows: identity mapping
        for (int k = 0; k < nq1d; ++k)
            Interp[(k + 1) * nq1d + k] = 1.0;

        // Boundary rows: Lagrange interpolation weights
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

    // Compute physical coordinates and interpolated solution at augmented points
    std::vector<double> xPts(totalPts), yPts(totalPts);
    std::vector<std::vector<double>> Uaug(NVAR2D, std::vector<double>(totalPts));

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
                                 * U[v][e * nqVol + a * nq1d + b];
                    Uaug[v][idx] = val;
                }
            }

    // Write binary VTK
    std::ofstream out(filename, std::ios::binary);
    out << "# vtk DataFile Version 3.0\n"
        << "2D Euler DG Solution at quadrature points\n"
        << "BINARY\n"
        << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << totalPts << " float\n";
    {
        std::vector<float> pts(totalPts * 3);
        for (int i = 0; i < totalPts; ++i) {
            pts[i*3]   = static_cast<float>(xPts[i]);
            pts[i*3+1] = static_cast<float>(yPts[i]);
            pts[i*3+2] = 0.0f;
        }
        toBigEndian(pts.data(), totalPts * 3, sizeof(float));
        out.write(reinterpret_cast<const char*>(pts.data()), totalPts * 3 * sizeof(float));
        out.put('\n');
    }

    int nSub   = nAug - 1;
    int nCells = nE * nSub * nSub;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    {
        std::vector<int> cells(nCells * 5);
        int ci = 0;
        for (int e = 0; e < nE; ++e) {
            int base = e * nAugVol;
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy) {
                    cells[ci++] = 4;
                    cells[ci++] = base + ix       * nAug + iy;
                    cells[ci++] = base + (ix + 1) * nAug + iy;
                    cells[ci++] = base + (ix + 1) * nAug + (iy + 1);
                    cells[ci++] = base + ix       * nAug + (iy + 1);
                }
        }
        writeBinInts(out, cells.data(), nCells * 5);
        out.put('\n');
    }

    out << "CELL_TYPES " << nCells << "\n";
    {
        std::vector<int> types(nCells, 9);
        writeBinInts(out, types.data(), nCells);
        out.put('\n');
    }

    out << "POINT_DATA " << totalPts << "\n";

    const char* varNames[4] = {"rho", "rhou", "rhov", "rhoE"};
    for (int v = 0; v < NVAR2D; ++v) {
        out << "SCALARS " << varNames[v] << " float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, Uaug[v].data(), totalPts);
        out.put('\n');
    }

    {
        std::vector<double> tmp(totalPts);
        for (int i = 0; i < totalPts; ++i) {
            double rho = std::max(Uaug[0][i], 1e-14);
            double ke  = 0.5 * (Uaug[1][i]*Uaug[1][i] + Uaug[2][i]*Uaug[2][i]) / rho;
            tmp[i] = std::max((GAMMA - 1.0) * (Uaug[3][i] - ke), 1e-14);
        }
        out << "SCALARS Pressure float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, tmp.data(), totalPts);
        out.put('\n');
    }

    {
        std::vector<double> tmp(totalPts);
        for (int i = 0; i < totalPts; ++i) {
            double rho = std::max(Uaug[0][i], 1e-14);
            double u   = Uaug[1][i] / rho;
            double v   = Uaug[2][i] / rho;
            double ke  = 0.5 * rho * (u * u + v * v);
            double p   = std::max((GAMMA - 1.0) * (Uaug[3][i] - ke), 1e-14);
            double c   = std::sqrt(GAMMA * p / rho);
            tmp[i] = std::sqrt(u * u + v * v) / c;
        }
        out << "SCALARS Mach float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, tmp.data(), totalPts);
        out.put('\n');
    }

    if (!epsilon.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        std::vector<double> cellData(nCells);
        int ci = 0;
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy)
                    cellData[ci++] = epsilon[e];
        out << "SCALARS ArtificialViscosity float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, cellData.data(), nCells);
        out.put('\n');
        if (!sensor.empty()) {
            ci = 0;
            for (int e = 0; e < nE; ++e)
                for (int ix = 0; ix < nSub; ++ix)
                    for (int iy = 0; iy < nSub; ++iy)
                        cellData[ci++] = sensor[e];
            out << "SCALARS ShockSensor float 1\nLOOKUP_TABLE default\n";
            writeBinFloats(out, cellData.data(), nCells);
            out.put('\n');
        }
    }

    out.close();
}

// ============================================================================
// Progress bar
// ============================================================================

// ============================================================================
// Restart file I/O (binary, quadrature-point solution)
// ============================================================================

static void writeRestart(const std::string& filename,
                         const std::vector<std::vector<double>>& U,
                         int nE, int nqVol, double time, int iterCount = 0)
{
    int totalDOF = nE * nqVol;
    std::ofstream out(filename, std::ios::binary);
    int nvar = NVAR2D;
    out.write(reinterpret_cast<const char*>(&nvar), sizeof(int));
    out.write(reinterpret_cast<const char*>(&nE), sizeof(int));
    out.write(reinterpret_cast<const char*>(&nqVol), sizeof(int));
    out.write(reinterpret_cast<const char*>(&time), sizeof(double));
    for (int v = 0; v < NVAR2D; ++v)
        out.write(reinterpret_cast<const char*>(U[v].data()), totalDOF * sizeof(double));
    out.write(reinterpret_cast<const char*>(&iterCount), sizeof(int));
    out.close();
    std::cout << "Restart written to " << filename
              << " (time = " << time << ", iter = " << iterCount << ")" << std::endl;
}

static bool readRestart(const std::string& filename,
                        std::vector<std::vector<double>>& U,
                        int nE, int nqVol, double& time, int& iterCount)
{
    iterCount = 0;
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
        std::cout << "Error: restart file mismatch (nvar=" << nvar
                  << " nE=" << nE_file << " nqVol=" << nqVol_file
                  << "), expected (4, " << nE << ", " << nqVol << ")" << std::endl;
        return false;
    }
    for (int v = 0; v < NVAR2D; ++v) {
        U[v].resize(totalDOF);
        in.read(reinterpret_cast<char*>(U[v].data()), totalDOF * sizeof(double));
    }
    if (in.read(reinterpret_cast<char*>(&iterCount), sizeof(int)).fail())
        iterCount = 0;
    in.close();
    std::cout << "Restart loaded from " << filename
              << " (time = " << time << ", iter = " << iterCount << ")" << std::endl;
    return true;
}

static void printProgressBar(int current, int total, int width = 50)
{
    float progress = static_cast<float>(current) / total;
    int barWidth = static_cast<int>(width * progress);
    fprintf(stderr, "\r[");
    for (int i = 0; i < width; ++i)
    {
        if (i < barWidth)       fprintf(stderr, "=");
        else if (i == barWidth) fprintf(stderr, ">");
        else                    fprintf(stderr, " ");
    }
    fprintf(stderr, "] %3d%%", static_cast<int>(progress * 100.0));
    fflush(stderr);
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

    std::vector<double> xVis_(totalPts), yVis_(totalPts);
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
                refToPhys(mesh, e, zVis[ix], zVis[iy], xVis_[idx], yVis_[idx]);
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

    std::ofstream out(filename, std::ios::binary);
    out << "# vtk DataFile Version 3.0\n"
        << "2D Euler Adjoint Solution\n"
        << "BINARY\n"
        << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << totalPts << " float\n";
    {
        std::vector<float> pts(totalPts * 3);
        for (int i = 0; i < totalPts; ++i) {
            pts[i*3]   = static_cast<float>(xVis_[i]);
            pts[i*3+1] = static_cast<float>(yVis_[i]);
            pts[i*3+2] = 0.0f;
        }
        toBigEndian(pts.data(), totalPts * 3, sizeof(float));
        out.write(reinterpret_cast<const char*>(pts.data()), totalPts * 3 * sizeof(float));
        out.put('\n');
    }

    int nSub = nVis - 1;
    int nCells = nE * nSub * nSub;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    {
        std::vector<int> cells(nCells * 5);
        int ci = 0;
        for (int e = 0; e < nE; ++e) {
            int base = e * nVisVol;
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy) {
                    cells[ci++] = 4;
                    cells[ci++] = base + ix * nVis + iy;
                    cells[ci++] = base + (ix+1) * nVis + iy;
                    cells[ci++] = base + (ix+1) * nVis + (iy+1);
                    cells[ci++] = base + ix * nVis + (iy+1);
                }
        }
        writeBinInts(out, cells.data(), nCells * 5);
        out.put('\n');
    }

    out << "CELL_TYPES " << nCells << "\n";
    {
        std::vector<int> types(nCells, 9);
        writeBinInts(out, types.data(), nCells);
        out.put('\n');
    }

    out << "POINT_DATA " << totalPts << "\n";
    const char* adjVarNames[4] = {"lambda_rho", "lambda_rhou", "lambda_rhov", "lambda_rhoE"};
    for (int v = 0; v < NVAR2D; ++v) {
        out << "SCALARS " << adjVarNames[v] << " float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, PsiVis[v].data(), totalPts);
        out.put('\n');
    }

    if (!eta.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        std::vector<double> cellData(nCells);
        int ci = 0;
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy)
                    cellData[ci++] = eta[e];
        out << "SCALARS ErrorIndicator float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, cellData.data(), nCells);
        out.put('\n');
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

    std::ofstream out(filename, std::ios::binary);
    out << "# vtk DataFile Version 3.0\n"
        << "2D Euler Adjoint at quadrature points\n"
        << "BINARY\n"
        << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << totalPts << " float\n";
    {
        std::vector<float> pts(totalPts * 3);
        for (int i = 0; i < totalPts; ++i) {
            pts[i*3]   = static_cast<float>(xPts[i]);
            pts[i*3+1] = static_cast<float>(yPts[i]);
            pts[i*3+2] = 0.0f;
        }
        toBigEndian(pts.data(), totalPts * 3, sizeof(float));
        out.write(reinterpret_cast<const char*>(pts.data()), totalPts * 3 * sizeof(float));
        out.put('\n');
    }

    int nSub   = nAug - 1;
    int nCells = nE * nSub * nSub;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    {
        std::vector<int> cells(nCells * 5);
        int ci = 0;
        for (int e = 0; e < nE; ++e) {
            int base = e * nAugVol;
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy) {
                    cells[ci++] = 4;
                    cells[ci++] = base + ix       * nAug + iy;
                    cells[ci++] = base + (ix + 1) * nAug + iy;
                    cells[ci++] = base + (ix + 1) * nAug + (iy + 1);
                    cells[ci++] = base + ix       * nAug + (iy + 1);
                }
        }
        writeBinInts(out, cells.data(), nCells * 5);
        out.put('\n');
    }

    out << "CELL_TYPES " << nCells << "\n";
    {
        std::vector<int> types(nCells, 9);
        writeBinInts(out, types.data(), nCells);
        out.put('\n');
    }

    out << "POINT_DATA " << totalPts << "\n";
    const char* adjVarNames[4] = {"lambda_rho", "lambda_rhou", "lambda_rhov", "lambda_rhoE"};
    for (int v = 0; v < NVAR2D; ++v) {
        out << "SCALARS " << adjVarNames[v] << " float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, PsiAug[v].data(), totalPts);
        out.put('\n');
    }

    if (!eta.empty()) {
        out << "CELL_DATA " << nCells << "\n";
        std::vector<double> cellData(nCells);
        int ci = 0;
        for (int e = 0; e < nE; ++e)
            for (int ix = 0; ix < nSub; ++ix)
                for (int iy = 0; iy < nSub; ++iy)
                    cellData[ci++] = eta[e];
        out << "SCALARS ErrorIndicator float 1\nLOOKUP_TABLE default\n";
        writeBinFloats(out, cellData.data(), nCells);
        out.put('\n');
    }

    out.close();
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

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    setvbuf(stderr, NULL, _IONBF, 0);

    std::string inputFile = "inputs2d.xml";
    if (argc > 1)
        inputFile = argv[1];

    Inputs2D* inp = ReadXmlFile2D(inputFile.c_str());
    if (!inp) return 1;

    int    P     = inp->porder;
    int    nq1d  = inp->nquad;
    double dt    = inp->dt;
    int    nt    = inp->nt;
    double CFL   = inp->CFL;

    // ========================================================================
    // Read mesh
    // ========================================================================
    Mesh2D mesh;
    mesh.readGmsh(inp->meshfile);

    std::cout << "Mesh: " << mesh.nElements << " elements, "
              << mesh.nFaces << " faces, "
              << mesh.nNodes << " nodes" << std::endl;

    // ========================================================================
    // Set up 1D quadrature and basis
    // ========================================================================
    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    if (inp->ptype == "GaussLegendre") {
        zwgl(zq.data(), wq.data(), nq1d);
    } else {
        zwgll(zq.data(), wq.data(), nq1d);
    }

    std::unique_ptr<BasisPoly> basis1D = BasisPoly::Create(inp->btype, P, inp->ptype, zq, wq);
    basis1D->ConstructBasis();

    int nqVol = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);

    std::vector<double> xiVol(nqVol), etaVol(nqVol);
    for (int i = 0; i < nq1d; ++i)
        for (int j = 0; j < nq1d; ++j)
        {
            xiVol[i * nq1d + j]  = zq[i];
            etaVol[i * nq1d + j] = zq[j];
        }

    // ========================================================================
    // Compute geometry
    // ========================================================================
    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

    // ========================================================================
    // Assemble and factor mass matrices
    // ========================================================================
    std::vector<std::vector<double>> Bmat = basis1D->GetB();
    std::vector<std::vector<double>> Dmat = basis1D->GetD();
    std::vector<std::vector<double>> blr  = basis1D->GetLeftRightBasisValues();

    std::vector<double> massLU;
    std::vector<int>    massPiv;
    assembleAndFactorMassMatrices(mesh, geom, Bmat, Bmat, wq, wq,
                                 P, nq1d, massLU, massPiv);

    // ========================================================================
    // Compute mass matrix inverse (for GPU)
    // ========================================================================
    std::vector<double> Minv;
    computeMassInverse(massLU, massPiv, mesh.nElements, nmodes, Minv);

    // ========================================================================
    // Build visualization basis (equispaced points including endpoints)
    // ========================================================================
    int nVis = std::max(nq1d, 15);
    std::vector<double> zVis(nVis);
    for (int i = 0; i < nVis; ++i)
        zVis[i] = -1.0 + 2.0 * i / (nVis - 1);

    std::vector<std::vector<double>> Bvis;
    if (inp->btype == "Modal")
        evalModalBasis1D(P, zVis, Bvis);
    else {
        std::vector<double> zn = basis1D->GetZn();
        evalNodalBasis1D(P, inp->ptype, zn, zVis, Bvis);
    }

    // ========================================================================
    // Initial conditions
    // ========================================================================
    int nE = mesh.nElements;
    int totalDOF = nE * nqVol;

    std::vector<std::vector<double>> U(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v)
        U[v].resize(totalDOF, 0.0);

    {
        double AoA_rad = inp->AoA * M_PI / 180.0;
        g_rhoInf = 1.0;
        g_pInf   = 1.0 / (GAMMA * inp->Mach * inp->Mach);
        double cInf = std::sqrt(GAMMA * g_pInf / g_rhoInf);
        g_uInf = inp->Mach * cInf * std::cos(AoA_rad);
        g_vInf = inp->Mach * cInf * std::sin(AoA_rad);
    }

    if (inp->testcase == "NACA0012")
    {
        double rhoE_inf = g_pInf / (GAMMA - 1.0)
                        + 0.5 * g_rhoInf * (g_uInf * g_uInf + g_vInf * g_vInf);
        for (int i = 0; i < totalDOF; ++i)
        {
            U[0][i] = g_rhoInf;
            U[1][i] = g_rhoInf * g_uInf;
            U[2][i] = g_rhoInf * g_vInf;
            U[3][i] = rhoE_inf;
        }
    }
    else if (inp->testcase == "IsentropicVortex")
    {
        const double gamma  = 1.4;
        const double gm1    = gamma - 1.0;
        const double Minf   = 0.5;
        const double beta   = 5.0;
        const double uInf = 1.0, vInf = 1.0;
        const double pInf = 1.0 / (gamma * Minf * Minf);
        const double rhoInf = 1.0, TInf = pInf / rhoInf;

        for (int e = 0; e < nE; ++e)
            for (int q = 0; q < nqVol; ++q)
            {
                int idx = e * nqVol + q;
                double x = geom.xPhys[idx], y = geom.yPhys[idx];
                double Lx = 10.0, Ly = 10.0, x0 = 5.0, y0 = 5.0;
                double xr = x - x0, yr = y - y0;
                xr -= Lx * std::floor(xr / Lx + 0.5);
                yr -= Ly * std::floor(yr / Ly + 0.5);
                double r2 = xr*xr + yr*yr;
                double f  = beta / (2.0*M_PI) * std::exp(0.5*(1.0 - r2));
                double du = -yr*f, dv = xr*f;
                double dT = -gm1*beta*beta / (8.0*gamma*M_PI*M_PI) * std::exp(1.0 - r2);
                double T = TInf + dT, u = uInf + du, v = vInf + dv;
                double rho = std::pow(T/TInf, 1.0/gm1) * rhoInf;
                double p = rho * T;
                U[0][idx] = rho;
                U[1][idx] = rho * u;
                U[2][idx] = rho * v;
                U[3][idx] = p / gm1 + 0.5 * rho * (u*u + v*v);
            }
    }

    // Load restart if specified
    double time_restart = 0.0;
    int iter_offset = 0;
    if (!inp->restartfile.empty()) {
        if (!readRestart(inp->restartfile, U, nE, nqVol, time_restart, iter_offset)) {
            std::cout << "Restart failed, exiting." << std::endl;
            return 1;
        }
    }

    writeVTK("solution2d_init.vtk", mesh, geom, U, Bmat, Bvis, Minv, wq, zVis, P, nq1d);
    writeVTK_solpts("solution2d_init_solpts.vtk", mesh, geom, U, zq, nq1d, inp->ptype);

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
                if (tag == 1) face_bcType[f] = 1;      // slip wall
                else          face_bcType[f] = 2;      // farfield / wake
            }
        }
    }

    // Flatten solution: U_flat[v * totalDOF + i]
    std::vector<double> U_flat(NVAR2D * totalDOF);
    for (int v = 0; v < NVAR2D; ++v)
        std::memcpy(&U_flat[v * totalDOF], U[v].data(), totalDOF * sizeof(double));

    // ========================================================================
    // Allocate and initialise GPU
    // ========================================================================
    GPUSolverData gpu;
    gpuAllocate(gpu, nE, nF, P, nq1d);
    gpu.rhoInf = g_rhoInf;
    gpu.uInf   = g_uInf;
    gpu.vInf   = g_vInf;
    gpu.pInf   = g_pInf;
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

    gpuCopySolutionToDevice(gpu, U_flat.data());

    // Nodal-to-modal transform for shock sensor
    {
        int P1 = P + 1;
        std::vector<double> T(P1 * P1, 0.0);
        if (inp->btype == "Modal") {
            for (int i = 0; i < P1; ++i) T[i * P1 + i] = 1.0;
        } else {
            std::vector<double> zn = basis1D->GetZn();
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
                    double f = aug[r * 2*P1 + col];
                    for (int c = 0; c < 2 * P1; ++c)
                        aug[r * 2*P1 + c] -= f * aug[col * 2*P1 + c];
                }
            }
            for (int r = 0; r < P1; ++r)
                for (int c = 0; c < P1; ++c)
                    T[r * P1 + c] = aug[r * 2*P1 + P1 + c];
        }
        gpuSetNodalToModal(T.data(), P1);
        adjointGpuSetNodalToModal(T.data(), P1);
    }

    std::cout << "GPU initialised: " << nE << " elements, P=" << P
              << ", nq1d=" << nq1d << ", nmodes=" << nmodes
              << ", totalDOF=" << totalDOF << std::endl;

    // ========================================================================
    // Time integration on GPU
    // ========================================================================
    double time = time_restart;
    bool nanFound = false;
    int finalIterCount = iter_offset;
    auto tStart = std::chrono::high_resolution_clock::now();

    {
        // ==================================================================
        // Explicit RK4 time integration
        // ==================================================================
        const int logInterval = 100;
        const int timerInterval = 10000;
        int lastGlobalStep = iter_offset;

        std::ofstream csvFile;
        if (iter_offset > 0) {
            csvFile.open("residual_history.csv", std::ios::app);
        } else {
            csvFile.open("residual_history.csv");
            csvFile << "iter,rho,rhou,rhov,rhoE\n";
        }

        auto tBatch = std::chrono::high_resolution_clock::now();

        for (int t_step = 0; t_step < nt && !nanFound; ++t_step)
        {
            int globalStep = iter_offset + t_step;

            gpuSnapshotSolution(gpu);

            if (CFL > 0.0)
                dt = gpuComputeCFL(gpu, CFL, P);

            if (t_step < 5 && iter_offset == 0)
                std::cout << "Step " << globalStep << ": dt = " << dt << std::endl;

            gpuComputeDGRHS(gpu, false, time);

            bool doLog = (globalStep % logInterval == 0) && !(t_step == 0 && iter_offset > 0);
            if (doLog) {
                double perVarNorms[4];
                gpuResidualNormPerVarFused(gpu, perVarNorms);
                csvFile << globalStep
                        << "," << perVarNorms[0]
                        << "," << perVarNorms[1]
                        << "," << perVarNorms[2]
                        << "," << perVarNorms[3] << "\n";
            }

            gpuRK4Stage(gpu, dt, 1);

            gpuComputeDGRHS(gpu, true, time + 0.5 * dt);
            gpuRK4Stage(gpu, dt, 2);

            gpuComputeDGRHS(gpu, true, time + 0.5 * dt);
            gpuRK4Stage(gpu, dt, 3);

            gpuComputeDGRHS(gpu, true, time + dt);
            gpuRK4Stage(gpu, dt, 4);

            time += dt;
            lastGlobalStep = globalStep + 1;

            nanFound = gpuCheckNaN(gpu);
            if (nanFound) {
                std::cout << "\nNaN first detected at step " << globalStep << std::endl;

                gpuCopyPrevSolutionToHost(gpu, U_flat.data());
                for (int v = 0; v < NVAR2D; ++v)
                    std::memcpy(U[v].data(), &U_flat[v * totalDOF], totalDOF * sizeof(double));

                std::vector<double> eps_nan, sensor_nan;
                if (gpu.useAV) {
                    eps_nan.resize(nE);
                    sensor_nan.resize(nE);
                    gpuCopyEpsilonToHost(gpu, eps_nan.data());
                    gpuCopySensorToHost(gpu, sensor_nan.data());
                }
                writeVTK("solution2d_prenan.vtk", mesh, geom, U, Bmat, Bvis,
                         Minv, wq, zVis, P, nq1d, eps_nan, sensor_nan);
                std::cout << "Pre-NaN solution written to solution2d_prenan.vtk" << std::endl;
            }

            if ((t_step + 1) % timerInterval == 0) {
                auto tNow = std::chrono::high_resolution_clock::now();
                double batchMs = std::chrono::duration<double, std::milli>(tNow - tBatch).count();
                double msPerStep = batchMs / timerInterval;
                fprintf(stderr, " [%.3f ms/step]", msPerStep);
                tBatch = tNow;
            }

            if (inp->checkpoint > 0 && (globalStep + 1) % inp->checkpoint == 0 && !nanFound) {
                gpuCopySolutionToHost(gpu, U_flat.data());
                for (int v = 0; v < NVAR2D; ++v)
                    std::memcpy(U[v].data(), &U_flat[v * totalDOF], totalDOF * sizeof(double));

                std::string chkVtk = "checkpoint_" + std::to_string(globalStep + 1) + ".vtk";
                std::vector<double> eps_chk, sensor_chk;
                if (gpu.useAV) {
                    eps_chk.resize(nE);
                    sensor_chk.resize(nE);
                    gpuCopyEpsilonToHost(gpu, eps_chk.data());
                    gpuCopySensorToHost(gpu, sensor_chk.data());
                }
                writeVTK(chkVtk, mesh, geom, U, Bmat, Bvis, Minv, wq, zVis, P, nq1d,
                         eps_chk, sensor_chk);
                std::string chkSolpts = "checkpoint_" + std::to_string(globalStep + 1) + "_solpts.vtk";
                writeVTK_solpts(chkSolpts, mesh, geom, U, zq, nq1d, inp->ptype,
                                eps_chk, sensor_chk);
                std::string chkBin = "restart2d_" + std::to_string(globalStep + 1) + ".bin";
                writeRestart(chkBin, U, nE, nqVol, time, globalStep + 1);
                std::cout << "\nCheckpoint at step " << globalStep + 1
                          << " (time = " << time << ")" << std::endl;
            }

            printProgressBar(t_step + 1, nt);
        }

        csvFile.close();
        std::cout << "Residual history written to residual_history.csv" << std::endl;
        finalIterCount = lastGlobalStep;
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);

    std::cout << "\nWall-clock time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Final time = " << time << std::endl;

    if (nanFound)
        std::cout << "NaN detected -- simulation terminated early." << std::endl;

    // ========================================================================
    // Copy solution back to host
    // ========================================================================
    gpuCopySolutionToHost(gpu, U_flat.data());
    for (int v = 0; v < NVAR2D; ++v)
        std::memcpy(U[v].data(), &U_flat[v * totalDOF], totalDOF * sizeof(double));

    // ========================================================================
    // Compute L2 error for isentropic vortex
    // ========================================================================
    if (inp->testcase == "IsentropicVortex" && !nanFound)
    {
        const double gamma  = 1.4;
        const double gm1    = gamma - 1.0;
        const double Minf   = 0.5;
        const double beta   = 5.0;
        const double uInf = 1.0, vInf = 1.0;
        const double pInfV = 1.0 / (gamma * Minf * Minf);
        const double rhoInf = 1.0, TInf = pInfV / rhoInf;

        double l2err_rho = 0.0, l2norm = 0.0;
        for (int e = 0; e < nE; ++e)
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe)
                {
                    int idx = e * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[idx];
                    double x = geom.xPhys[idx], y = geom.yPhys[idx];
                    double Lx=10,Ly=10,x0=5,y0=5;
                    double xr = x - uInf*time - x0, yr = y - vInf*time - y0;
                    xr -= Lx*std::floor(xr/Lx+0.5);
                    yr -= Ly*std::floor(yr/Ly+0.5);
                    double r2=xr*xr+yr*yr;
                    double f=beta/(2.0*M_PI)*std::exp(0.5*(1.0-r2));
                    double dT=-gm1*beta*beta/(8.0*gamma*M_PI*M_PI)*std::exp(1.0-r2);
                    double T=TInf+dT;
                    double rhoEx=std::pow(T/TInf,1.0/gm1)*rhoInf;
                    double diff = U[0][idx] - rhoEx;
                    l2err_rho += w * diff * diff;
                    l2norm    += w * rhoEx * rhoEx;
                }

        l2err_rho = std::sqrt(l2err_rho);
        l2norm    = std::sqrt(l2norm);
        std::cout << std::scientific << std::setprecision(8);
        std::cout << "L2 error (density)          = " << l2err_rho << std::endl;
        std::cout << "Relative L2 error (density) = " << l2err_rho / l2norm << std::endl;
    }

    // ========================================================================
    // Write final solution
    // ========================================================================
    std::vector<double> eps_host, sensor_host;
    if (gpu.useAV) {
        eps_host.resize(nE);
        sensor_host.resize(nE);
        gpuCopyEpsilonToHost(gpu, eps_host.data());
        gpuCopySensorToHost(gpu, sensor_host.data());
    }
    writeVTK("solution2d_final.vtk", mesh, geom, U, Bmat, Bvis, Minv, wq, zVis, P, nq1d,
             eps_host, sensor_host);
    writeVTK_solpts("solution2d_final_solpts.vtk", mesh, geom, U, zq, nq1d,
                    inp->ptype, eps_host, sensor_host);
    std::cout << "Output written to solution2d_init.vtk, solution2d_final.vtk" << std::endl;
    std::cout << "Solution-point output written to solution2d_init_solpts.vtk, solution2d_final_solpts.vtk" << std::endl;

    if (!nanFound)
        writeRestart("restart2d.bin", U, nE, nqVol, time, finalIterCount);

    // ========================================================================
    // Adjoint solve (if requested)
    // ========================================================================
    if (inp->runAdjoint && !nanFound)
    {
        std::cout << "\n========================================================" << std::endl;
        std::cout << "  Starting adjoint solve" << std::endl;
        std::cout << "========================================================" << std::endl;

        double AoA_rad = inp->AoA * M_PI / 180.0;
        double chordRef = inp->adjChordRef;
        std::string adjObjective = inp->adjObjective;

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

        // Freeze forward state: populate Ucoeff and epsilon
        gpuComputeDGRHS(gpu, false, 0.0);
        std::cout << "Forward solution frozen (Ucoeff/epsilon)." << std::endl;

        // Allocate adjoint data
        AdjointGPUData adj;
        adjointGpuAllocate(adj, gpu);
        adj.chordRef = chordRef;

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

        int solSize = NVAR_GPU * totalDOF;
        double adjTol = inp->adjTol;
        int adjMaxIter = nt;

        std::cout << "Adjoint solver: " << nE << " elements, P=" << P
                  << ", nq1d=" << nq1d << ", totalDOF=" << totalDOF << std::endl;
        std::cout << "Max adjoint iterations = " << adjMaxIter << " (= forward nt)" << std::endl;
        std::cout << "Adjoint tolerance      = " << adjTol << std::endl;

        // Adjoint RK4 pseudo-time iteration
        auto tAdjStart = std::chrono::high_resolution_clock::now();
        bool adjConverged = false;
        bool adjNanFound = false;
        double adjDt = dt;

        std::vector<double> psi_chk(solSize);

        std::ofstream adjCsvFile("adjoint_residual_history.csv");
        adjCsvFile << "iter,lambda_rho,lambda_rhou,lambda_rhov,lambda_rhoE\n";

        for (int iter = 0; iter < adjMaxIter && !adjConverged && !adjNanFound; ++iter)
        {
            if (CFL > 0.0)
                adjDt = gpuComputeCFL(gpu, CFL, P);

            adjointComputeRHS(adj, gpu, false);
            adjointRK4Stage(adj, adjDt, 1, solSize);

            adjointComputeRHS(adj, gpu, true);
            adjointRK4Stage(adj, adjDt, 2, solSize);

            adjointComputeRHS(adj, gpu, true);
            adjointRK4Stage(adj, adjDt, 3, solSize);

            adjointComputeRHS(adj, gpu, true);
            adjointRK4Stage(adj, adjDt, 4, solSize);

            adjNanFound = adjointCheckNaN(adj, solSize);
            if (adjNanFound) {
                std::cout << "\nNaN detected in adjoint at iteration " << iter << std::endl;
                break;
            }

            if ((iter + 1) % 100 == 0 || iter < 5) {
                adjointComputeRHS(adj, gpu, false);

                double perVarNorms[4];
                adjointResidualNormPerVar(adj, totalDOF, perVarNorms);
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
                          << "  dt=" << std::scientific << std::setprecision(4) << adjDt
                          << "  |res|=" << resL2 << std::endl;

                if (resL2 < adjTol) {
                    adjConverged = true;
                    std::cout << "Adjoint converged at iteration " << (iter+1) << std::endl;
                }
            }

            if (inp->checkpoint > 0 && (iter + 1) % inp->checkpoint == 0 && !adjNanFound) {
                adjointCopySolutionToHost(adj, psi_chk.data(), solSize);

                std::vector<std::vector<double>> psi_vtk(NVAR2D);
                for (int v = 0; v < NVAR2D; ++v) {
                    psi_vtk[v].resize(totalDOF);
                    std::memcpy(psi_vtk[v].data(), &psi_chk[v * totalDOF],
                                totalDOF * sizeof(double));
                }

                std::vector<double> eta_chk = computeErrorIndicator(
                    psi_chk.data(), nE, nqVol, nq1d, totalDOF, wq, geom);

                std::string chkVtk = "adjoint_checkpoint_" + std::to_string(iter + 1) + ".vtk";
                writeAdjointVTK(chkVtk, mesh, geom, psi_vtk, Bmat, Bvis,
                                Minv, wq, zVis, P, nq1d, eta_chk);

                std::string chkBin = "adjoint_restart_" + std::to_string(iter + 1) + ".bin";
                writeAdjointRestart(chkBin, psi_chk.data(), solSize, nE, nqVol, iter + 1);

                std::cout << "\nAdjoint checkpoint at iteration " << (iter + 1) << std::endl;
            }

            printProgressBar(iter + 1, adjMaxIter);
        }

        adjCsvFile.close();

        auto tAdjEnd = std::chrono::high_resolution_clock::now();
        auto adjElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tAdjEnd - tAdjStart);
        std::cout << "\nAdjoint wall-clock time: " << adjElapsed.count() << " ms" << std::endl;

        // Copy adjoint solution back and write VTK
        std::vector<double> psi_flat(solSize);
        adjointCopySolutionToHost(adj, psi_flat.data(), solSize);

        std::vector<std::vector<double>> psi(NVAR2D);
        for (int v = 0; v < NVAR2D; ++v) {
            psi[v].resize(totalDOF);
            std::memcpy(psi[v].data(), &psi_flat[v * totalDOF], totalDOF * sizeof(double));
        }

        std::vector<double> eta = computeErrorIndicator(
            psi_flat.data(), nE, nqVol, nq1d, totalDOF, wq, geom);

        writeAdjointVTK("adjoint2d_final.vtk", mesh, geom, psi, Bmat, Bvis,
                        Minv, wq, zVis, P, nq1d, eta);
        writeAdjointVTK_solpts("adjoint2d_final_solpts.vtk", mesh, geom, psi, zq, nq1d,
                               inp->ptype, eta);
        std::cout << "Adjoint solution written to adjoint2d_final.vtk" << std::endl;
        std::cout << "Adjoint solution-point output written to adjoint2d_final_solpts.vtk" << std::endl;

        if (!adjNanFound)
            writeAdjointRestart("adjoint_restart.bin", psi_flat.data(), solSize, nE, nqVol, adjMaxIter);

        // Finite-difference gradient check
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
                double Jp = adjointComputeForceCoeff(adj, gpu, chordRef, forceNx, forceNy);

                U_pert[var * totalDOF + dof] -= 2.0 * fd_eps;
                gpuCopySolutionToDevice(gpu, U_pert.data());
                gpuComputeDGRHS(gpu, false, 0.0);
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
        }

        // Write sensitivity indicator
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

        adjointGpuFree(adj);
    }

    gpuFree(gpu);
    delete inp;
    return 0;
}
