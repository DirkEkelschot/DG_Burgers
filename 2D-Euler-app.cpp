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
    for (int p = 0; p <= P; ++p)
        for (int i = 0; i < npts; ++i) {
            double z = zpts[i];
            double val;
            polylib::jacobfd(1, &z, &val, NULL, p, 0.0, 0.0);
            Bvis[p][i] = val;
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
                            const std::vector<double>& sensor = {},
                            const std::vector<int>& elemP = {})
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

    bool hasCellData = !epsilon.empty() || !elemP.empty();
    if (hasCellData) {
        out << "CELL_DATA " << nCells << "\n";
        std::vector<double> cellData(nCells);
        if (!epsilon.empty()) {
            int ci = 0;
            for (int e = 0; e < nE; ++e)
                for (int ix = 0; ix < nSub; ++ix)
                    for (int iy = 0; iy < nSub; ++iy)
                        cellData[ci++] = epsilon[e];
            out << "SCALARS ArtificialViscosity float 1\nLOOKUP_TABLE default\n";
            writeBinFloats(out, cellData.data(), nCells);
            out.put('\n');
            if (!sensor.empty()) {
                int ci2 = 0;
                for (int e = 0; e < nE; ++e)
                    for (int ix = 0; ix < nSub; ++ix)
                        for (int iy = 0; iy < nSub; ++iy)
                            cellData[ci2++] = sensor[e];
                out << "SCALARS ShockSensor float 1\nLOOKUP_TABLE default\n";
                writeBinFloats(out, cellData.data(), nCells);
                out.put('\n');
            }
        }
        if (!elemP.empty()) {
            int ci = 0;
            for (int e = 0; e < nE; ++e)
                for (int ix = 0; ix < nSub; ++ix)
                    for (int iy = 0; iy < nSub; ++iy)
                        cellData[ci++] = static_cast<double>(elemP[e]);
            out << "SCALARS PolynomialOrder float 1\nLOOKUP_TABLE default\n";
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

// Read a restart file that may have a different nqVol (quadrature size) than
// what the current solver uses.  When nqVol differs, the solution is
// interpolated from the old GL quadrature to the new one via tensor-product
// Lagrange interpolation.
static bool readRestartInterp(const std::string& filename,
                              std::vector<std::vector<double>>& U,
                              int nE, int nqVol_new, int nq1d_new,
                              double& time, int& iterCount)
{
    iterCount = 0;
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

    if (nvar != NVAR2D || nE_file != nE) {
        std::cout << "Error: restart file mismatch (nvar=" << nvar
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
        if (in.read(reinterpret_cast<char*>(&iterCount), sizeof(int)).fail())
            iterCount = 0;
        in.close();
        std::cout << "Restart loaded from " << filename
                  << " (time = " << time << ", iter = " << iterCount << ")" << std::endl;
        return true;
    }

    // Quadrature mismatch: read old data and interpolate
    int nq1d_old = static_cast<int>(std::round(std::sqrt((double)nqVol_file)));
    if (nq1d_old * nq1d_old != nqVol_file) {
        std::cout << "Error: nqVol_file=" << nqVol_file
                  << " is not a perfect square" << std::endl;
        return false;
    }

    std::cout << "Restart quadrature mismatch (file nq1d=" << nq1d_old
              << ", solver nq1d=" << nq1d_new
              << "); interpolating..." << std::endl;

    int totalDOF_old = nE * nqVol_file;
    std::vector<std::vector<double>> U_old(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) {
        U_old[v].resize(totalDOF_old);
        in.read(reinterpret_cast<char*>(U_old[v].data()),
                totalDOF_old * sizeof(double));
    }
    if (in.read(reinterpret_cast<char*>(&iterCount), sizeof(int)).fail())
        iterCount = 0;
    in.close();

    // Old GL points
    std::vector<double> zq_old(nq1d_old), wq_old(nq1d_old);
    polylib::zwgl(zq_old.data(), wq_old.data(), nq1d_old);

    // New GL points
    std::vector<double> zq_new(nq1d_new), wq_new(nq1d_new);
    polylib::zwgl(zq_new.data(), wq_new.data(), nq1d_new);

    // 1D interpolation matrix: I[i * nq1d_old + j] = L_j(zq_new[i])
    std::vector<double> Imat(nq1d_new * nq1d_old);
    for (int i = 0; i < nq1d_new; ++i)
        for (int j = 0; j < nq1d_old; ++j)
            Imat[i * nq1d_old + j] = polylib::hgj(j, zq_new[i],
                                                   zq_old.data(), nq1d_old,
                                                   0.0, 0.0);

    // Tensor-product interpolation per element
    int totalDOF_new = nE * nqVol_new;
    for (int v = 0; v < NVAR2D; ++v) {
        U[v].resize(totalDOF_new);
        for (int e = 0; e < nE; ++e) {
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
    }

    std::cout << "Restart loaded and interpolated from " << filename
              << " (time = " << time << ", iter = " << iterCount << ")" << std::endl;
    return true;
}

// ============================================================================
// Compute lift and drag coefficients on CPU by integrating wall pressure
// ============================================================================

static void computeForceCoefficients(
    const std::vector<std::vector<double>>& U,
    const Mesh2D& mesh,
    const GeomData2D& geom,
    const std::vector<std::vector<double>>& Bmat,
    const std::vector<std::vector<double>>& blr,
    const std::vector<double>& wq,
    const std::vector<double>& massLU,
    const std::vector<int>& massPiv,
    int P, int nq1d,
    double rhoInf, double uInf, double vInf, double AoA_deg,
    double chordRef,
    double& Cl, double& Cd)
{
    int nE     = mesh.nElements;
    int nqVol  = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);
    int P1     = P + 1;
    int nqFace = nq1d;

    double AoA_rad = AoA_deg * M_PI / 180.0;
    double Vinf2 = uInf * uInf + vInf * vInf;
    double norm  = 0.5 * rhoInf * Vinf2 * chordRef;
    if (norm < 1e-30) norm = 1.0;

    double liftNx = -std::sin(AoA_rad), liftNy =  std::cos(AoA_rad);
    double dragNx =  std::cos(AoA_rad), dragNy =  std::sin(AoA_rad);

    // Project U to modal coefficients
    std::vector<std::vector<double>> Ucoeff(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v)
        Ucoeff[v].assign(nE * nmodes, 0.0);

    for (int e = 0; e < nE; ++e) {
        for (int v = 0; v < NVAR2D; ++v) {
            std::vector<double> rhs(nmodes, 0.0);
            for (int i = 0; i < P1; ++i)
                for (int j = 0; j < P1; ++j) {
                    int m = i * P1 + j;
                    for (int qx = 0; qx < nq1d; ++qx)
                        for (int qe = 0; qe < nq1d; ++qe) {
                            int qIdx = e * nqVol + qx * nq1d + qe;
                            double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                            rhs[m] += w * Bmat[i][qx] * Bmat[j][qe] * U[v][qIdx];
                        }
                }

            std::vector<double> Mcopy(nmodes * nmodes);
            std::vector<int> pivCopy(nmodes);
            std::memcpy(Mcopy.data(), &massLU[e * nmodes * nmodes],
                        nmodes * nmodes * sizeof(double));
            std::memcpy(pivCopy.data(), &massPiv[e * nmodes],
                        nmodes * sizeof(int));

            unsigned char TRANS = 'N';
            int NRHS = 1, INFO;
            dgetrs_(&TRANS, &nmodes, &NRHS, Mcopy.data(), &nmodes,
                    pivCopy.data(), rhs.data(), &nmodes, &INFO);

            for (int m = 0; m < nmodes; ++m)
                Ucoeff[v][e * nmodes + m] = rhs[m];
        }
    }

    // Integrate pressure on wall faces (bcTag == 1)
    Cl = 0.0;
    Cd = 0.0;
    for (int f = 0; f < mesh.nFaces; ++f) {
        if (mesh.faces[f].elemR >= 0) continue;
        if (mesh.faces[f].bcTag != 1) continue;

        int eL = mesh.faces[f].elemL;
        int lfL = mesh.faces[f].faceL;

        for (int q = 0; q < nqFace; ++q) {
            double Uf[NVAR2D] = {0, 0, 0, 0};
            for (int v = 0; v < NVAR2D; ++v)
                for (int i = 0; i < P1; ++i)
                    for (int j = 0; j < P1; ++j) {
                        int mIdx = eL * nmodes + i * P1 + j;
                        double phiXi, phiEta;
                        if      (lfL == 0) { phiXi = Bmat[i][q];               phiEta = blr[j][0]; }
                        else if (lfL == 1) { phiXi = blr[i][1];                phiEta = Bmat[j][q]; }
                        else if (lfL == 2) { phiXi = Bmat[i][nqFace - 1 - q];  phiEta = blr[j][1]; }
                        else               { phiXi = blr[i][0];                phiEta = Bmat[j][nqFace - 1 - q]; }
                        Uf[v] += Ucoeff[v][mIdx] * phiXi * phiEta;
                    }

            double p = pressure2D(Uf[0], Uf[1], Uf[2], Uf[3]);
            int fIdx = f * nqFace + q;
            double nx = geom.faceNx[fIdx];
            double ny = geom.faceNy[fIdx];
            double wf = wq[q] * geom.faceJac[fIdx];

            Cl += wf * p * (liftNx * nx + liftNy * ny) / norm;
            Cd += wf * p * (dragNx * nx + dragNy * ny) / norm;
        }
    }
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
// Variable-P forward solver
// ============================================================================

static int runVariableP(Inputs2D* inp, Mesh2D& mesh)
{
    int nE = mesh.nElements;
    int nF = mesh.nFaces;
    int pMin = inp->pMin, pMax = inp->pMax;
    int nq1d = pMax + 2;
    int nqVol = nq1d * nq1d;
    int totalDOF = nE * nqVol;

    std::vector<int> elemP;
    if (!inp->errorIndicatorFile.empty()) {
        std::vector<double> eta = readErrorIndicator(inp->errorIndicatorFile, nE);
        elemP = assignElementP(eta, pMin, pMax, inp->pAdaptThresholds);
    } else {
        elemP.assign(nE, pMax);
        std::cout << "No error indicator file; using P=" << pMax << " everywhere" << std::endl;
    }
    auto groups = buildPGroups(elemP, pMin, pMax);

    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    polylib::zwgl(zq.data(), wq.data(), nq1d);

    std::vector<double> xiVol(nqVol), etaVol(nqVol);
    for (int i = 0; i < nq1d; ++i)
        for (int j = 0; j < nq1d; ++j) {
            xiVol[i * nq1d + j]  = zq[i];
            etaVol[i * nq1d + j] = zq[j];
        }
    GeomData2D geom = computeGeometry(mesh, xiVol, etaVol, nqVol, zq, nq1d);

    double AoA_rad = inp->AoA * M_PI / 180.0;
    g_rhoInf = 1.0;
    g_pInf   = 1.0 / (GAMMA * inp->Mach * inp->Mach);
    double cInfVal = std::sqrt(GAMMA * g_pInf / g_rhoInf);
    g_uInf = inp->Mach * cInfVal * std::cos(AoA_rad);
    g_vInf = inp->Mach * cInfVal * std::sin(AoA_rad);

    std::vector<std::vector<double>> U(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v) U[v].resize(totalDOF, 0.0);
    double rhoE_inf = g_pInf / (GAMMA - 1.0)
                    + 0.5 * g_rhoInf * (g_uInf * g_uInf + g_vInf * g_vInf);
    for (int i = 0; i < totalDOF; ++i) {
        U[0][i] = g_rhoInf; U[1][i] = g_rhoInf * g_uInf;
        U[2][i] = g_rhoInf * g_vInf; U[3][i] = rhoE_inf;
    }

    double time_restart = 0.0;
    int iter_offset = 0;
    if (!inp->restartfile.empty()) {
        if (!readRestartInterp(inp->restartfile, U, nE, nqVol, nq1d,
                               time_restart, iter_offset)) {
            std::cout << "Restart failed, exiting." << std::endl;
            return 1;
        }
    }

    GPUSolverData gpu;
    gpuAllocate(gpu, nE, nF, pMax, nq1d);
    gpu.rhoInf = g_rhoInf; gpu.uInf = g_uInf;
    gpu.vInf   = g_vInf;   gpu.pInf = g_pInf;
    gpu.fluxType = (inp->fluxtype == "HLLC") ? 1 : 0;
    gpu.useAV    = inp->useAV;
    gpu.AVkappa  = inp->AVkappa;
    gpu.AVs0     = (inp->AVs0 != 0.0) ? inp->AVs0
                   : -(4.25 * std::log10((double)std::max(pMax, 1)) + 0.5);
    gpu.AVscale  = inp->AVscale;

    std::map<int, PGroupGPU> gpuGroups;
    for (auto& [p, ginfo] : groups) {
        int nEG = (int)ginfo.globalElemIdx.size();
        PGroupGPU& grp = gpuGroups[p];
        gpuAllocateGroup(grp, p, nEG);
        gpuUploadGroupElemIdx(grp, ginfo.globalElemIdx.data());

        std::vector<double> zq_c(zq), wq_c(wq);
        auto basis = BasisPoly::Create("Nodal", p, inp->ptype, zq_c, wq_c);
        basis->ConstructBasis();
        auto Bg = basis->GetB();
        auto Dg = basis->GetD();
        auto blrg = basis->GetLeftRightBasisValues();
        int P1 = p + 1, nmg = P1 * P1;

        grp.h_Bmat.resize(P1 * nq1d);
        grp.h_Dmat.resize(P1 * nq1d);
        grp.h_blr.resize(P1 * 2);
        for (int i = 0; i < P1; ++i) {
            for (int q = 0; q < nq1d; ++q) {
                grp.h_Bmat[i * nq1d + q] = Bg[i][q];
                grp.h_Dmat[i * nq1d + q] = Dg[i][q];
            }
            grp.h_blr[i * 2] = blrg[i][0];
            grp.h_blr[i * 2 + 1] = blrg[i][1];
        }
        grp.h_wq.assign(wq.begin(), wq.end());
        grp.h_faceInterp.resize(2 * nq1d);
        std::vector<double> zq_m(zq);
        for (int k = 0; k < nq1d; ++k) {
            grp.h_faceInterp[k]        = polylib::hgj(k, -1.0, zq_m.data(), nq1d, 0.0, 0.0);
            grp.h_faceInterp[nq1d + k] = polylib::hgj(k,  1.0, zq_m.data(), nq1d, 0.0, 0.0);
        }

        // NodalToModal for shock sensor
        {
            std::vector<double> zn = basis->GetZn();
            grp.h_NodalToModal.resize(P1 * P1);
            std::vector<double> V(P1 * P1);
            for (int n = 0; n < P1; ++n)
                for (int pp = 0; pp < P1; ++pp) {
                    double val;
                    polylib::jacobfd(1, &zn[n], &val, NULL, pp, 0.0, 0.0);
                    V[n * P1 + pp] = val;
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
                    grp.h_NodalToModal[r * P1 + c] = aug[r * 2*P1 + P1 + c];
        }

        // Mass matrices
        int blockSz = nmg * nmg;
        std::vector<double> massLU_g(nEG * blockSz, 0.0);
        std::vector<int> massPiv_g(nEG * nmg, 0);
        for (int eL = 0; eL < nEG; ++eL) {
            int eG = ginfo.globalElemIdx[eL];
            double* M = &massLU_g[eL * blockSz];
            for (int qx = 0; qx < nq1d; ++qx)
                for (int qe = 0; qe < nq1d; ++qe) {
                    int qIdx = eG * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[qIdx];
                    for (int i1 = 0; i1 < P1; ++i1)
                        for (int j1 = 0; j1 < P1; ++j1) {
                            int row = i1 * P1 + j1;
                            double phiR = Bg[i1][qx] * Bg[j1][qe];
                            for (int i2 = 0; i2 < P1; ++i2)
                                for (int j2 = 0; j2 < P1; ++j2) {
                                    int col = i2 * P1 + j2;
                                    M[row * nmg + col] += w * phiR * Bg[i2][qx] * Bg[j2][qe];
                                }
                        }
                }
            int INFO;
            dgetrf_(&nmg, &nmg, M, &nmg, &massPiv_g[eL * nmg], &INFO);
        }
        std::vector<double> Minv_g(nEG * blockSz, 0.0);
        for (int eL = 0; eL < nEG; ++eL) {
            double* Mb = &Minv_g[eL * blockSz];
            for (int i = 0; i < nmg; ++i) Mb[i * nmg + i] = 1.0;
            std::vector<double> LUc(&massLU_g[eL * blockSz], &massLU_g[(eL+1) * blockSz]);
            std::vector<int> pc(&massPiv_g[eL * nmg], &massPiv_g[(eL+1) * nmg]);
            unsigned char TR = 'N'; int N = nmg, NR = nmg, LA = nmg, LB = nmg, INFO;
            dgetrs_(&TR, &N, &NR, LUc.data(), &LA, pc.data(), Mb, &LB, &INFO);
        }
        gpuUploadGroupMinv(grp, Minv_g.data());
        std::cout << "  P-group P=" << p << ": " << nEG << " elements, nmodes=" << nmg << std::endl;
    }

    // Upload global static data
    {
        auto& grpMax = gpuGroups.rbegin()->second;
        int P1 = pMax + 1, nm = P1 * P1;
        std::vector<double> Minv_dummy(nE * nm * nm, 0.0);
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
            grpMax.h_Bmat.data(), grpMax.h_Dmat.data(), grpMax.h_blr.data(),
            Minv_dummy.data(), wq.data(),
            e2f.data(), feL.data(), feR.data(), ffL.data(), ffR.data(), fbc.data());
    }

    std::vector<double> U_flat(NVAR2D * totalDOF);
    for (int v = 0; v < NVAR2D; ++v)
        std::memcpy(&U_flat[v * totalDOF], U[v].data(), totalDOF * sizeof(double));
    gpuCopySolutionToDevice(gpu, U_flat.data());

    std::cout << "Variable-P GPU initialised: " << nE << " elements, PMin=" << pMin
              << ", PMax=" << pMax << ", nq1d=" << nq1d
              << ", totalDOF=" << totalDOF << std::endl;

    auto computeRHS = [&](bool useUtmp, double t) {
        for (auto& [p, grp] : gpuGroups)
            gpuComputeDGRHS_group(gpu, grp, useUtmp, t);
    };

    double time = time_restart, dt = inp->dt, CFL = inp->CFL;
    int nt = inp->nt;
    bool nanFound = false;
    auto tStart = std::chrono::high_resolution_clock::now();
    std::ofstream csvFile("residual_history.csv");
    csvFile << "iter,rho,rhou,rhov,rhoE\n";

    for (int t_step = 0; t_step < nt && !nanFound; ++t_step) {
        int gs = iter_offset + t_step;
        gpuSnapshotSolution(gpu);
        if (CFL > 0.0) dt = gpuComputeCFL(gpu, CFL, pMax);
        if (t_step == 0 && CFL > 0.0) {
            int worstElem = gpuCFLDiagnostic(gpu, CFL, pMax);
            const auto& enodes = mesh.elements[worstElem];
            double cx = 0, cy = 0;
            for (int n : enodes) { cx += mesh.nodes[n][0]; cy += mesh.nodes[n][1]; }
            cx /= enodes.size(); cy /= enodes.size();
            printf("  Limiting element centroid: (%.6f, %.6f)\n", cx, cy);
        }
        if (t_step < 5) std::cout << "Step " << gs << ": dt = " << dt << std::endl;

        computeRHS(false, time);
        if (gs % 100 == 0) {
            double n4[4]; gpuResidualNormPerVarFused(gpu, n4);
            csvFile << gs << "," << n4[0] << "," << n4[1] << "," << n4[2] << "," << n4[3] << "\n";
        }
        gpuRK4Stage(gpu, dt, 1);
        computeRHS(true, time + 0.5 * dt); gpuRK4Stage(gpu, dt, 2);
        computeRHS(true, time + 0.5 * dt); gpuRK4Stage(gpu, dt, 3);
        computeRHS(true, time + dt);       gpuRK4Stage(gpu, dt, 4);
        time += dt;

        nanFound = gpuCheckNaN(gpu);
        if (nanFound) std::cout << "\nNaN first detected at step " << gs << std::endl;
        printProgressBar(t_step + 1, nt);
    }
    csvFile.close();
    std::cout << "\nResidual history written to residual_history.csv" << std::endl;

    auto tEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Wall-clock time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count()
              << " ms\nFinal time = " << time << std::endl;
    if (nanFound) std::cout << "NaN detected -- simulation terminated early." << std::endl;

    gpuCopySolutionToHost(gpu, U_flat.data());
    for (int v = 0; v < NVAR2D; ++v)
        std::memcpy(U[v].data(), &U_flat[v * totalDOF], totalDOF * sizeof(double));
    writeVTK_solpts("solution2d_final_solpts.vtk", mesh, geom, U, zq, nq1d, inp->ptype,
                    {}, {}, elemP);
    std::cout << "Output written to solution2d_final_solpts.vtk" << std::endl;

    if (!nanFound) {
        writeRestart("restart2d.bin", U, nE, nqVol, time, iter_offset + nt);
        std::ofstream pf("elem_p_distribution.dat");
        pf << "# element P\n";
        for (int e = 0; e < nE; ++e) pf << e << " " << elemP[e] << "\n";
        pf.close();
        std::cout << "Element P distribution written to elem_p_distribution.dat" << std::endl;
    }

    for (auto& [p, grp] : gpuGroups) gpuFreeGroup(grp);
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

    // Variable-P mode
    if (inp->pMin > 0 && inp->pMax > 0) {
        int ret = runVariableP(inp, mesh);
        delete inp;
        return ret;
    }

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
    bool modalMode = (inp->btype == "Modal");

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
    if (modalMode)
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

    // In modal mode, project quadrature-point IC to modal coefficients
    std::vector<double> U_coeff;
    if (modalMode) {
        U_coeff.resize(NVAR2D * nE * nmodes, 0.0);
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
        std::cout << "IC projected to " << nE * nmodes << " modal coefficients" << std::endl;
    }

    // ========================================================================
    // Allocate and initialise GPU
    // ========================================================================
    GPUSolverData gpu;
    gpuAllocate(gpu, nE, nF, P, nq1d, modalMode);
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

    if (modalMode)
        gpuCopySolutionToDevice(gpu, U_coeff.data());
    else
        gpuCopySolutionToDevice(gpu, U_flat.data());

    // Face interpolation weights: Lagrange polynomials at GL points evaluated at z=±1
    {
        std::vector<double> interpL(nq1d), interpR(nq1d);
        std::vector<double> zq_mut(zq);
        for (int k = 0; k < nq1d; ++k) {
            interpL[k] = polylib::hgj(k, -1.0, zq_mut.data(), nq1d, 0.0, 0.0);
            interpR[k] = polylib::hgj(k,  1.0, zq_mut.data(), nq1d, 0.0, 0.0);
        }
        gpuSetFaceInterp(interpL.data(), interpR.data(), nq1d);
    }

    // Nodal-to-modal transform for shock sensor.
    // In modal mode with Legendre basis, coefficients are already Legendre → identity.
    // In nodal mode, T = V^{-1} converts Lagrange coefficients to Legendre.
    {
        int P1 = P + 1;
        std::vector<double> T(P1 * P1, 0.0);
        if (modalMode) {
            for (int i = 0; i < P1; ++i)
                T[i * P1 + i] = 1.0;
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
              << ", totalDOF=" << totalDOF
              << (modalMode ? " [MODAL]" : " [NODAL]") << std::endl;

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

        bool logForces = (inp->testcase == "NACA0012");

        std::ofstream csvFile;
        if (iter_offset > 0) {
            csvFile.open("residual_history.csv", std::ios::app);
        } else {
            csvFile.open("residual_history.csv");
            csvFile << "iter,rho,rhou,rhov,rhoE";
            if (logForces) csvFile << ",Cl,Cd";
            csvFile << "\n";
        }

        auto tBatch = std::chrono::high_resolution_clock::now();

        for (int t_step = 0; t_step < nt && !nanFound; ++t_step)
        {
            int globalStep = iter_offset + t_step;

            gpuSnapshotSolution(gpu);

            if (CFL > 0.0)
                dt = gpuComputeCFL(gpu, CFL, P);

            if (t_step == 0 && CFL > 0.0) {
                int worstElem = gpuCFLDiagnostic(gpu, CFL, P);
                const auto& enodes = mesh.elements[worstElem];
                double cx2 = 0, cy2 = 0;
                for (int n : enodes) { cx2 += mesh.nodes[n][0]; cy2 += mesh.nodes[n][1]; }
                cx2 /= enodes.size(); cy2 /= enodes.size();
                printf("  Limiting element centroid: (%.6f, %.6f)\n", cx2, cy2);
                fflush(stdout);
            }

            if (t_step < 5)
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
                        << "," << perVarNorms[3];
                if (logForces) {
                    gpuSyncUcoeff(gpu);
                    double Cl, Cd;
                    gpuComputeForceCoeff(gpu, inp->adjChordRef, inp->AoA, Cl, Cd);
                    csvFile << "," << Cl << "," << Cd;
                }
                csvFile << "\n";
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

                if (modalMode)
                    gpuCopyPrevQuadPointsToHost(gpu, U_flat.data());
                else
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
                if (modalMode)
                    gpuCopyQuadPointsToHost(gpu, U_flat.data());
                else
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

            if (t_step >= 5)
                printProgressBar(t_step + 1, nt);
        }

        csvFile.close();
        std::cout << "\nResidual history written to residual_history.csv" << std::endl;
        finalIterCount = lastGlobalStep;
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);

    std::cout << "\nWall-clock time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Final time = " << time << std::endl;

    if (nanFound)
        std::cout << "NaN detected -- simulation terminated early." << std::endl;

    // ========================================================================
    // Copy solution back to host (quad-point values for VTK/error computation)
    // ========================================================================
    if (modalMode)
        gpuCopyQuadPointsToHost(gpu, U_flat.data());
    else
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
    // Compute and print lift/drag coefficients (NACA0012)
    // ========================================================================
    if (inp->testcase == "NACA0012" && !nanFound) {
        gpuSyncUcoeff(gpu);
        double Cl, Cd;
        gpuComputeForceCoeff(gpu, inp->adjChordRef, inp->AoA, Cl, Cd);
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "\nForce coefficients:" << std::endl;
        std::cout << "  Lift coefficient (Cl) = " << Cl << std::endl;
        std::cout << "  Drag coefficient (Cd) = " << Cd << std::endl;
        std::cout << std::defaultfloat;
    }

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
            forceNx =  std::cos(AoA_rad);
            forceNy =  std::sin(AoA_rad);
        } else {
            forceNx = -std::sin(AoA_rad);
            forceNy =  std::cos(AoA_rad);
        }
        std::cout << "Adjoint objective  = " << adjObjective << std::endl;
        std::cout << "Force direction    = (" << forceNx << ", " << forceNy << ")" << std::endl;

        // Freeze forward state: populate Ucoeff and epsilon
        gpuComputeDGRHS(gpu, false, 0.0);
        gpuSyncUcoeff(gpu);
        std::cout << "Forward solution frozen (Ucoeff/epsilon)." << std::endl;

        // Allocate adjoint data
        AdjointGPUData adj;
        adjointGpuAllocate(adj, gpu);
        adj.chordRef = chordRef;
        adj.maxAVscale = (double)P / (2.0 * CFL * (2.0 * P + 1.0));
        std::cout << "Adjoint AV: maxAVscale(CFL) = " << adj.maxAVscale
                  << ", input AVscale = " << gpu.AVscale
                  << ", effective = " << std::min((double)gpu.AVscale, adj.maxAVscale)
                  << std::endl;

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
        double adjTol = inp->adjTol;
        int adjMaxIter = inp->adjMaxIter;

        std::cout << "Adjoint solver: " << nE << " elements, P=" << P
                  << ", nq1d=" << nq1d << ", totalDOF=" << totalDOF << std::endl;
        std::cout << "Max adjoint iterations = " << adjMaxIter << std::endl;
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
                          << "  dt=" << std::scientific << std::setprecision(4) << adjDt
                          << "  |res|=" << resL2 << std::endl;

                if (resL2 < adjTol) {
                    adjConverged = true;
                    std::cout << "Adjoint converged at iteration " << (iter+1) << std::endl;
                }
            }

            if (inp->checkpoint > 0 && (iter + 1) % inp->checkpoint == 0 && !adjNanFound) {
                // Always get quad-point values for VTK output
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

                // Save raw coefficients (or quad values) for restart
                adjointCopySolutionToHost(adj, psi_chk.data(), solSize);
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

        // Copy adjoint solution back (quad-point values for VTK)
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
        writeAdjointVTK_solpts("adjoint2d_final_solpts.vtk", mesh, geom, psi, zq, nq1d,
                               inp->ptype, eta);
        std::cout << "Adjoint solution written to adjoint2d_final.vtk" << std::endl;
        std::cout << "Adjoint solution-point output written to adjoint2d_final_solpts.vtk" << std::endl;

        if (!adjNanFound) {
            std::vector<double> psi_raw(solSize);
            adjointCopySolutionToHost(adj, psi_raw.data(), solSize);
            writeAdjointRestart("adjoint_restart.bin", psi_raw.data(), solSize, nE, nqVol, adjMaxIter);
        }

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
