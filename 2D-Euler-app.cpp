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

using namespace polylib;

// ============================================================================
// Isentropic vortex initial / exact solution
// ============================================================================

static void isentropicVortexState(double x, double y, double t,
                                  double Ubc[NVAR2D]);

static void isentropicVortex(double x, double y, double t,
                             double& rho, double& rhou, double& rhov, double& rhoE)
{
    const double gamma  = 1.4;
    const double gm1    = gamma - 1.0;
    const double Minf   = 0.5;
    const double beta   = 5.0;

    const double uInf = 1.0;
    const double vInf = 1.0;
    const double pInf = 1.0 / (gamma * Minf * Minf);
    const double rhoInf = 1.0;
    const double TInf = pInf / rhoInf;

    // Periodicity on [0, 10] x [0, 10]
    double Lx = 10.0, Ly = 10.0;
    double x0 = 5.0, y0 = 5.0;

    double xr = x - uInf * t - x0;
    double yr = y - vInf * t - y0;

    // Wrap for periodicity
    xr = xr - Lx * std::floor(xr / Lx + 0.5);
    yr = yr - Ly * std::floor(yr / Ly + 0.5);

    double r2 = xr * xr + yr * yr;
    double f  = beta / (2.0 * M_PI) * std::exp(0.5 * (1.0 - r2));

    double du = -yr * f;
    double dv =  xr * f;
    double dT = -gm1 * beta * beta / (8.0 * gamma * M_PI * M_PI) * std::exp(1.0 - r2);

    double T   = TInf + dT;
    double u   = uInf + du;
    double v   = vInf + dv;

    rho  = std::pow(T / TInf, 1.0 / gm1) * rhoInf;
    double p = rho * T;

    rhou = rho * u;
    rhov = rho * v;
    rhoE = p / gm1 + 0.5 * rho * (u * u + v * v);
}

static void isentropicVortexState(double x, double y, double t,
                                  double Ubc[NVAR2D])
{
    isentropicVortex(x, y, t, Ubc[0], Ubc[1], Ubc[2], Ubc[3]);
}

static double g_currentTime = 0.0;

static void isentropicVortexBC(const double /*UL*/[NVAR2D], double /*nx*/, double /*ny*/,
                               double x, double y, double /*t*/,
                               double UR[NVAR2D])
{
    isentropicVortex(x, y, g_currentTime, UR[0], UR[1], UR[2], UR[3]);
}

// ============================================================================
// Freestream state (globals set from input file before the time loop)
// ============================================================================

static double g_rhoInf, g_uInf, g_vInf, g_pInf;

// ============================================================================
// Slip wall boundary condition (inviscid wall -- reflect normal velocity)
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

// ============================================================================
// Riemann invariant (characteristic) farfield boundary condition
// ============================================================================

static void riemannInvariantBC(const double UL[NVAR2D], double nx, double ny,
                               double /*x*/, double /*y*/, double /*t*/,
                               double UR[NVAR2D])
{
    const double gm1 = GAMMA - 1.0;

    double rhoI = UL[0];
    double uI   = UL[1] / rhoI;
    double vI   = UL[2] / rhoI;
    double pI   = pressure2D(UL[0], UL[1], UL[2], UL[3]);
    double cI   = soundSpeed2D(rhoI, pI);
    double VnI  = uI * nx + vI * ny;

    double cInf  = std::sqrt(GAMMA * g_pInf / g_rhoInf);
    double VnInf = g_uInf * nx + g_vInf * ny;

    double Rplus, Rminus;
    double sB, VtB_x, VtB_y;

    if (VnI < 0.0)
    {
        // Inflow: one outgoing characteristic (R+) from interior,
        //         all other info from freestream
        Rplus  = VnI + 2.0 * cI / gm1;
        Rminus = VnInf - 2.0 * cInf / gm1;
        sB     = g_pInf / std::pow(g_rhoInf, GAMMA);
        VtB_x  = g_uInf - VnInf * nx;
        VtB_y  = g_vInf - VnInf * ny;
    }
    else
    {
        // Outflow: one incoming characteristic (R-) from freestream,
        //          all other info from interior
        Rplus  = VnI + 2.0 * cI / gm1;
        Rminus = VnInf - 2.0 * cInf / gm1;
        sB     = pI / std::pow(rhoI, GAMMA);
        VtB_x  = uI - VnI * nx;
        VtB_y  = vI - VnI * ny;
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
// VTK output (unstructured grid)
// ============================================================================

static void writeVTK(const std::string& filename,
                     const Mesh2D& mesh,
                     const GeomData2D& geom,
                     const std::vector<std::vector<double>>& U,
                     int nq1d, int nqVol)
{
    int nE = mesh.nElements;
    int totalPts = nE * nqVol;

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler DG Solution\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    // Points (all quadrature points in physical space)
    out << "POINTS " << totalPts << " double\n";
    for (int i = 0; i < totalPts; ++i)
        out << geom.xPhys[i] << " " << geom.yPhys[i] << " 0.0\n";

    // Cells: each sub-cell is a quad formed by 4 neighbouring quadrature points
    int nSubX = nq1d - 1;
    int nSubY = nq1d - 1;
    int nCells = nE * nSubX * nSubY;
    out << "CELLS " << nCells << " " << nCells * 5 << "\n";
    for (int e = 0; e < nE; ++e)
    {
        int base = e * nqVol;
        for (int ix = 0; ix < nSubX; ++ix)
        {
            for (int iy = 0; iy < nSubY; ++iy)
            {
                int p0 = base + ix * nq1d + iy;
                int p1 = base + (ix + 1) * nq1d + iy;
                int p2 = base + (ix + 1) * nq1d + (iy + 1);
                int p3 = base + ix * nq1d + (iy + 1);
                out << "4 " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
    }

    out << "CELL_TYPES " << nCells << "\n";
    for (int i = 0; i < nCells; ++i)
        out << "9\n";   // VTK_QUAD

    // Point data
    out << "POINT_DATA " << totalPts << "\n";

    // Density
    out << "SCALARS Density double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < totalPts; ++i)
        out << U[0][i] << "\n";

    // Velocity magnitude
    out << "SCALARS VelocityMagnitude double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < totalPts; ++i)
    {
        double u = U[1][i] / U[0][i];
        double v = U[2][i] / U[0][i];
        out << std::sqrt(u * u + v * v) << "\n";
    }

    // Pressure
    out << "SCALARS Pressure double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < totalPts; ++i)
        out << pressure2D(U[0][i], U[1][i], U[2][i], U[3][i]) << "\n";

    // Mach number
    out << "SCALARS Mach double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < totalPts; ++i)
    {
        double u = U[1][i] / U[0][i];
        double v = U[2][i] / U[0][i];
        double p = pressure2D(U[0][i], U[1][i], U[2][i], U[3][i]);
        double c = soundSpeed2D(U[0][i], p);
        out << std::sqrt(u * u + v * v) / c << "\n";
    }

    out.close();
}

// ============================================================================
// Progress bar
// ============================================================================

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
// Compute CFL-based time step
// ============================================================================

static double computeDt_CFL(const Mesh2D& mesh,
                            const GeomData2D& geom,
                            const std::vector<std::vector<double>>& U,
                            int nqVol, double CFL, int P)
{
    double dtMin = 1e20;
    int nE = mesh.nElements;

    for (int e = 0; e < nE; ++e)
    {
        // Estimate element size from Jacobian (use first quad point)
        double det = std::abs(geom.detJ[e * nqVol]);
        double h   = std::sqrt(det);

        for (int q = 0; q < nqVol; ++q)
        {
            int idx = e * nqVol + q;
            double rho  = U[0][idx];
            double u    = U[1][idx] / rho;
            double v    = U[2][idx] / rho;
            double p    = pressure2D(U[0][idx], U[1][idx], U[2][idx], U[3][idx]);
            double c    = soundSpeed2D(rho, p);
            double smax = std::sqrt(u * u + v * v) + c;

            double dt_local = CFL * h / (smax * (2.0 * P + 1.0));
            dtMin = std::min(dtMin, dt_local);
        }
    }
    return dtMin;
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

    // ========================================================================
    // Set up 1D quadrature and basis
    // ========================================================================
    std::vector<double> zq(nq1d, 0.0), wq(nq1d, 0.0);
    if (inp->ptype == "GaussLegendre") {
        zwgl(zq.data(), wq.data(), nq1d)
    } else {
        zwgll(zq.data(), wq.data(), nq1d)
    }

    std::unique_ptr<BasisPoly> basis1D = BasisPoly::Create(inp->btype, P, inp->ptype, zq, wq);
    basis1D->ConstructBasis();

    int nqVol = nq1d * nq1d;
    int nmodes = (P + 1) * (P + 1);

    // Build 2D tensor-product quadrature point arrays
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

    std::vector<double> massLU;
    std::vector<int>    massPiv;
    assembleAndFactorMassMatrices(mesh, geom, Bmat, Bmat, wq, wq,
                                 P, nq1d, massLU, massPiv);

    // ========================================================================
    // Initial conditions
    // ========================================================================
    int nE = mesh.nElements;
    int totalDOF = nE * nqVol;

    std::vector<std::vector<double>> U(NVAR2D);
    for (int v = 0; v < NVAR2D; ++v)
        U[v].resize(totalDOF, 0.0);

    // Set up freestream state from input parameters (used by Riemann invariant BC)
    {
        double AoA_rad = inp->AoA * M_PI / 180.0;
        g_rhoInf = 1.0;
        g_pInf   = 1.0 / (GAMMA * inp->Mach * inp->Mach);
        double cInf = std::sqrt(GAMMA * g_pInf / g_rhoInf);
        g_uInf = inp->Mach * cInf * std::cos(AoA_rad);
        g_vInf = inp->Mach * cInf * std::sin(AoA_rad);
    }

    if (inp->testcase == "IsentropicVortex")
    {
        for (int e = 0; e < nE; ++e)
        {
            for (int q = 0; q < nqVol; ++q)
            {
                int idx = e * nqVol + q;
                double xp = geom.xPhys[idx];
                double yp = geom.yPhys[idx];
                double rho, rhou, rhov, rhoE;
                isentropicVortex(xp, yp, 0.0, rho, rhou, rhov, rhoE);
                U[0][idx] = rho;
                U[1][idx] = rhou;
                U[2][idx] = rhov;
                U[3][idx] = rhoE;
            }
        }
    }
    else if (inp->testcase == "NACA0012")
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

    // Write initial condition
    writeVTK("solution2d_init.vtk", mesh, geom, U, nq1d, nqVol);

    // ========================================================================
    // RK4 time integration
    // ========================================================================
    std::vector<std::vector<double>> R(NVAR2D);
    std::vector<std::vector<double>> k1(NVAR2D), k2(NVAR2D), k3(NVAR2D), k4(NVAR2D);
    std::vector<std::vector<double>> Utmp(NVAR2D);

    for (int v = 0; v < NVAR2D; ++v)
    {
        R[v].resize(totalDOF, 0.0);
        k1[v].resize(totalDOF, 0.0);
        k2[v].resize(totalDOF, 0.0);
        k3[v].resize(totalDOF, 0.0);
        k4[v].resize(totalDOF, 0.0);
        Utmp[v].resize(totalDOF, 0.0);
    }

    // ========================================================================
    // Build boundary condition map
    // ========================================================================
    std::map<int, BoundaryStateFunc> bcMap;
    if (inp->testcase == "IsentropicVortex")
    {
        for (auto& kv : mesh.bcFaces)
            bcMap[kv.first] = isentropicVortexBC;
    }
    else if (inp->testcase == "NACA0012")
    {
        bcMap[1] = slipWallBC;          // Airfoil
        bcMap[2] = riemannInvariantBC;  // Farfield
        bcMap[3] = riemannInvariantBC;  // Wake
    }

    double time = 0.0;
    bool nanFound = false;
    auto tStart = std::chrono::high_resolution_clock::now();

    for (int t_step = 0; t_step < nt && !nanFound; ++t_step)
    {
        // CFL-based dt
        if (CFL > 0.0)
            dt = computeDt_CFL(mesh, geom, U, nqVol, CFL, P);

        // --- Stage 1 ---
        g_currentTime = time;
        computeDGRHS2D(mesh, geom, basis1D.get(), P, nq1d, nq1d,
                       wq, wq, zq, zq, U, R, massLU, massPiv, nmodes,
                       time, bcMap);
        for (int v = 0; v < NVAR2D; ++v)
            for (int i = 0; i < totalDOF; ++i)
            {
                k1[v][i] = dt * R[v][i];
                Utmp[v][i] = U[v][i] + 0.5 * k1[v][i];
            }

        // --- Stage 2 ---
        g_currentTime = time + 0.5 * dt;
        computeDGRHS2D(mesh, geom, basis1D.get(), P, nq1d, nq1d,
                       wq, wq, zq, zq, Utmp, R, massLU, massPiv, nmodes,
                       time + 0.5 * dt, bcMap);
        for (int v = 0; v < NVAR2D; ++v)
            for (int i = 0; i < totalDOF; ++i)
            {
                k2[v][i] = dt * R[v][i];
                Utmp[v][i] = U[v][i] + 0.5 * k2[v][i];
            }

        // --- Stage 3 ---
        computeDGRHS2D(mesh, geom, basis1D.get(), P, nq1d, nq1d,
                       wq, wq, zq, zq, Utmp, R, massLU, massPiv, nmodes,
                       time + 0.5 * dt, bcMap);
        for (int v = 0; v < NVAR2D; ++v)
            for (int i = 0; i < totalDOF; ++i)
            {
                k3[v][i] = dt * R[v][i];
                Utmp[v][i] = U[v][i] + k3[v][i];
            }

        // --- Stage 4 ---
        g_currentTime = time + dt;
        computeDGRHS2D(mesh, geom, basis1D.get(), P, nq1d, nq1d,
                       wq, wq, zq, zq, Utmp, R, massLU, massPiv, nmodes,
                       time + dt, bcMap);
        for (int v = 0; v < NVAR2D; ++v)
            for (int i = 0; i < totalDOF; ++i)
            {
                k4[v][i] = dt * R[v][i];
                U[v][i] += (1.0 / 6.0) * (k1[v][i] + 2.0 * k2[v][i]
                                            + 2.0 * k3[v][i] + k4[v][i]);

                if (std::isnan(U[v][i]))
                {
                    nanFound = true;
                    break;
                }
            }

        time += dt;
        printProgressBar(t_step + 1, nt);
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);

    std::cout << "\nWall-clock time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Final time = " << time << std::endl;

    if (nanFound)
        std::cout << "NaN detected -- simulation terminated early." << std::endl;

    // ========================================================================
    // Compute L2 error for isentropic vortex
    // ========================================================================
    if (inp->testcase == "IsentropicVortex" && !nanFound)
    {
        double l2err_rho = 0.0;
        double l2norm    = 0.0;

        for (int e = 0; e < nE; ++e)
        {
            for (int qx = 0; qx < nq1d; ++qx)
            {
                for (int qe = 0; qe < nq1d; ++qe)
                {
                    int idx = e * nqVol + qx * nq1d + qe;
                    double w = wq[qx] * wq[qe] * geom.detJ[idx];

                    double rhoEx, rhouEx, rhovEx, rhoEEx;
                    isentropicVortex(geom.xPhys[idx], geom.yPhys[idx], time,
                                    rhoEx, rhouEx, rhovEx, rhoEEx);

                    double diff = U[0][idx] - rhoEx;
                    l2err_rho += w * diff * diff;
                    l2norm    += w * rhoEx * rhoEx;
                }
            }
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
    writeVTK("solution2d_final.vtk", mesh, geom, U, nq1d, nqVol);

    std::cout << "Output written to solution2d_init.vtk, solution2d_final.vtk" << std::endl;

    delete inp;
    return 0;
}
