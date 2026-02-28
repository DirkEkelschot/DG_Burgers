#ifndef EULER2D_H
#define EULER2D_H

#include "mesh2d.h"
#include "geom2d.h"
#include "basis_poly.h"
#include <vector>
#include <array>
#include <map>
#include <cmath>

constexpr int NVAR2D = 4;   // rho, rho*u, rho*v, rho*E
constexpr double GAMMA = 1.4;

// State at a single point
struct State2D {
    double rho, rhou, rhov, rhoE;
};

// Compute pressure from conservative state
inline double pressure2D(double rho, double rhou, double rhov, double rhoE)
{
    double u = rhou / rho;
    double v = rhov / rho;
    return (GAMMA - 1.0) * (rhoE - 0.5 * rho * (u * u + v * v));
}

// Compute sound speed
inline double soundSpeed2D(double rho, double p)
{
    return std::sqrt(GAMMA * p / rho);
}

// Evaluate the x-flux F(U) at a single point
void eulerFluxX(double rho, double rhou, double rhov, double rhoE,
                double F[NVAR2D]);

// Evaluate the y-flux G(U) at a single point
void eulerFluxY(double rho, double rhou, double rhov, double rhoE,
                double G[NVAR2D]);

// Lax-Friedrichs numerical flux in normal direction (nx, ny)
void laxFriedrichsFlux2D(const double UL[NVAR2D], const double UR[NVAR2D],
                         double nx, double ny,
                         double Fnum[NVAR2D]);

// Boundary condition callback: given interior state, outward normal, and position,
// compute the ghost (exterior) state UR.
typedef void (*BoundaryStateFunc)(const double UL[NVAR2D], double nx, double ny,
                                  double x, double y, double t,
                                  double UR[NVAR2D]);

// Full DG RHS computation
// U: solution stored as U[var][elem * nqVol + q]
// R: output RHS stored as R[var][elem * nqVol + q]
void computeDGRHS2D(const Mesh2D& mesh,
                    const GeomData2D& geom,
                    BasisPoly* basis1D,
                    int P, int nq1d, int nqFace,
                    const std::vector<double>& wq1d,
                    const std::vector<double>& wqFace,
                    const std::vector<double>& zq1d,
                    const std::vector<double>& zqFace,
                    const std::vector<std::vector<double>>& U,
                    std::vector<std::vector<double>>& R,
                    const std::vector<double>& massLU,
                    const std::vector<int>& massPiv,
                    int nModesPerElem,
                    double time = 0.0,
                    const std::map<int, BoundaryStateFunc>& bcMap = {});

// Assemble the per-element mass matrices into a block-diagonal structure
// and LU-factor each block. Returns flat array and pivot array.
void assembleAndFactorMassMatrices(
    const Mesh2D& mesh,
    const GeomData2D& geom,
    const std::vector<std::vector<double>>& Bxi,
    const std::vector<std::vector<double>>& Beta,
    const std::vector<double>& wxi,
    const std::vector<double>& weta,
    int P, int nq1d,
    std::vector<double>& massLU,
    std::vector<int>& massPiv);

// Compute dense M^{-1} from the LU-factored mass matrices (for GPU solver)
void computeMassInverse(
    const std::vector<double>& massLU,
    const std::vector<int>& massPiv,
    int nE, int nmodes,
    std::vector<double>& Minv);

#endif
