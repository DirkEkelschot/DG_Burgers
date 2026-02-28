#include "geom2d.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

// ============================================================================
// 1D Lagrange basis on equispaced nodes in [-1, 1]
// ============================================================================

static inline void lagrange1D(int order, double z,
                               double* L, double* dL)
{
    int n = order + 1;
    double nodes[4];
    for (int i = 0; i < n; ++i)
        nodes[i] = -1.0 + 2.0 * i / order;

    for (int i = 0; i < n; ++i)
    {
        double li = 1.0, dli = 0.0;
        for (int j = 0; j < n; ++j)
        {
            if (j == i) continue;
            double lij = (z - nodes[j]) / (nodes[i] - nodes[j]);
            double dlij = 1.0 / (nodes[i] - nodes[j]);

            dli = dli * lij + li * dlij;
            li *= lij;
        }
        L[i]  = li;
        dL[i] = dli;
    }
}

// ============================================================================
// (i_xi, i_eta) indices for each node in Gmsh ordering
//
// Q1: corners CCW  0=(−1,−1) 1=(+1,−1) 2=(+1,+1) 3=(−1,+1)
//     i_xi:  0, 1, 1, 0      i_eta: 0, 0, 1, 1
//
// Q2: Gmsh 9-node quad layout
//     3---6---2
//     |       |
//     7   8   5
//     |       |
//     0---4---1
//
//     node k  : 0  1  2  3  4  5  6  7  8
//     i_xi    : 0  2  2  0  1  2  1  0  1
//     i_eta   : 0  0  2  2  0  1  2  1  1
// ============================================================================

static const int Q1_ixi [4] = {0, 1, 1, 0};
static const int Q1_ieta[4] = {0, 0, 1, 1};

static const int Q2_ixi [9] = {0, 2, 2, 0, 1, 2, 1, 0, 1};
static const int Q2_ieta[9] = {0, 0, 2, 2, 0, 1, 2, 1, 1};

void geomShapeFunctions(int geomOrder, double xi, double eta,
                        double* N, double* dNdxi, double* dNdeta)
{
    int n1d = geomOrder + 1;
    double Lxi[4], dLxi[4], Leta[4], dLeta[4];
    lagrange1D(geomOrder, xi,  Lxi,  dLxi);
    lagrange1D(geomOrder, eta, Leta, dLeta);

    const int* ixi;
    const int* ieta;
    int nNodes;

    if (geomOrder == 1) {
        ixi  = Q1_ixi;
        ieta = Q1_ieta;
        nNodes = 4;
    } else {
        ixi  = Q2_ixi;
        ieta = Q2_ieta;
        nNodes = 9;
    }

    for (int k = 0; k < nNodes; ++k)
    {
        int ix = ixi[k];
        int ie = ieta[k];
        N[k]      = Lxi[ix]  * Leta[ie];
        dNdxi[k]  = dLxi[ix] * Leta[ie];
        dNdeta[k] = Lxi[ix]  * dLeta[ie];
    }
}

// ============================================================================
// Legacy bilinear interface (unchanged for any callers that still use it)
// ============================================================================

void bilinearShapeFunctions(double xi, double eta,
                            double N[4], double dNdxi[4], double dNdeta[4])
{
    geomShapeFunctions(1, xi, eta, N, dNdxi, dNdeta);
}

// ============================================================================
// Reference to physical coordinate mapping
// ============================================================================

void refToPhys(const Mesh2D& mesh, int elem,
               double xi, double eta,
               double& xp, double& yp)
{
    int nGN = mesh.nGeomNodes;
    double N[9], dNdxi[9], dNdeta[9];
    geomShapeFunctions(mesh.geomOrder, xi, eta, N, dNdxi, dNdeta);

    xp = 0.0;
    yp = 0.0;
    for (int k = 0; k < nGN; ++k)
    {
        xp += N[k] * mesh.nodes[mesh.elements[elem][k]][0];
        yp += N[k] * mesh.nodes[mesh.elements[elem][k]][1];
    }
}

// ============================================================================
// Jacobian computation
// ============================================================================

static void computeJacobian(const Mesh2D& mesh, int e,
                            double xi, double eta,
                            double& dxdxi, double& dxdeta,
                            double& dydxi, double& dydeta,
                            double& det)
{
    int nGN = mesh.nGeomNodes;
    double N[9], dNdxi[9], dNdeta[9];
    geomShapeFunctions(mesh.geomOrder, xi, eta, N, dNdxi, dNdeta);

    dxdxi  = 0.0; dxdeta = 0.0;
    dydxi  = 0.0; dydeta = 0.0;

    for (int k = 0; k < nGN; ++k)
    {
        double xk = mesh.nodes[mesh.elements[e][k]][0];
        double yk = mesh.nodes[mesh.elements[e][k]][1];
        dxdxi  += dNdxi[k]  * xk;
        dxdeta += dNdeta[k] * xk;
        dydxi  += dNdxi[k]  * yk;
        dydeta += dNdeta[k] * yk;
    }

    det = dxdxi * dydeta - dxdeta * dydxi;
}

// Reference-space coordinates on each local face, parameterised by s in [-1,1].
static void faceRefCoords(int localFace, double s, double& xi, double& eta)
{
    switch (localFace)
    {
        case 0: xi =  s;   eta = -1.0; break;
        case 1: xi =  1.0; eta =  s;   break;
        case 2: xi = -s;   eta =  1.0; break;
        case 3: xi = -1.0; eta = -s;   break;
        default: xi = 0; eta = 0; break;
    }
}

// ============================================================================
// Precompute geometry for all elements and faces
// ============================================================================

GeomData2D computeGeometry(const Mesh2D& mesh,
                           const std::vector<double>& xiVol,
                           const std::vector<double>& etaVol,
                           int nqVol,
                           const std::vector<double>& zFace,
                           int nqFace)
{
    GeomData2D gd;
    gd.nElements = mesh.nElements;
    gd.nqVol     = nqVol;
    gd.nqFace    = nqFace;

    int nE = mesh.nElements;

    gd.detJ.resize(nE * nqVol);
    gd.dxidx.resize(nE * nqVol);
    gd.dxidy.resize(nE * nqVol);
    gd.detadx.resize(nE * nqVol);
    gd.detady.resize(nE * nqVol);
    gd.xPhys.resize(nE * nqVol);
    gd.yPhys.resize(nE * nqVol);

    for (int e = 0; e < nE; ++e)
    {
        for (int q = 0; q < nqVol; ++q)
        {
            double xi  = xiVol[q];
            double eta = etaVol[q];

            double dxdxi, dxdeta, dydxi, dydeta, det;
            computeJacobian(mesh, e, xi, eta, dxdxi, dxdeta, dydxi, dydeta, det);

            if (det <= 0.0)
            {
                std::cerr << "WARNING: non-positive Jacobian det=" << det
                          << " in element " << e << " at q=" << q << std::endl;
            }

            int idx = e * nqVol + q;
            gd.detJ[idx]   = det;
            gd.dxidx[idx]  =  dydeta / det;
            gd.dxidy[idx]  = -dxdeta / det;
            gd.detadx[idx] = -dydxi  / det;
            gd.detady[idx] =  dxdxi  / det;

            double xp, yp;
            refToPhys(mesh, e, xi, eta, xp, yp);
            gd.xPhys[idx] = xp;
            gd.yPhys[idx] = yp;
        }
    }

    // Face geometry
    int nF = mesh.nFaces;
    gd.faceNx.resize(nF * nqFace);
    gd.faceNy.resize(nF * nqFace);
    gd.faceJac.resize(nF * nqFace);
    gd.faceXPhys.resize(nF * nqFace);
    gd.faceYPhys.resize(nF * nqFace);

    for (int f = 0; f < nF; ++f)
    {
        int eL = mesh.faces[f].elemL;
        int lf = mesh.faces[f].faceL;

        for (int q = 0; q < nqFace; ++q)
        {
            double s = zFace[q];
            double xi, eta;
            faceRefCoords(lf, s, xi, eta);

            double dxdxi, dxdeta, dydxi, dydeta, det;
            computeJacobian(mesh, eL, xi, eta, dxdxi, dxdeta, dydxi, dydeta, det);

            double dxids, detads;
            switch (lf)
            {
                case 0: dxids = 1.0; detads = 0.0; break;
                case 1: dxids = 0.0; detads = 1.0; break;
                case 2: dxids = -1.0; detads = 0.0; break;
                case 3: dxids = 0.0; detads = -1.0; break;
                default: dxids = 0; detads = 0; break;
            }

            double dxds = dxdxi * dxids + dxdeta * detads;
            double dyds = dydxi * dxids + dydeta * detads;

            double nx =  dyds;
            double ny = -dxds;
            double mag = std::sqrt(nx * nx + ny * ny);

            int idx = f * nqFace + q;
            gd.faceNx[idx]  = nx / mag;
            gd.faceNy[idx]  = ny / mag;
            gd.faceJac[idx] = mag;

            double xp, yp;
            refToPhys(mesh, eL, xi, eta, xp, yp);
            gd.faceXPhys[idx] = xp;
            gd.faceYPhys[idx] = yp;
        }
    }

    return gd;
}
