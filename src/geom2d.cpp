#include "geom2d.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

void bilinearShapeFunctions(double xi, double eta,
                            double N[4], double dNdxi[4], double dNdeta[4])
{
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

    dNdxi[0] = -0.25 * (1.0 - eta);
    dNdxi[1] =  0.25 * (1.0 - eta);
    dNdxi[2] =  0.25 * (1.0 + eta);
    dNdxi[3] = -0.25 * (1.0 + eta);

    dNdeta[0] = -0.25 * (1.0 - xi);
    dNdeta[1] = -0.25 * (1.0 + xi);
    dNdeta[2] =  0.25 * (1.0 + xi);
    dNdeta[3] =  0.25 * (1.0 - xi);
}

void refToPhys(const Mesh2D& mesh, int elem,
               double xi, double eta,
               double& xp, double& yp)
{
    double N[4], dNdxi[4], dNdeta[4];
    bilinearShapeFunctions(xi, eta, N, dNdxi, dNdeta);

    xp = 0.0;
    yp = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        xp += N[k] * mesh.nodes[mesh.elements[elem][k]][0];
        yp += N[k] * mesh.nodes[mesh.elements[elem][k]][1];
    }
}

// Compute Jacobian at a point inside element e
static void computeJacobian(const Mesh2D& mesh, int e,
                            double xi, double eta,
                            double& dxdxi, double& dxdeta,
                            double& dydxi, double& dydeta,
                            double& det)
{
    double N[4], dNdxi[4], dNdeta[4];
    bilinearShapeFunctions(xi, eta, N, dNdxi, dNdeta);

    dxdxi  = 0.0; dxdeta = 0.0;
    dydxi  = 0.0; dydeta = 0.0;

    for (int k = 0; k < 4; ++k)
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
// Returns (xi, eta) on the face boundary.
// Local face numbering:
//   face 0: eta=-1,  xi  = s        (bottom)
//   face 1: xi=+1,   eta = s        (right)
//   face 2: eta=+1,  xi  = -s       (top, reversed so normal points outward)
//   face 3: xi=-1,   eta = -s       (left, reversed)
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
