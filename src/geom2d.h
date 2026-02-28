#ifndef GEOM2D_H
#define GEOM2D_H

#include "mesh2d.h"
#include <vector>
#include <array>

// Precomputed geometry data for all elements at all quadrature points
struct GeomData2D {
    int nElements;
    int nqVol;   // number of volume quadrature points per element
    int nqFace;  // number of face quadrature points per face

    // Volume data  – indexed [elem * nqVol + q]
    std::vector<double> detJ;
    std::vector<double> dxidx;   // J^{-1}_{00}
    std::vector<double> dxidy;   // J^{-1}_{01}
    std::vector<double> detadx;  // J^{-1}_{10}
    std::vector<double> detady;  // J^{-1}_{11}
    std::vector<double> xPhys;   // physical x
    std::vector<double> yPhys;   // physical y

    // Face data – indexed [faceIdx * nqFace + q]
    std::vector<double> faceNx;
    std::vector<double> faceNy;
    std::vector<double> faceJac;
    std::vector<double> faceXPhys;
    std::vector<double> faceYPhys;
};

// Bilinear shape functions and their derivatives for a 4-node quad
// N_k(xi, eta), k=0..3
void bilinearShapeFunctions(double xi, double eta,
                            double N[4], double dNdxi[4], double dNdeta[4]);

// Compute all geometry data for every element and face
GeomData2D computeGeometry(const Mesh2D& mesh,
                           const std::vector<double>& xiVol,
                           const std::vector<double>& etaVol,
                           int nqVol,
                           const std::vector<double>& zFace,
                           int nqFace);

// Map reference coordinates to physical for a single element
void refToPhys(const Mesh2D& mesh, int elem,
               double xi, double eta,
               double& xp, double& yp);

#endif
