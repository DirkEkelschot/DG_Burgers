#ifndef HICKS_HENNE_H
#define HICKS_HENNE_H

#include "mesh2d.h"
#include <vector>
#include <array>
#include <string>

struct WallNodeInfo {
    int nodeID;
    double x;
    bool isUpper;
};

struct HicksHenneParam {
    int nBumpsUpper;
    int nBumpsLower;
    double thickness;

    std::vector<double> peakLocUpper;
    std::vector<double> peakLocLower;
    double bumpWidth;

    HicksHenneParam() : nBumpsUpper(0), nBumpsLower(0),
                        thickness(0.12), bumpWidth(3.0) {}

    void setupUniformBumps(int nUpper, int nLower);

    int nDesignVars() const { return nBumpsUpper + nBumpsLower; }
};

double nacaThickness(double x, double t);

double hicksHenneBump(double x, double peakLoc, double width);

double evaluateSurface(const HicksHenneParam& hh,
                       double x,
                       const std::vector<double>& alpha,
                       bool upper);

// Compute only the bump displacement (without the baseline profile).
// This is what gets added to the original mesh node positions.
double evaluateBumpDisplacement(const HicksHenneParam& hh,
                                double x,
                                const std::vector<double>& alpha,
                                bool upper);

std::vector<WallNodeInfo> classifyWallNodes(
    const Mesh2D& mesh,
    const std::vector<int>& wallFaceIndices);

void writeGmshMesh(const std::string& filename,
                   const Mesh2D& mesh);

#endif
