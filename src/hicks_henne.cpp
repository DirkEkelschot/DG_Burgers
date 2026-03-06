#include "hicks_henne.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>

void HicksHenneParam::setupUniformBumps(int nUpper, int nLower)
{
    nBumpsUpper = nUpper;
    nBumpsLower = nLower;
    peakLocUpper.resize(nUpper);
    peakLocLower.resize(nLower);

    for (int k = 0; k < nUpper; ++k)
        peakLocUpper[k] = (k + 1.0) / (nUpper + 1.0);

    for (int k = 0; k < nLower; ++k)
        peakLocLower[k] = (k + 1.0) / (nLower + 1.0);
}

double nacaThickness(double x, double t)
{
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 0.0;
    return 5.0 * t * (0.2969 * std::sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x * x
                    + 0.2843 * x * x * x
                    - 0.1036 * x * x * x * x);
}

double hicksHenneBump(double x, double peakLoc, double width)
{
    if (x <= 0.0 || x >= 1.0) return 0.0;
    double logArg = std::log(0.5) / std::log(peakLoc);
    double base = std::sin(M_PI * std::pow(x, logArg));
    return std::pow(base, width);
}

double evaluateSurface(const HicksHenneParam& hh,
                       double x,
                       const std::vector<double>& alpha,
                       bool upper)
{
    double yt = nacaThickness(x, hh.thickness);
    double y_base = upper ? yt : -yt;
    return y_base + evaluateBumpDisplacement(hh, x, alpha, upper);
}

double evaluateBumpDisplacement(const HicksHenneParam& hh,
                                double x,
                                const std::vector<double>& alpha,
                                bool upper)
{
    double dy = 0.0;
    if (upper) {
        for (int k = 0; k < hh.nBumpsUpper; ++k)
            dy += alpha[k] * hicksHenneBump(x, hh.peakLocUpper[k], hh.bumpWidth);
    } else {
        int offset = hh.nBumpsUpper;
        for (int k = 0; k < hh.nBumpsLower; ++k)
            dy -= alpha[offset + k] * hicksHenneBump(x, hh.peakLocLower[k], hh.bumpWidth);
    }
    return dy;
}

std::vector<WallNodeInfo> classifyWallNodes(
    const Mesh2D& mesh,
    const std::vector<int>& wallFaceIndices)
{
    std::set<int> wallNodeSet;
    for (int fi : wallFaceIndices) {
        const Face2D& face = mesh.faces[fi];
        wallNodeSet.insert(face.nodeIDs[0]);
        wallNodeSet.insert(face.nodeIDs[1]);

        int eL = face.elemL;
        int lf = face.faceL;
        const auto& elem = mesh.elements[eL];
        if (mesh.geomOrder == 2) {
            int midNode = elem[4 + lf];
            wallNodeSet.insert(midNode);
        }
    }

    std::vector<WallNodeInfo> info;
    info.reserve(wallNodeSet.size());

    for (int nid : wallNodeSet) {
        WallNodeInfo w;
        w.nodeID = nid;
        w.x = mesh.nodes[nid][0];
        double y = mesh.nodes[nid][1];

        if (std::fabs(w.x) < 1e-12 || std::fabs(w.x - 1.0) < 1e-12) {
            w.isUpper = (y >= 0.0);
        } else {
            w.isUpper = (y >= 0.0);
        }

        info.push_back(w);
    }

    return info;
}

void writeGmshMesh(const std::string& filename, const Mesh2D& mesh)
{
    // Collect all node IDs actually referenced by elements and boundary faces
    std::set<int> usedNodes;
    for (int e = 0; e < mesh.nElements; ++e)
        for (int nid : mesh.elements[e])
            usedNodes.insert(nid);
    for (int f = 0; f < mesh.nFaces; ++f)
        if (mesh.faces[f].elemR < 0)
            for (int nid : mesh.faces[f].nodeIDs)
                usedNodes.insert(nid);

    // Build contiguous renumbering: old ID -> new 1-based ID
    std::map<int, int> newID;
    int nextID = 1;
    for (int nid : usedNodes)
        newID[nid] = nextID++;
    int nOutNodes = static_cast<int>(usedNodes.size());

    std::ofstream out(filename);
    out << std::setprecision(17);
    out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";

    out << "$Nodes\n" << nOutNodes << "\n";
    for (int nid : usedNodes)
        out << newID[nid] << " "
            << mesh.nodes[nid][0] << " " << mesh.nodes[nid][1] << " 0\n";
    out << "$EndNodes\n";

    int nBdryEdges = 0;
    for (int f = 0; f < mesh.nFaces; ++f)
        if (mesh.faces[f].elemR < 0) nBdryEdges++;

    int nTotalElements = nBdryEdges + mesh.nElements;
    out << "$Elements\n" << nTotalElements << "\n";

    int eid = 1;
    for (int f = 0; f < mesh.nFaces; ++f) {
        if (mesh.faces[f].elemR < 0) {
            int tag = mesh.faces[f].bcTag;
            if (tag <= 0) tag = 1;
            out << eid << " 1 2 " << tag << " " << tag << " "
                << newID[mesh.faces[f].nodeIDs[0]] << " "
                << newID[mesh.faces[f].nodeIDs[1]] << "\n";
            eid++;
        }
    }

    for (int e = 0; e < mesh.nElements; ++e) {
        const auto& elem = mesh.elements[e];
        if (mesh.geomOrder == 2 && static_cast<int>(elem.size()) == 9) {
            out << eid << " 10 2 5 5";
            for (int k = 0; k < 9; ++k)
                out << " " << newID[elem[k]];
            out << "\n";
        } else {
            out << eid << " 3 2 5 5";
            for (int k = 0; k < 4; ++k)
                out << " " << newID[elem[k]];
            out << "\n";
        }
        eid++;
    }
    out << "$EndElements\n";
    out.close();
}
