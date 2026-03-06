#include "mesh_deform.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

void MeshDeformer::setup(const Mesh2D& mesh,
                         const std::set<int>& boundaryNodeIDs,
                         double exp)
{
    controlNodeIDs.assign(boundaryNodeIDs.begin(), boundaryNodeIDs.end());
    nControlPts = static_cast<int>(controlNodeIDs.size());
    exponent = exp;

    controlCoords.resize(nControlPts);
    for (int i = 0; i < nControlPts; ++i)
        controlCoords[i] = mesh.nodes[controlNodeIDs[i]];

    std::cout << "MeshDeformer: " << nControlPts
              << " control points, IDW exponent=" << exponent << std::endl;
}

std::array<double, 2> MeshDeformer::interpolate(
    double px, double py,
    const std::vector<std::array<double, 2>>& controlDisplacements) const
{
    double sum_wx = 0.0, sum_wy = 0.0, sum_w = 0.0;

    for (int j = 0; j < nControlPts; ++j) {
        double dx = px - controlCoords[j][0];
        double dy = py - controlCoords[j][1];
        double r = std::sqrt(dx * dx + dy * dy);
        if (r < 1e-14) {
            return controlDisplacements[j];
        }
        double w = 1.0 / std::pow(r, exponent);
        sum_wx += w * controlDisplacements[j][0];
        sum_wy += w * controlDisplacements[j][1];
        sum_w += w;
    }

    if (sum_w < 1e-30) return {0.0, 0.0};
    return {sum_wx / sum_w, sum_wy / sum_w};
}

void deformMesh(Mesh2D& mesh,
                const HicksHenneParam& hh,
                const std::vector<double>& alpha,
                const std::vector<WallNodeInfo>& wallNodes,
                MeshDeformer& deformer,
                const std::vector<std::array<double, 2>>& baselineNodes)
{
    mesh.nodes = baselineNodes;

    // Step 1: Apply bump displacements to wall nodes
    for (const auto& w : wallNodes) {
        double x = baselineNodes[w.nodeID][0];
        if (x <= 0.0 || x >= 1.0) continue;
        double dy = evaluateBumpDisplacement(hh, x, alpha, w.isUpper);
        mesh.nodes[w.nodeID][1] = baselineNodes[w.nodeID][1] + dy;
    }

    // Step 2: Compute boundary displacements
    std::vector<std::array<double, 2>> controlDisp(deformer.nControlPts, {0.0, 0.0});
    for (int i = 0; i < deformer.nControlPts; ++i) {
        int nid = deformer.controlNodeIDs[i];
        controlDisp[i][0] = mesh.nodes[nid][0] - baselineNodes[nid][0];
        controlDisp[i][1] = mesh.nodes[nid][1] - baselineNodes[nid][1];
    }

    // Step 3: IDW interpolation to all interior nodes
    std::set<int> boundarySet(deformer.controlNodeIDs.begin(),
                              deformer.controlNodeIDs.end());

    std::set<int> referencedNodes;
    for (int e = 0; e < mesh.nElements; ++e)
        for (int nid : mesh.elements[e])
            referencedNodes.insert(nid);

    for (int nid : referencedNodes) {
        if (boundarySet.count(nid)) continue;
        double px = baselineNodes[nid][0];
        double py = baselineNodes[nid][1];
        auto disp = deformer.interpolate(px, py, controlDisp);
        mesh.nodes[nid][0] = baselineNodes[nid][0] + disp[0];
        mesh.nodes[nid][1] = baselineNodes[nid][1] + disp[1];
    }

    // Step 4: Re-apply bump displacement to wall nodes
    for (const auto& w : wallNodes) {
        double x = baselineNodes[w.nodeID][0];
        if (x <= 0.0 || x >= 1.0) continue;
        double dy = evaluateBumpDisplacement(hh, x, alpha, w.isUpper);
        mesh.nodes[w.nodeID][1] = baselineNodes[w.nodeID][1] + dy;
    }
}
