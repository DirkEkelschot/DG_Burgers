#ifndef MESH_DEFORM_H
#define MESH_DEFORM_H

#include "mesh2d.h"
#include "hicks_henne.h"
#include <vector>
#include <set>

struct MeshDeformer {
    int nControlPts;
    std::vector<int> controlNodeIDs;
    std::vector<std::array<double, 2>> controlCoords;
    double exponent;

    void setup(const Mesh2D& mesh,
               const std::set<int>& boundaryNodeIDs,
               double exp = 3.0);

    std::array<double, 2> interpolate(
        double px, double py,
        const std::vector<std::array<double, 2>>& controlDisplacements) const;
};

void deformMesh(Mesh2D& mesh,
                const HicksHenneParam& hh,
                const std::vector<double>& alpha,
                const std::vector<WallNodeInfo>& wallNodes,
                MeshDeformer& deformer,
                const std::vector<std::array<double, 2>>& baselineNodes);

#endif
