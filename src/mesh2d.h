#ifndef MESH2D_H
#define MESH2D_H

#include <vector>
#include <array>
#include <string>
#include <map>

struct Face2D {
    int elemL, elemR;
    int faceL, faceR;
    std::array<int, 2> nodeIDs;
    int bcTag;
};

struct Mesh2D {
    int nNodes    = 0;
    int nElements = 0;
    int nFaces    = 0;
    int geomOrder = 1;
    int nGeomNodes = 4;

    std::vector<std::array<double, 2>> nodes;
    std::vector<std::vector<int>>      elements;
    std::vector<Face2D>                faces;

    // boundary tag -> list of face indices
    std::map<int, std::vector<int>>    bcFaces;

    // per-element list of face indices (4 per quad, ordered bottom/right/top/left)
    std::vector<std::array<int, 4>>    elem2face;

    void readGmsh(const std::string& filename);
    void buildFaces();
};

#endif
