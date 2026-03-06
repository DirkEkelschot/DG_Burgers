#include "mesh2d.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <cassert>
#include <cmath>

// Local face numbering for a quad (CCW node ordering 0-1-2-3):
//   face 0: nodes 0-1 (bottom,  eta = -1)
//   face 1: nodes 1-2 (right,   xi  = +1)
//   face 2: nodes 3-2 (top,     eta = +1)  stored as (2,3) sorted
//   face 3: nodes 0-3 (left,    xi  = -1)
static const int localFaceNodes[4][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}
};

static std::array<int, 2> sortedEdge(int a, int b)
{
    return (a < b) ? std::array<int,2>{a, b} : std::array<int,2>{b, a};
}

void Mesh2D::readGmsh(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("Cannot open mesh file: " + filename);

    std::string line;

    // Collect boundary edges from 1D line elements (physical tags)
    // key: sorted (nodeA, nodeB), value: physical tag
    std::map<std::array<int,2>, int> boundaryEdgeTag;

    while (std::getline(in, line))
    {
        if (line.find("$Nodes") != std::string::npos)
        {
            std::getline(in, line);
            nNodes = std::stoi(line);
            nodes.resize(nNodes);

            for (int i = 0; i < nNodes; ++i)
            {
                std::getline(in, line);
                std::istringstream iss(line);
                int id;
                double x, y, z;
                iss >> id >> x >> y >> z;
                nodes[id - 1] = {x, y};
            }
        }
        else if (line.find("$Elements") != std::string::npos)
        {
            std::getline(in, line);
            int nTotal = std::stoi(line);

            for (int i = 0; i < nTotal; ++i)
            {
                std::getline(in, line);
                std::istringstream iss(line);
                int id, elType, nTags;
                iss >> id >> elType >> nTags;

                std::vector<int> tags(nTags);
                for (int t = 0; t < nTags; ++t)
                    iss >> tags[t];

                int physTag = (nTags > 0) ? tags[0] : 0;

                if (elType == 1)
                {
                    int n1, n2;
                    iss >> n1 >> n2;
                    n1--; n2--;
                    boundaryEdgeTag[sortedEdge(n1, n2)] = physTag;
                }
                else if (elType == 8)
                {
                    int n1, n2, n3;
                    iss >> n1 >> n2 >> n3;
                    n1--; n2--;
                    boundaryEdgeTag[sortedEdge(n1, n2)] = physTag;
                }
                else if (elType == 3)
                {
                    int n1, n2, n3, n4;
                    iss >> n1 >> n2 >> n3 >> n4;
                    n1--; n2--; n3--; n4--;
                    elements.push_back({n1, n2, n3, n4});
                    geomOrder = 1;
                    nGeomNodes = 4;
                }
                else if (elType == 10)
                {
                    int nd[9];
                    for (int k = 0; k < 9; ++k) { iss >> nd[k]; nd[k]--; }
                    elements.push_back({nd[0], nd[1], nd[2], nd[3],
                                        nd[4], nd[5], nd[6], nd[7], nd[8]});
                    geomOrder = 2;
                    nGeomNodes = 9;
                }
                else if (elType == 15)
                {
                    // Point element, skip
                }
            }
            nElements = static_cast<int>(elements.size());
        }
    }
    in.close();

    std::cout << "Mesh2D: read " << nNodes << " nodes, "
              << nElements << " quad elements, "
              << boundaryEdgeTag.size() << " boundary edges" << std::endl;

    checkForCoincidentNodes(1e-10);

    buildFaces();

    // Assign boundary tags from the line elements
    for (auto& f : faces)
    {
        auto key = sortedEdge(f.nodeIDs[0], f.nodeIDs[1]);
        auto it = boundaryEdgeTag.find(key);
        if (it != boundaryEdgeTag.end())
            f.bcTag = it->second;
    }

    // Build bcFaces map
    for (int i = 0; i < nFaces; ++i)
    {
        if (faces[i].bcTag > 0)
            bcFaces[faces[i].bcTag].push_back(i);
    }
}

void Mesh2D::buildFaces()
{
    // Map from sorted edge nodes to face index
    std::map<std::array<int,2>, int> edgeMap;

    elem2face.resize(nElements);

    for (int e = 0; e < nElements; ++e)
    {
        for (int lf = 0; lf < 4; ++lf)
        {
            int n0 = elements[e][localFaceNodes[lf][0]];
            int n1 = elements[e][localFaceNodes[lf][1]];
            auto key = sortedEdge(n0, n1);

            auto it = edgeMap.find(key);
            if (it == edgeMap.end())
            {
                // New face
                Face2D face;
                face.elemL   = e;
                face.elemR   = -1;
                face.faceL   = lf;
                face.faceR   = -1;
                face.nodeIDs = {n0, n1};
                face.bcTag   = 0;

                int fi = static_cast<int>(faces.size());
                faces.push_back(face);
                edgeMap[key] = fi;
                elem2face[e][lf] = fi;
            }
            else
            {
                int fi = it->second;
                faces[fi].elemR = e;
                faces[fi].faceR = lf;
                elem2face[e][lf] = fi;
            }
        }
    }

    nFaces = static_cast<int>(faces.size());
    std::cout << "Mesh2D: built " << nFaces << " faces" << std::endl;
}

void Mesh2D::checkForCoincidentNodes(double tol) const
{
    double scale = 1.0 / std::max(tol, 1e-15);

    std::map<std::pair<long long, long long>, int> posMap;
    int nDuplicates = 0;

    for (int i = 0; i < nNodes; ++i) {
        long long qx = std::llround(nodes[i][0] * scale);
        long long qy = std::llround(nodes[i][1] * scale);
        auto key = std::make_pair(qx, qy);

        auto it = posMap.find(key);
        if (it != posMap.end()) {
            if (nDuplicates == 0)
                std::cerr << "WARNING: mesh contains coincident nodes "
                             "(fix the .geo file so Gmsh shares nodes):\n";
            std::cerr << "  node " << i << " duplicates node " << it->second
                      << " at (" << nodes[i][0] << ", " << nodes[i][1] << ")\n";
            nDuplicates++;
        } else {
            posMap[key] = i;
        }
    }

    if (nDuplicates > 0)
        throw std::runtime_error("Mesh has " + std::to_string(nDuplicates)
            + " coincident nodes. Regenerate the mesh with shared geometry.");
}
