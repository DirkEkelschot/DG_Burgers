#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <cstring>

#include "src/io.h"
#include "src/mesh2d.h"
#include "src/hicks_henne.h"
#include "src/mesh_deform.h"

static std::vector<double> readAlphaFile(const std::string& filename)
{
    std::vector<double> alpha;
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open alpha file " << filename << std::endl;
        return alpha;
    }
    double val;
    while (in >> val)
        alpha.push_back(val);
    in.close();
    return alpha;
}

int main(int argc, char* argv[])
{
    std::string inputFile = "inputs2d.xml";
    std::string alphaFile = "alpha.dat";
    std::string outputFile = "deformed.msh";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--alpha" && i + 1 < argc)
            alphaFile = argv[++i];
        else if (std::string(argv[i]) == "--output" && i + 1 < argc)
            outputFile = argv[++i];
        else if (argv[i][0] != '-')
            inputFile = argv[i];
    }

    Inputs2D* inp = ReadXmlFile2D(inputFile.c_str());
    if (!inp) return 1;

    Mesh2D mesh;
    mesh.readGmsh(inp->meshfile);

    HicksHenneParam hh;
    hh.setupUniformBumps(inp->optNBumpsUpper, inp->optNBumpsLower);
    int nDesign = hh.nDesignVars();

    std::vector<double> alpha = readAlphaFile(alphaFile);
    if (static_cast<int>(alpha.size()) != nDesign) {
        std::cerr << "Error: alpha file has " << alpha.size()
                  << " values, expected " << nDesign << std::endl;
        if (alpha.empty()) {
            std::cout << "Using zero design variables." << std::endl;
            alpha.assign(nDesign, 0.0);
        } else {
            delete inp;
            return 1;
        }
    }

    // Classify wall nodes
    std::vector<int> wallFaceIndices;
    for (int f = 0; f < mesh.nFaces; ++f) {
        if (mesh.faces[f].elemR < 0) {
            int tag = mesh.faces[f].bcTag;
            if (tag == 1) wallFaceIndices.push_back(f);
        }
    }

    auto wallNodes = classifyWallNodes(mesh, wallFaceIndices);
    std::cout << "Wall nodes: " << wallNodes.size() << std::endl;

    auto baselineNodes = mesh.nodes;

    // Setup RBF with all boundary nodes as control points
    std::set<int> boundaryNodeIDs;
    for (int f = 0; f < mesh.nFaces; ++f) {
        if (mesh.faces[f].elemR < 0) {
            boundaryNodeIDs.insert(mesh.faces[f].nodeIDs[0]);
            boundaryNodeIDs.insert(mesh.faces[f].nodeIDs[1]);
            if (mesh.geomOrder == 2) {
                int eL = mesh.faces[f].elemL;
                int lf = mesh.faces[f].faceL;
                int midNode = mesh.elements[eL][4 + lf];
                boundaryNodeIDs.insert(midNode);
            }
        }
    }

    MeshDeformer deformer;
    deformer.setup(mesh, boundaryNodeIDs);

    deformMesh(mesh, hh, alpha, wallNodes, deformer, baselineNodes);

    writeGmshMesh(outputFile, mesh);
    std::cout << "Deformed mesh written to " << outputFile << std::endl;

    delete inp;
    return 0;
}
