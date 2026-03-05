#include <stdio.h>
#include "tinyxml.h"
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
#include <iostream>

#ifndef IO_H
#define IO_H

struct Inputs{
    int porder;
    int nelem;
    double dt;
    int nt;
    std::string btype;
    std::string ptype;
    int nquad;
    std::string timescheme;
    int restart;
};

struct Inputs2D {
    int porder     = 2;
    int nquad      = 0;
    double dt      = 1e-3;
    int nt         = 1000;
    double CFL     = 0.0;
    std::string btype       = "Modal";
    std::string ptype       = "GaussLegendre";
    std::string timescheme  = "RK4";
    std::string meshfile;
    std::string testcase    = "IsentropicVortex";
    double Mach             = 0.5;
    double AoA              = 0.0;
    std::string restartfile;
    std::string fluxtype    = "LaxFriedrichs";
    int checkpoint          = 0;
    bool   useAV            = false;
    double AVs0             = 0.0;
    double AVkappa          = 1.0;
    double AVscale          = 1.0;
    int    adjMaxIter       = 10000;
    double adjTol           = 1e-10;
    double adjChordRef      = 1.0;
    bool   adjFDCheck       = false;
    std::string adjObjective = "Lift";
    std::string adjRestartFile;
    bool   runAdjoint       = false;

    // Variable-P (p-adaptivity)
    int    pMin             = 0;   // 0 = disabled, use global porder
    int    pMax             = 0;
    std::string errorIndicatorFile;
    std::vector<double> pAdaptThresholds;
};

void ParseEquals(const std::string &line, std::string &lhs,
                                std::string &rhs);


Inputs*   ReadXmlFile(const char* filename);
Inputs2D* ReadXmlFile2D(const char* filename);

#endif