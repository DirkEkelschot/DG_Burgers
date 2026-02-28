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
};

void ParseEquals(const std::string &line, std::string &lhs,
                                std::string &rhs);


Inputs*   ReadXmlFile(const char* filename);
Inputs2D* ReadXmlFile2D(const char* filename);

#endif