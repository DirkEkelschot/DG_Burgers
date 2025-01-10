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

void ParseEquals(const std::string &line, std::string &lhs,
                                std::string &rhs);


Inputs* ReadXmlFile(const char* filename);

#endif