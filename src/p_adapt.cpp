#include "p_adapt.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

std::vector<double> readErrorIndicator(const std::string& filename, int nE)
{
    std::vector<double> eta(nE, 0.0);
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("Cannot open error indicator file: " + filename);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int idx;
        double val;
        if (iss >> idx >> val) {
            if (idx >= 0 && idx < nE)
                eta[idx] = val;
        }
    }
    in.close();
    return eta;
}

std::vector<int> assignElementP(const std::vector<double>& eta,
                                int pMin, int pMax,
                                const std::vector<double>& thresholds)
{
    int nE = static_cast<int>(eta.size());
    std::vector<int> elemP(nE, pMin);

    if (pMin == pMax) {
        std::fill(elemP.begin(), elemP.end(), pMin);
        return elemP;
    }

    int nLevels = pMax - pMin + 1;

    // Build effective thresholds (nLevels - 1 boundaries between P levels)
    std::vector<double> thresh = thresholds;
    if (thresh.empty()) {
        for (int i = 1; i < nLevels; ++i)
            thresh.push_back(static_cast<double>(i) / nLevels);
        std::cout << "Using default evenly spaced thresholds:";
        for (double t : thresh) std::cout << " " << t;
        std::cout << std::endl;
    }

    if (static_cast<int>(thresh.size()) != nLevels - 1) {
        std::cerr << "Warning: PAdaptThresholds has " << thresh.size()
                  << " values but expected " << (nLevels - 1)
                  << " for PMin=" << pMin << " PMax=" << pMax
                  << ". Using evenly spaced defaults." << std::endl;
        thresh.clear();
        for (int i = 1; i < nLevels; ++i)
            thresh.push_back(static_cast<double>(i) / nLevels);
    }

    double maxEta = *std::max_element(eta.begin(), eta.end());
    if (maxEta < 1e-15) {
        std::cout << "All error indicators near zero; assigning P=" << pMin << " everywhere." << std::endl;
        return elemP;
    }

    std::cout << "Normalizing error indicator by max(eta) = " << maxEta << std::endl;

    for (int e = 0; e < nE; ++e) {
        double normalized = eta[e] / maxEta;
        int level = 0;
        for (int k = 0; k < static_cast<int>(thresh.size()); ++k) {
            if (normalized > thresh[k])
                level = k + 1;
        }
        elemP[e] = pMin + level;
    }

    // Report distribution
    std::vector<int> counts(nLevels, 0);
    for (int e = 0; e < nE; ++e) counts[elemP[e] - pMin]++;
    std::cout << "Variable-P distribution (normalized thresholds:";
    for (double t : thresh) std::cout << " " << t;
    std::cout << "):" << std::endl;
    for (int p = pMin; p <= pMax; ++p)
        std::cout << "  P=" << p << ": " << counts[p - pMin] << " elements" << std::endl;

    return elemP;
}

std::map<int, PGroupInfo> buildPGroups(const std::vector<int>& elemP,
                                       int pMin, int pMax)
{
    std::map<int, PGroupInfo> groups;

    for (int p = pMin; p <= pMax; ++p) {
        PGroupInfo g;
        g.P      = p;
        g.nq1d   = p + 2;
        g.nmodes = (p + 1) * (p + 1);
        g.nqVol  = g.nq1d * g.nq1d;
        g.nqFace = g.nq1d;
        groups[p] = g;
    }

    int nE = static_cast<int>(elemP.size());
    for (int e = 0; e < nE; ++e) {
        int p = elemP[e];
        groups[p].globalElemIdx.push_back(e);
    }

    // Remove empty groups
    for (auto it = groups.begin(); it != groups.end(); ) {
        if (it->second.globalElemIdx.empty())
            it = groups.erase(it);
        else
            ++it;
    }

    return groups;
}
