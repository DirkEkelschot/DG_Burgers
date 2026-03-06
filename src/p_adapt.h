#ifndef P_ADAPT_H
#define P_ADAPT_H

#include <vector>
#include <map>
#include <string>

struct PGroupInfo {
    int P;
    int nq1d;
    int nmodes;
    int nqVol;
    int nqFace;
    std::vector<int> globalElemIdx;
};

// Read error_indicator.dat: ASCII file with lines "elemIdx eta"
std::vector<double> readErrorIndicator(const std::string& filename, int nE);

// Assign per-element P based on normalized error indicator and thresholds.
// eta is normalized by max(eta); thresholds define the boundaries between P levels.
// If thresholds is empty, evenly spaced defaults are used.
std::vector<int> assignElementP(const std::vector<double>& eta,
                                int pMin, int pMax,
                                const std::vector<double>& thresholds = {});

// Group elements by polynomial order.
// Returns map from P -> PGroupInfo (with globalElemIdx filled, nq1d = P+2).
std::map<int, PGroupInfo> buildPGroups(const std::vector<int>& elemP,
                                       int pMin, int pMax);

#endif
