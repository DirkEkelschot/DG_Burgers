// NACA 0012 airfoil - structured quad C-mesh (4 blocks)
//
// Four-block C-mesh where all blocks meet at a single trailing-edge
// point.  The wake line from TE to the exit is shared by the upper
// and lower wake blocks, making it an interior face (no slit).
//
// Requires Gmsh 4.x.  Generate with:
//   gmsh naca0012_quad.geo -2 -o naca0012_quad.msh

// ============================================================
//  Tuneable parameters
// ============================================================
nPts      = 80;       // spline control points per airfoil surface
nAirfoil  = 60;       // mesh cells along each airfoil half
nWake     = 20;       // mesh cells in the wake (TE -> exit)
nNormal   = 30;       // cells in wall-normal / radial direction
R         = 100;      // farfield distance (chord lengths)
progR     = 1.2;      // normal growth ratio (>1 clusters near wall)
progWake  = 1.4;      // wake growth ratio  (>1 clusters near TE)
progAirfoil  = 1.2;   // airfoil growth ratio  (>1 clusters near TE)

bumpCoef  = 0.4;      // airfoil Bump coeff (lower = more LE/TE clustering)

// ============================================================
//  NACA 0012 airfoil (closed trailing edge, chord = 1)
// ============================================================

Point(1) = {0, 0, 0};   // leading edge

For i In {1:nPts}
  xc  = 0.5 * (1.0 - Cos(Pi * i / nPts));
  yt  = 0.6 * (0.2969*Sqrt(xc) - 0.1260*xc - 0.3516*xc^2
              + 0.2843*xc^3 - 0.1036*xc^4);
  Point(1 + i) = {xc, yt, 0};               // upper surface
EndFor
// Point(nPts+1) = TE at (1, 0, 0)

pTE = nPts + 1;   // single trailing-edge point (shared upper/lower)

For i In {1:nPts-1}
  xc  = 0.5 * (1.0 - Cos(Pi * i / nPts));
  yt  = 0.6 * (0.2969*Sqrt(xc) - 0.1260*xc - 0.3516*xc^2
              + 0.2843*xc^3 - 0.1036*xc^4);
  Point(nPts + 1 + i) = {xc, -yt, 0};       // lower surface
EndFor

// ============================================================
//  Farfield and wake geometry
// ============================================================
cx    = 0.5;
xExit = cx + R;

pWake    = 2*nPts + 1;   Point(pWake)    = {xExit, 0,  0};
pExitT   = 2*nPts + 2;   Point(pExitT)   = {xExit, R,  0};
pExitB   = 2*nPts + 3;   Point(pExitB)   = {xExit,-R,  0};
pCenter  = 2*nPts + 4;   Point(pCenter)  = {cx,    0,  0};
pSemiT   = 2*nPts + 5;   Point(pSemiT)   = {cx,    R,  0};
pUpstr   = 2*nPts + 6;   Point(pUpstr)   = {cx - R,0,  0};
pSemiB   = 2*nPts + 7;   Point(pSemiB)   = {cx,   -R,  0};

// ============================================================
//  Curves
// ============================================================

// --- airfoil splines ---
Spline(1) = {1 : nPts+1};                         // upper: LE -> TE

lowerPts[] = {1};                                  // lower: LE -> TE
For i In {1:nPts-1}
  lowerPts[] += {nPts + 1 + i};
EndFor
lowerPts[] += {pTE};
Spline(2) = lowerPts[];

// --- wake (single curve shared by upper and lower blocks) ---
Line(3) = {pTE, pWake};

// --- exit ---
Line(5) = {pWake, pExitT};                        // upper exit
Line(6) = {pWake, pExitB};                        // lower exit

// --- farfield ---
Line(7)    = {pExitT,  pSemiT};                    // upper horizontal
Circle(8)  = {pSemiT,  pCenter, pUpstr};           // upper-left arc
Circle(9)  = {pUpstr,  pCenter, pSemiB};           // lower-left arc
Line(10)   = {pSemiB,  pExitB};                    // lower horizontal

// --- radial cuts ---
Line(11) = {1,   pUpstr};                          // LE -> upstream
Line(12) = {pTE, pSemiT};                          // TE -> farfield top
Line(13) = {pTE, pSemiB};                          // TE -> farfield bottom

// ============================================================
//  Surfaces (four structured blocks meeting at the TE)
// ============================================================

// Block 1: upper airfoil
Curve Loop(1) = {1, 12, 8, -11};
Plane Surface(1) = {1};

// Block 2: upper wake
Curve Loop(2) = {3, 5, 7, -12};
Plane Surface(2) = {2};

// Block 3: lower airfoil
Curve Loop(3) = {11, 9, -13, -2};
Plane Surface(3) = {3};

// Block 4: lower wake (uses wake curve 3 in reverse)
Curve Loop(4) = {13, 10, -6, -3};
Plane Surface(4) = {4};

// ============================================================
//  Transfinite (structured) quad meshing
// ============================================================

Transfinite Curve {1, 2}       = nAirfoil + 1  Using Bump bumpCoef;
Transfinite Curve {3, 10}      = nWake + 1     Using Progression progWake;
Transfinite Curve {5, 6}       = nNormal + 1   Using Progression progR;
Transfinite Curve {7}          = nWake + 1     Using Progression 1/progWake;
Transfinite Curve {8}          = nAirfoil + 1  Using Progression progAirfoil;
Transfinite Curve {9}          = nAirfoil + 1  Using Progression 1/progAirfoil;
Transfinite Curve {11, 12, 13} = nNormal + 1   Using Progression progR;

Transfinite Surface {1} = {1, pTE, pSemiT, pUpstr};
Transfinite Surface {2} = {pTE, pWake, pExitT, pSemiT};
Transfinite Surface {3} = {1, pUpstr, pSemiB, pTE};
Transfinite Surface {4} = {pTE, pSemiB, pExitB, pWake};

Recombine Surface {1, 2, 3, 4};

// ============================================================
//  Physical groups
// ============================================================
Physical Curve("Airfoil",  1) = {1, 2};
Physical Curve("Farfield", 2) = {5, 6, 7, 8, 9, 10};
Physical Surface("Domain", 5) = {1, 2, 3, 4};

// ============================================================
//  Output
// ============================================================
Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder   = 2;
