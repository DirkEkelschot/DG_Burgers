// NACA 0012 airfoil - structured quad C-mesh (4 blocks)
//
// Four-block C-mesh where all blocks meet at the trailing edge.
// Two radial cuts from the TE to the farfield top/bottom split
// each half into an airfoil block and a wake block.  A third
// radial cut from the LE to the upstream farfield separates
// the upper and lower airfoil blocks.
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
progR     = 1.2;     // normal growth ratio (>1 clusters near wall)
progWake  = 1.4;     // wake growth ratio  (>1 clusters near TE)
progAirfoil  = 1.2;     // wake growth ratio  (>1 clusters near TE)

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
// Point(nPts+1) = TE_upper at (1, 0, 0)

For i In {1:nPts-1}
  xc  = 0.5 * (1.0 - Cos(Pi * i / nPts));
  yt  = 0.6 * (0.2969*Sqrt(xc) - 0.1260*xc - 0.3516*xc^2
              + 0.2843*xc^3 - 0.1036*xc^4);
  Point(nPts + 1 + i) = {xc, -yt, 0};       // lower surface
EndFor

pTEL = 2*nPts + 1;
Point(pTEL) = {1, 0, 0};                    // TE_lower (wake slit)

// ============================================================
//  Farfield and wake geometry
// ============================================================
cx    = 0.5;
xExit = cx + R;

pWakeU   = 2*nPts + 2;   Point(pWakeU)   = {xExit, 0,  0};
pWakeL   = 2*nPts + 3;   Point(pWakeL)   = {xExit, 0,  0};
pExitT   = 2*nPts + 4;   Point(pExitT)   = {xExit, R,  0};
pExitB   = 2*nPts + 5;   Point(pExitB)   = {xExit,-R,  0};
pCenter  = 2*nPts + 6;   Point(pCenter)  = {cx,    0,  0};
pSemiT   = 2*nPts + 7;   Point(pSemiT)   = {cx,    R,  0};
pUpstr   = 2*nPts + 8;   Point(pUpstr)   = {cx - R,0,  0};
pSemiB   = 2*nPts + 9;   Point(pSemiB)   = {cx,   -R,  0};

// ============================================================
//  Curves
// ============================================================

// --- airfoil splines ---
Spline(1) = {1 : nPts+1};                         // upper: LE -> TE_upper

lowerPts[] = {1};                                  // lower: LE -> TE_lower
For i In {1:nPts-1}
  lowerPts[] += {nPts + 1 + i};
EndFor
lowerPts[] += {pTEL};
Spline(2) = lowerPts[];

// --- wake ---
Line(3) = {nPts + 1, pWakeU};                     // upper wake
Line(4) = {pTEL,     pWakeL};                     // lower wake

// --- exit ---
Line(5) = {pWakeU, pExitT};                       // upper exit
Line(6) = {pWakeL, pExitB};                       // lower exit

// --- farfield ---
Line(7)    = {pExitT,  pSemiT};                    // upper horizontal
Circle(8)  = {pSemiT,  pCenter, pUpstr};           // upper-left arc
Circle(9)  = {pUpstr,  pCenter, pSemiB};           // lower-left arc
Line(10)   = {pSemiB,  pExitB};                    // lower horizontal

// --- radial cuts ---
Line(11) = {1,        pUpstr};                     // LE -> upstream
Line(12) = {nPts + 1, pSemiT};                    // TE_upper -> farfield top
Line(13) = {pTEL,     pSemiB};                    // TE_lower -> farfield bottom

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

// Block 4: lower wake
Curve Loop(4) = {13, 10, -6, -4};
Plane Surface(4) = {4};

// ============================================================
//  Transfinite (structured) quad meshing
// ============================================================
//
//  All four blocks share the TE as a corner.  Each block is a
//  clean four-sided quad (one curve per side).
//
//  Block 1: LE, TE_upper, pSemiT, pUpstr
//    airfoil (nAirfoil+1)  <->  arc   (nAirfoil+1)
//    TE rad. (nNormal+1)   <->  LE rad. (nNormal+1)
//
//  Block 2: TE_upper, pWakeU, pExitT, pSemiT
//    wake    (nWake+1)     <->  horiz. (nWake+1)
//    exit    (nNormal+1)   <->  TE rad. (nNormal+1)
//
//  Blocks 3, 4: lower-half mirrors

Transfinite Curve {1, 2}      = nAirfoil + 1  Using Bump bumpCoef;
Transfinite Curve {3, 4, 10}  = nWake + 1     Using Progression progWake;
Transfinite Curve {5, 6}      = nNormal + 1   Using Progression progR;
Transfinite Curve {7}         = nWake + 1     Using Progression 1/progWake;
Transfinite Curve {8}         = nAirfoil + 1  Using Progression progAirfoil;
Transfinite Curve {9}         = nAirfoil + 1  Using Progression 1/progAirfoil;
Transfinite Curve {11, 12, 13} = nNormal + 1  Using Progression progR;

Transfinite Surface {1} = {1, nPts+1, pSemiT, pUpstr};
Transfinite Surface {2} = {nPts+1, pWakeU, pExitT, pSemiT};
Transfinite Surface {3} = {1, pUpstr, pSemiB, pTEL};
Transfinite Surface {4} = {pTEL, pSemiB, pExitB, pWakeL};

Recombine Surface {1, 2, 3, 4};

// ============================================================
//  Physical groups
// ============================================================
Physical Curve("Airfoil",  1) = {1, 2};
Physical Curve("Farfield", 2) = {5, 6, 7, 8, 9, 10};
Physical Curve("Wake",     3) = {3, 4};
Physical Surface("Domain", 5) = {1, 2, 3, 4};

// ============================================================
//  Output
// ============================================================
Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder   = 2;
