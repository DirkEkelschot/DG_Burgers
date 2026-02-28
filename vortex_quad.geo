// Isentropic vortex domain [0,10] x [0,10]
// Structured quad mesh

lc = 1.0;   // characteristic length (controls element size)
Nx = 10;    // number of elements in x
Ny = 10;    // number of elements in y

Point(1) = {0,  0,  0, lc};
Point(2) = {10, 0,  0, lc};
Point(3) = {10, 10, 0, lc};
Point(4) = {0,  10, 0, lc};

Line(1) = {1, 2};  // bottom
Line(2) = {2, 3};  // right
Line(3) = {3, 4};  // top
Line(4) = {4, 1};  // left

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Make structured quad mesh
Transfinite Curve {1, 3} = Nx + 1;
Transfinite Curve {2, 4} = Ny + 1;
Transfinite Surface {1};
Recombine Surface {1};

// Physical groups for boundary conditions
Physical Curve("Bottom", 1) = {1};
Physical Curve("Right",  2) = {2};
Physical Curve("Top",    3) = {3};
Physical Curve("Left",   4) = {4};
Physical Surface("Domain", 5) = {1};

// Use Gmsh format 2.2
Mesh.MshFileVersion = 2.2;
