#!/usr/bin/env python3
"""
Transonic NACA 0012 shape optimization driver.

Replicates the manual design optimization process:
  1. Generate baseline mesh (gmsh)
  2. Run forward Euler solver (app2d)
  3. Run discrete adjoint solver (app2d_discrete_adjoint)
  4. Compute shape gradient (shape_grad)
  5. Update design variables via steepest descent
  6. Deform mesh and repeat from step 2

Usage:
    python run_optimization.py [options]

    python run_optimization.py --max-iter 4 --step 0.01
    python run_optimization.py --restart-iter 2  # resume from iteration 2
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
#  Solver / flow parameters (matching the transonic_test configuration)
# ---------------------------------------------------------------------------
POLY_ORDER = 3
N_QUAD = 5
CFL = 0.60
DT = 1.0e-4
MACH = 0.8
AOA = 1.25
FLUX = "HLLC"
BASIS = "Modal"
POINTS = "GaussLegendre"
TIME_SCHEME = "RK4"
TEST_CASE = "NACA0012"
CHECKPOINT_INTERVAL = 50000

# Forward solver
FORWARD_NT = 2000000
FWD_AV_SCALE = 6.0
FWD_AV_S0 = -4.0
FWD_AV_KAPPA = 1.5
FWD_AV_FREEZE_AFTER = 900000000

# Adjoint solver
ADJOINT_MAX_ITER = 2000000
ADJOINT_TOL = 1e-10
ADJOINT_OBJECTIVE = "LiftOverDrag"
ADJ_AV_SCALE = 6.0
ADJ_AV_S0 = -4.0
ADJ_AV_KAPPA = 1.5
# Shape parameterisation
N_BUMPS_UPPER = 10
N_BUMPS_LOWER = 10
FD_EPSILON = 1e-7
CHORD_REF = 1.0

MAXIMIZE_OBJECTIVES = {"LiftOverDrag"}
MAXIMIZE = ADJOINT_OBJECTIVE in MAXIMIZE_OBJECTIVES
OPT_SIGN = 1.0 if MAXIMIZE else -1.0  # +1 = ascent (maximize), -1 = descent (minimize)


# ---------------------------------------------------------------------------
#  XML helpers
# ---------------------------------------------------------------------------

def _param_line(key, val):
    return f"        <P> {key} = {val} </P>\n"


def write_forward_xml(path, mesh_file, restart_file=""):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write("<DGSOLVER>\n    <PARAMETERS>\n")
        f.write(_param_line("PolynomialOrder", POLY_ORDER))
        f.write(_param_line("nQuadrature", N_QUAD))
        f.write(_param_line("CFL", CFL))
        f.write(_param_line("dt", DT))
        f.write(_param_line("nt", FORWARD_NT))
        f.write(_param_line("BasisType", BASIS))
        f.write(_param_line("PointsType", POINTS))
        f.write(_param_line("TimeScheme", TIME_SCHEME))
        f.write(_param_line("MeshFile", mesh_file))
        f.write(_param_line("TestCase", TEST_CASE))
        f.write(_param_line("FluxType", FLUX))
        f.write(_param_line("Mach", MACH))
        f.write(_param_line("AoA", AOA))
        f.write(_param_line("ArtificialViscosity", "true"))
        f.write(_param_line("AVscale", FWD_AV_SCALE))
        f.write(_param_line("AVs0", FWD_AV_S0))
        f.write(_param_line("AVkappa", FWD_AV_KAPPA))
        f.write(_param_line("AVfreezeAfter", FWD_AV_FREEZE_AFTER))
        f.write(_param_line("CheckpointInterval", CHECKPOINT_INTERVAL))
        f.write(_param_line("RestartFile", restart_file))
        f.write("    </PARAMETERS>\n</DGSOLVER>\n")


def write_adjoint_xml(path, mesh_file):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write("<DGSOLVER>\n    <PARAMETERS>\n")
        f.write(_param_line("PolynomialOrder", POLY_ORDER))
        f.write(_param_line("nQuadrature", N_QUAD))
        f.write(_param_line("CFL", CFL))
        f.write(_param_line("dt", DT))
        f.write(_param_line("nt", FORWARD_NT))
        f.write(_param_line("BasisType", BASIS))
        f.write(_param_line("RestartFile", "restart2d.bin"))
        f.write(_param_line("PointsType", POINTS))
        f.write(_param_line("TimeScheme", TIME_SCHEME))
        f.write(_param_line("MeshFile", mesh_file))
        f.write(_param_line("TestCase", TEST_CASE))
        f.write(_param_line("FluxType", FLUX))
        f.write(_param_line("Mach", MACH))
        f.write(_param_line("AoA", AOA))
        f.write(_param_line("ArtificialViscosity", "true"))
        f.write(_param_line("AVscale", ADJ_AV_SCALE))
        f.write(_param_line("AVs0", ADJ_AV_S0))
        f.write(_param_line("AVkappa", ADJ_AV_KAPPA))
        f.write(_param_line("RunAdjoint", "true"))
        f.write(_param_line("AdjointObjective", ADJOINT_OBJECTIVE))
        f.write(_param_line("AdjointMaxIter", ADJOINT_MAX_ITER))
        f.write(_param_line("AdjointTolerance", ADJOINT_TOL))
        f.write(_param_line("CheckpointInterval", CHECKPOINT_INTERVAL))
        f.write("    </PARAMETERS>\n</DGSOLVER>\n")


def write_deform_xml(path, base_mesh_file):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write("<DGSOLVER>\n    <PARAMETERS>\n")
        f.write(_param_line("PolynomialOrder", POLY_ORDER))
        f.write(_param_line("nQuadrature", N_QUAD))
        f.write(_param_line("CFL", CFL))
        f.write(_param_line("dt", DT))
        f.write(_param_line("nt", FORWARD_NT))
        f.write(_param_line("BasisType", BASIS))
        f.write(_param_line("PointsType", POINTS))
        f.write(_param_line("TimeScheme", TIME_SCHEME))
        f.write(_param_line("MeshFile", base_mesh_file))
        f.write(_param_line("TestCase", TEST_CASE))
        f.write(_param_line("FluxType", FLUX))
        f.write(_param_line("Mach", MACH))
        f.write(_param_line("AoA", AOA))
        f.write(_param_line("ArtificialViscosity", "true"))
        f.write(_param_line("AVscale", FWD_AV_SCALE))
        f.write(_param_line("AVs0", FWD_AV_S0))
        f.write(_param_line("AVkappa", FWD_AV_KAPPA))
        f.write(_param_line("OptNBumpsUpper", N_BUMPS_UPPER))
        f.write(_param_line("OptNBumpsLower", N_BUMPS_LOWER))
        f.write("    </PARAMETERS>\n</DGSOLVER>\n")


def write_shape_grad_xml(path, mesh_file, base_mesh_file):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write("<DGSOLVER>\n    <PARAMETERS>\n")
        f.write(_param_line("PolynomialOrder", POLY_ORDER))
        f.write(_param_line("nQuadrature", N_QUAD))
        f.write(_param_line("CFL", CFL))
        f.write(_param_line("dt", DT))
        f.write(_param_line("nt", FORWARD_NT))
        f.write(_param_line("BasisType", BASIS))
        f.write(_param_line("PointsType", POINTS))
        f.write(_param_line("TimeScheme", TIME_SCHEME))
        f.write(_param_line("MeshFile", mesh_file))
        f.write(_param_line("TestCase", TEST_CASE))
        f.write(_param_line("FluxType", FLUX))
        f.write(_param_line("Mach", MACH))
        f.write(_param_line("AoA", AOA))
        f.write(_param_line("ArtificialViscosity", "true"))
        f.write(_param_line("AVscale", FWD_AV_SCALE))
        f.write(_param_line("AVs0", FWD_AV_S0))
        f.write(_param_line("AVkappa", FWD_AV_KAPPA))
        f.write(_param_line("RestartFile", "restart2d.bin"))
        f.write(_param_line("AdjointObjective", ADJOINT_OBJECTIVE))
        f.write(_param_line("AdjointRestartFile", "discrete_adjoint_restart.bin"))
        f.write(_param_line("AdjointChordRef", CHORD_REF))
        f.write(_param_line("BaseMeshFile", base_mesh_file))
        f.write(_param_line("OptNBumpsUpper", N_BUMPS_UPPER))
        f.write(_param_line("OptNBumpsLower", N_BUMPS_LOWER))
        f.write(_param_line("OptFDEpsilon", FD_EPSILON))
        f.write("    </PARAMETERS>\n</DGSOLVER>\n")


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def write_alpha(path, alpha):
    np.savetxt(path, alpha, fmt="%.15e")


def read_alpha(path):
    return np.loadtxt(path)


def read_gradient(path):
    return np.loadtxt(path)


def read_objective(path):
    with open(path) as f:
        return float(f.read().strip())


def run(cmd, label, cwd):
    """Execute a subprocess, stream output summary, and return (success, stdout)."""
    print(f"\n  [{label}] {' '.join(cmd)}")
    print(f"           cwd = {cwd}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  ** {label} FAILED (exit code {result.returncode}) **")
        tail = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
        if tail:
            print(tail)
        tail = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        if tail:
            print(tail)
        return False, result.stdout
    lines = result.stdout.strip().split("\n") if result.stdout else []
    for line in lines[-5:]:
        print(f"    {line}")
    return True, result.stdout


import re

_CL_RE = re.compile(r"Lift coefficient \(Cl\)\s*=\s*([0-9eE.+\-]+)")
_CD_RE = re.compile(r"Drag coefficient \(Cd\)\s*=\s*([0-9eE.+\-]+)")


def parse_cl_cd_from_output(stdout):
    """Extract lift and drag coefficients from forward solver stdout."""
    cl_match = _CL_RE.search(stdout)
    cd_match = _CD_RE.search(stdout)
    cl = float(cl_match.group(1)) if cl_match else None
    cd = float(cd_match.group(1)) if cd_match else None
    return cl, cd


# ---------------------------------------------------------------------------
#  Main optimisation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transonic NACA 0012 shape optimisation driver")
    parser.add_argument("--max-iter", type=int, default=4,
                        help="Number of design iterations (default: 4)")
    parser.add_argument("--step", type=float, default=0.0025,
                        help="Default steepest-descent step size (default: 0.0025)")
    parser.add_argument("--initial-step", type=float, default=None,
                        help="Step size for the first iteration only "
                             "(default: same as --step)")
    parser.add_argument("--step-schedule", type=float, nargs="+", default=None,
                        help="Per-iteration step sizes, e.g. --step-schedule 0.5 0.25 0.1 0.05. "
                             "Overrides --step and --initial-step. If fewer values than iterations, "
                             "the last value is repeated.")
    parser.add_argument("--max-backtrack", type=int, default=5,
                        help="Max backtracking halvings when objective increases (default: 5)")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Gradient norm convergence tolerance")
    parser.add_argument("--build-dir", default="..",
                        help="Directory containing executables (default: ..)")
    parser.add_argument("--work-dir", default=".",
                        help="Working directory for all outputs (default: .)")
    parser.add_argument("--geo-file", default="naca0012_quad.geo",
                        help="Gmsh geometry file for the baseline mesh")
    parser.add_argument("--restart-iter", type=int, default=0,
                        help="Resume from this iteration (reads alpha_iterN.dat)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline forward/adjoint (assume sim0 exists)")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    build_dir = os.path.abspath(args.build_dir)

    app2d = os.path.join(build_dir, "app2d")
    adjoint_exe = os.path.join(build_dir, "app2d_discrete_adjoint")
    shape_grad_exe = os.path.join(build_dir, "shape_grad")
    deform_mesh_exe = os.path.join(build_dir, "deform_mesh")

    for exe in [app2d, adjoint_exe, shape_grad_exe, deform_mesh_exe]:
        if not os.path.exists(exe):
            print(f"Error: executable not found: {exe}")
            sys.exit(1)

    os.makedirs(work_dir, exist_ok=True)

    n_design = N_BUMPS_UPPER + N_BUMPS_LOWER
    geo_file = os.path.join(work_dir, args.geo_file)
    base_mesh = os.path.join(work_dir, "naca0012_quad.msh")

    # ------------------------------------------------------------------
    #  Step 0 — Generate baseline mesh
    # ------------------------------------------------------------------
    if not os.path.exists(base_mesh):
        print("=" * 60)
        print("  Generating baseline mesh from", geo_file)
        print("=" * 60)
        ok, _ = run([
            "python3", "-c",
            ("import gmsh, sys; gmsh.initialize(sys.argv); "
             f"gmsh.open('{geo_file}'); "
             "gmsh.model.mesh.generate(2); "
             f"gmsh.write('{base_mesh}'); "
             "gmsh.finalize()")
        ], "gmsh", cwd=work_dir)
        if not ok or not os.path.exists(base_mesh):
            print("Mesh generation failed!")
            sys.exit(1)
    else:
        print(f"Baseline mesh already exists: {base_mesh}")

    base_mesh = os.path.abspath(base_mesh)

    # ------------------------------------------------------------------
    #  Step 1 — Baseline solve  (iteration 0: alpha = 0)
    # ------------------------------------------------------------------
    alpha = np.zeros(n_design)
    alpha_file = os.path.join(work_dir, "alpha.dat")
    gradient_file = os.path.join(work_dir, "gradient.dat")
    objective_file = os.path.join(work_dir, "objective.dat")
    history_file = os.path.join(work_dir, "opt_history.csv")

    def get_step(iteration):
        """Return the step size for a given 1-based iteration index."""
        if args.step_schedule is not None:
            idx = min(iteration - 1, len(args.step_schedule) - 1)
            return args.step_schedule[idx]
        if iteration == 1 and args.initial_step is not None:
            return args.initial_step
        return args.step

    initial_step = get_step(1)

    if args.restart_iter > 0:
        prev_iter = args.restart_iter - 1
        if prev_iter == 0:
            prev_grad_file = gradient_file
            prev_obj_file = objective_file
        else:
            prev_dir = os.path.join(work_dir, f"iter{prev_iter}")
            prev_grad_file = os.path.join(prev_dir, "gradient.dat")
            prev_obj_file = os.path.join(prev_dir, "objective.dat")

        prev_alpha_file = os.path.join(
            work_dir, f"alpha_iter{prev_iter}.dat") if prev_iter > 0 else None

        for needed in [prev_grad_file, prev_obj_file]:
            if not os.path.exists(needed):
                print(f"Error: {needed} not found for restart")
                sys.exit(1)

        prev_gradient = read_gradient(prev_grad_file)
        prev_objective_val = read_objective(prev_obj_file)
        prev_grad_norm = np.linalg.norm(prev_gradient)
        prev_direction_vec = OPT_SIGN * prev_gradient / prev_grad_norm

        if prev_alpha_file and os.path.exists(prev_alpha_file):
            prev_alpha_vec = read_alpha(prev_alpha_file)
        else:
            prev_alpha_vec = np.zeros(n_design)

        step_size = get_step(args.restart_iter)
        alpha = prev_alpha_vec + step_size * prev_direction_vec

        print(f"Restarting at iteration {args.restart_iter}")
        print(f"  prev objective ({ADJOINT_OBJECTIVE})  = {prev_objective_val:.10e}")
        print(f"  prev |gradient|      = {prev_grad_norm:.6e}")
        print(f"  step size            = {step_size:.6e}")

    if args.restart_iter == 0 and not args.skip_baseline:
        print("\n" + "=" * 60)
        print("  BASELINE  (iteration 0, alpha = 0)")
        print("=" * 60)

        write_alpha(alpha_file, alpha)

        fwd_xml = os.path.join(work_dir, "inputs_forward_p3.xml")
        write_forward_xml(fwd_xml, "naca0012_quad.msh", restart_file="")

        ok, _ = run([app2d, fwd_xml], "forward (baseline)", cwd=work_dir)
        if not ok:
            print("Baseline forward solve failed!")
            sys.exit(1)

        adj_xml = os.path.join(work_dir, "inputs_adjoint.xml")
        write_adjoint_xml(adj_xml, "naca0012_quad.msh")

        ok, _ = run([adjoint_exe, adj_xml], "adjoint (baseline)", cwd=work_dir)
        if not ok:
            print("Baseline adjoint solve failed!")
            sys.exit(1)

        sg_xml = os.path.join(work_dir, "inputs_shape_grad.xml")
        write_shape_grad_xml(sg_xml, "naca0012_quad.msh", base_mesh)

        ok, _ = run([
            shape_grad_exe, sg_xml,
            "--alpha", alpha_file,
            "--gradient", os.path.abspath(gradient_file),
            "--objective", os.path.abspath(objective_file),
        ], "shape_grad (baseline)", cwd=work_dir)
        if not ok:
            print("Baseline shape gradient failed!")
            sys.exit(1)

        gradient = read_gradient(gradient_file)
        objective = read_objective(objective_file)
        grad_norm = np.linalg.norm(gradient)

        print(f"\n  Baseline objective ({ADJOINT_OBJECTIVE}) = {objective:.10e}")
        print(f"  |gradient|              = {grad_norm:.6e}")

        # Save baseline results into sim0/
        sim0 = os.path.join(work_dir, "sim0")
        os.makedirs(sim0, exist_ok=True)
        for fn in os.listdir(work_dir):
            src = os.path.join(work_dir, fn)
            if os.path.isfile(src) and fn not in (
                "run_optimization.py", "plot_cd_vs_iteration.py",
                "gen_nacamesh.sh", "opt_history.csv",
            ) and not fn.startswith("alpha_iter"):
                dst = os.path.join(sim0, fn)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)

        # Start history CSV
        with open(history_file, "w", newline="") as hf:
            w = csv.writer(hf)
            w.writerow(["iter", "objective", "grad_norm", "step_size"]
                       + [f"alpha_{k}" for k in range(n_design)]
                       + [f"grad_{k}" for k in range(n_design)])
            w.writerow([0, objective, grad_norm, 0.0]
                       + list(alpha) + list(gradient))

        direction = OPT_SIGN * gradient / grad_norm
        prev_objective = objective
        prev_alpha = alpha.copy()
        prev_direction = direction.copy()
        arrival_step = initial_step
        alpha = alpha + initial_step * direction

    elif args.restart_iter > 0:
        prev_objective = prev_objective_val
        prev_alpha = prev_alpha_vec.copy()
        prev_direction = prev_direction_vec.copy()
        arrival_step = step_size
    else:
        # --skip-baseline: load existing baseline gradient
        gradient = read_gradient(gradient_file)
        objective = read_objective(objective_file)
        grad_norm = np.linalg.norm(gradient)
        direction = OPT_SIGN * gradient / grad_norm
        prev_objective = objective
        prev_alpha = alpha.copy()
        prev_direction = direction.copy()
        arrival_step = initial_step
        alpha = alpha + initial_step * direction

    # ------------------------------------------------------------------
    #  Design iterations
    # ------------------------------------------------------------------
    start = max(args.restart_iter, 1)

    print("\n" + "=" * 60)
    opt_verb = "MAXIMISING" if MAXIMIZE else "MINIMISING"
    print(f"  DESIGN OPTIMISATION — {opt_verb} {ADJOINT_OBJECTIVE}  ({args.max_iter} iterations)")
    if args.step_schedule is not None:
        print(f"  Step schedule = {args.step_schedule}")
    else:
        print(f"  Initial step = {initial_step},  step = {args.step}")
    print("=" * 60)

    for iteration in range(start, start + args.max_iter):
        print(f"\n{'─' * 60}")
        print(f"  Design iteration {iteration}")
        print(f"{'─' * 60}")

        pipeline_failed = False

        for backtrack in range(args.max_backtrack + 1):
            bt_suffix = "" if backtrack == 0 else f"_bt{backtrack}"
            iter_dir = os.path.join(work_dir, f"iter{iteration}{bt_suffix}")
            os.makedirs(iter_dir, exist_ok=True)

            write_alpha(alpha_file, alpha)
            write_alpha(os.path.join(work_dir, f"alpha_iter{iteration}.dat"), alpha)

            # --- 1. Deform mesh ---
            deform_xml = os.path.join(iter_dir, "inputs_deform.xml")
            write_deform_xml(deform_xml, base_mesh)

            deformed_mesh = os.path.join(iter_dir, "naca0012_deformed.msh")
            ok, _ = run([
                deform_mesh_exe, deform_xml,
                "--alpha", os.path.abspath(alpha_file),
                "--output", os.path.abspath(deformed_mesh),
            ], "deform_mesh", cwd=iter_dir)
            if not ok:
                print(f"  Mesh deformation failed at iteration {iteration}!")
                if prev_objective is not None and backtrack < args.max_backtrack:
                    arrival_step *= 0.5
                    alpha = prev_alpha + arrival_step * prev_direction
                    print(f"      Halving step to {arrival_step:.6e}, "
                          f"retrying (backtrack {backtrack + 1}/{args.max_backtrack})...")
                    continue
                pipeline_failed = True
                break

            # --- 2. Forward solve ---
            fwd_xml = os.path.join(iter_dir, "inputs_forward.xml")
            write_forward_xml(fwd_xml, "naca0012_deformed.msh", restart_file="")

            ok, fwd_stdout = run([app2d, fwd_xml], "forward", cwd=iter_dir)
            restart_path = os.path.join(iter_dir, "restart2d.bin")
            if not ok or not os.path.exists(restart_path):
                print(f"  Forward solve failed at iteration {iteration}!")
                if prev_objective is not None and backtrack < args.max_backtrack:
                    arrival_step *= 0.5
                    alpha = prev_alpha + arrival_step * prev_direction
                    print(f"      Halving step to {arrival_step:.6e}, "
                          f"retrying (backtrack {backtrack + 1}/{args.max_backtrack})...")
                    continue
                pipeline_failed = True
                break

            # --- Early overshoot check (before adjoint) ---
            fwd_cl, fwd_cd = parse_cl_cd_from_output(fwd_stdout)
            if fwd_cl is not None:
                print(f"    Forward Cl = {fwd_cl:.10e}")
            if fwd_cd is not None:
                print(f"    Forward Cd = {fwd_cd:.10e}")

            fwd_objective_est = None
            if ADJOINT_OBJECTIVE == "LiftOverDrag" and fwd_cl is not None and fwd_cd is not None and fwd_cd != 0.0:
                fwd_objective_est = fwd_cl / fwd_cd
                print(f"    Forward L/D = {fwd_objective_est:.10e}")
            elif ADJOINT_OBJECTIVE == "Drag" and fwd_cd is not None:
                fwd_objective_est = fwd_cd
            elif ADJOINT_OBJECTIVE == "Lift" and fwd_cl is not None:
                fwd_objective_est = fwd_cl

            if (fwd_objective_est is not None
                    and prev_objective is not None
                    and backtrack < args.max_backtrack):
                if MAXIMIZE and fwd_objective_est < prev_objective:
                    arrival_step *= 0.5
                    alpha = prev_alpha + arrival_step * prev_direction
                    print(f"  *** Overshoot: {ADJOINT_OBJECTIVE} decreased "
                          f"({fwd_objective_est:.6e} < {prev_objective:.6e})")
                    print(f"      Halving step to {arrival_step:.6e}, "
                          f"retrying (backtrack {backtrack + 1}/{args.max_backtrack})...")
                    continue
                elif not MAXIMIZE and fwd_objective_est > prev_objective:
                    arrival_step *= 0.5
                    alpha = prev_alpha + arrival_step * prev_direction
                    print(f"  *** Overshoot: {ADJOINT_OBJECTIVE} increased "
                          f"({fwd_objective_est:.6e} > {prev_objective:.6e})")
                    print(f"      Halving step to {arrival_step:.6e}, "
                          f"retrying (backtrack {backtrack + 1}/{args.max_backtrack})...")
                    continue

            if (fwd_objective_est is not None
                    and prev_objective is not None):
                went_wrong = (MAXIMIZE and fwd_objective_est < prev_objective) or \
                             (not MAXIMIZE and fwd_objective_est > prev_objective)
                if went_wrong:
                    print(f"  *** {ADJOINT_OBJECTIVE} still worsening after {args.max_backtrack} "
                          f"backtracks — accepting and computing adjoint.")

            # --- 3. Adjoint solve ---
            adj_xml = os.path.join(iter_dir, "inputs_adjoint.xml")
            write_adjoint_xml(adj_xml, "naca0012_deformed.msh")

            ok, _ = run([adjoint_exe, adj_xml], "adjoint", cwd=iter_dir)
            if not ok:
                print(f"  Adjoint solve failed at iteration {iteration}!")
                pipeline_failed = True
                break

            # --- 4. Shape gradient ---
            sg_xml = os.path.join(iter_dir, "inputs_shape_grad.xml")
            write_shape_grad_xml(sg_xml, "naca0012_deformed.msh", base_mesh)

            iter_grad = os.path.join(iter_dir, "gradient.dat")
            iter_obj = os.path.join(iter_dir, "objective.dat")

            ok, _ = run([
                shape_grad_exe, sg_xml,
                "--alpha", os.path.abspath(alpha_file),
                "--gradient", os.path.abspath(iter_grad),
                "--objective", os.path.abspath(iter_obj),
            ], "shape_grad", cwd=iter_dir)
            if not ok:
                print(f"  Shape gradient failed at iteration {iteration}!")
                pipeline_failed = True
                break

            gradient = read_gradient(iter_grad)
            objective = read_objective(iter_obj)
            grad_norm = np.linalg.norm(gradient)

            print(f"\n  Objective ({ADJOINT_OBJECTIVE}) = {objective:.10e}")
            print(f"  |gradient|     = {grad_norm:.6e}")

            break  # accept this result

        if pipeline_failed:
            break

        step_size = get_step(iteration)

        if arrival_step is not None:
            print(f"  Arrival step   = {arrival_step:.6e}")
            if arrival_step < step_size:
                step_size = arrival_step
                print(f"  (Capping next step to backtracked arrival step)")
        print(f"  Next step size = {step_size:.6e}")

        # Append to history
        with open(history_file, "a", newline="") as hf:
            w = csv.writer(hf)
            w.writerow([iteration, objective, grad_norm,
                        arrival_step if arrival_step else 0.0]
                       + list(alpha) + list(gradient))

        if grad_norm < args.tol:
            print(f"\n  Converged: |gradient| = {grad_norm:.6e} < {args.tol}")
            break

        direction = OPT_SIGN * gradient / grad_norm
        prev_objective = objective
        prev_alpha = alpha.copy()
        prev_direction = direction.copy()
        arrival_step = step_size
        alpha = alpha + step_size * direction

        delta = np.linalg.norm(alpha - prev_alpha)
        print(f"  |delta alpha|  = {delta:.6e}")
        print(f"  max |alpha|    = {np.max(np.abs(alpha)):.6e}")

    # ------------------------------------------------------------------
    #  Wrap up
    # ------------------------------------------------------------------
    write_alpha(alpha_file, alpha)
    print("\n" + "=" * 60)
    print("  Optimisation complete")
    print(f"  Final alpha:  {alpha_file}")
    print(f"  History:      {history_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
