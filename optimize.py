#!/usr/bin/env python3
"""
Shape optimization driver for the 2D Euler DG solver.

Orchestrates: deform_mesh -> app2d -> app2d_discrete_adjoint -> shape_grad
in a gradient-based optimization loop.

Usage:
    python optimize.py inputs2d_naca_opt.xml [options]

Options:
    --max-iter N      Maximum optimization iterations (default: 50)
    --step S          Initial step size (default: 0.01)
    --tol T           Gradient norm convergence tolerance (default: 1e-6)
    --build-dir DIR   Build directory (default: build)
    --work-dir DIR    Working directory for iteration outputs (default: opt_work)
    --lbfgs           Use L-BFGS instead of steepest descent
    --constraint C    Lift constraint, e.g. "Cl=0.5" (default: none)
    --penalty P       Penalty parameter for constraint (default: 100.0)
"""

import argparse
import os
import sys
import shutil
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import csv


def read_alpha(filename):
    if not os.path.exists(filename):
        return None
    return np.loadtxt(filename)


def write_alpha(filename, alpha):
    np.savetxt(filename, alpha, fmt="%.15e")


def read_gradient(filename):
    return np.loadtxt(filename)


def read_objective(filename):
    with open(filename) as f:
        return float(f.read().strip())


def modify_xml(base_xml, output_xml, overrides):
    """Create a modified copy of the XML input file with parameter overrides."""
    tree = ET.parse(base_xml)
    root = tree.getroot()
    params = root.find("PARAMETERS")

    existing_keys = {}
    for p_elem in params.findall("P"):
        text = p_elem.text.strip()
        if "=" in text:
            key = text.split("=")[0].strip()
            existing_keys[key] = p_elem

    for key, value in overrides.items():
        if key in existing_keys:
            existing_keys[key].text = f" {key} = {value} "
        else:
            new_p = ET.SubElement(params, "P")
            new_p.text = f" {key} = {value} "

    tree.write(output_xml, xml_declaration=True, encoding="utf-8")


def run_command(cmd, label="", cwd=None):
    """Run a subprocess and check for errors."""
    print(f"  [{label}] {' '.join(cmd)}")
    if cwd:
        print(f"           cwd={cwd}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  ERROR in {label}:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"    {line}")
    return True


class LBFGSOptimizer:
    """Limited-memory BFGS optimizer."""

    def __init__(self, n, m=10):
        self.n = n
        self.m = m
        self.s_list = []
        self.y_list = []
        self.rho_list = []

    def update(self, s, y):
        ys = np.dot(y, s)
        if ys > 1e-30:
            self.s_list.append(s.copy())
            self.y_list.append(y.copy())
            self.rho_list.append(1.0 / ys)
            if len(self.s_list) > self.m:
                self.s_list.pop(0)
                self.y_list.pop(0)
                self.rho_list.pop(0)

    def direction(self, grad):
        q = grad.copy()
        k = len(self.s_list)
        if k == 0:
            return -grad

        alpha = np.zeros(k)
        for i in range(k - 1, -1, -1):
            alpha[i] = self.rho_list[i] * np.dot(self.s_list[i], q)
            q -= alpha[i] * self.y_list[i]

        gamma = np.dot(self.s_list[-1], self.y_list[-1]) / np.dot(
            self.y_list[-1], self.y_list[-1]
        )
        r = gamma * q

        for i in range(k):
            beta = self.rho_list[i] * np.dot(self.y_list[i], r)
            r += self.s_list[i] * (alpha[i] - beta)

        return -r


def parse_constraint(constraint_str):
    """Parse a constraint like 'Cl=0.5' into (objective_name, target_value)."""
    if not constraint_str:
        return None, None
    parts = constraint_str.split("=")
    if len(parts) != 2:
        print(f"Warning: cannot parse constraint '{constraint_str}'")
        return None, None
    return parts[0].strip(), float(parts[1].strip())


def main():
    parser = argparse.ArgumentParser(description="Shape optimization driver")
    parser.add_argument("input_xml", help="Base XML input file")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--build-dir", default="..",
                        help="Directory containing executables (default: ..)")
    parser.add_argument("--work-dir", default=".",
                        help="Working directory for iteration outputs (default: .)")
    parser.add_argument("--lbfgs", action="store_true")
    parser.add_argument("--constraint", default="")
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--restart-iter", type=int, default=0,
                        help="Restart from a given iteration")
    args = parser.parse_args()

    base_xml = os.path.abspath(args.input_xml)
    xml_dir = os.path.dirname(base_xml)

    build_dir = os.path.abspath(args.build_dir)
    work_dir = os.path.abspath(args.work_dir)

    deform_mesh_exe = os.path.join(build_dir, "deform_mesh")
    app2d_exe = os.path.join(build_dir, "app2d")
    adjoint_exe = os.path.join(build_dir, "app2d_discrete_adjoint")
    shape_grad_exe = os.path.join(build_dir, "shape_grad")

    for exe in [deform_mesh_exe, app2d_exe, adjoint_exe, shape_grad_exe]:
        if not os.path.exists(exe):
            print(f"Error: executable not found: {exe}")
            print("Build with: cd build && cmake .. && make deform_mesh app2d app2d_discrete_adjoint shape_grad")
            sys.exit(1)

    os.makedirs(work_dir, exist_ok=True)

    # Read base XML to get nBumps and mesh file
    tree = ET.parse(base_xml)
    root = tree.getroot()
    params_elem = root.find("PARAMETERS")
    param_dict = {}
    for p in params_elem.findall("P"):
        text = p.text.strip()
        if "=" in text:
            k, v = text.split("=", 1)
            param_dict[k.strip()] = v.strip()

    n_bumps_upper = int(param_dict.get("OptNBumpsUpper", "10"))
    n_bumps_lower = int(param_dict.get("OptNBumpsLower", "10"))
    n_design = n_bumps_upper + n_bumps_lower

    # Resolve mesh file relative to the XML file's directory
    base_mesh_rel = param_dict.get("MeshFile", "naca0012_quad.msh")
    base_mesh_file = os.path.join(xml_dir, base_mesh_rel)
    if not os.path.exists(base_mesh_file):
        base_mesh_file = os.path.abspath(base_mesh_rel)
    base_mesh_file = os.path.abspath(base_mesh_file)

    constraint_name, constraint_target = parse_constraint(args.constraint)

    print("=" * 60)
    print("Shape Optimization")
    print("=" * 60)
    print(f"  Input file:     {base_xml}")
    print(f"  Design vars:    {n_design} ({n_bumps_upper} upper + {n_bumps_lower} lower)")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Step size:      {args.step}")
    print(f"  Tolerance:      {args.tol}")
    print(f"  Optimizer:      {'L-BFGS' if args.lbfgs else 'Steepest Descent'}")
    if constraint_name:
        print(f"  Constraint:     {constraint_name} = {constraint_target}")
    print("=" * 60)

    # Initialize design variables
    alpha_file = os.path.join(work_dir, "alpha.dat")
    if args.restart_iter > 0:
        restart_alpha = os.path.join(work_dir, f"alpha_{args.restart_iter:04d}.dat")
        if os.path.exists(restart_alpha):
            alpha = np.loadtxt(restart_alpha)
            print(f"Restarting from iteration {args.restart_iter}")
        else:
            print(f"Error: restart file {restart_alpha} not found")
            sys.exit(1)
    else:
        alpha = np.zeros(n_design)

    # History logging
    history_file = os.path.join(work_dir, "opt_history.csv")
    history_mode = "a" if args.restart_iter > 0 else "w"
    history = open(history_file, history_mode, newline="")
    csv_writer = csv.writer(history)
    if args.restart_iter == 0:
        csv_writer.writerow(["iter", "objective", "grad_norm", "step_size"]
                            + [f"alpha_{k}" for k in range(n_design)]
                            + [f"grad_{k}" for k in range(n_design)])

    lbfgs = LBFGSOptimizer(n_design) if args.lbfgs else None
    prev_grad = None
    prev_direction = None
    prev_alpha = None
    step_size = args.step

    start_iter = args.restart_iter

    for iteration in range(start_iter, start_iter + args.max_iter):
        print(f"\n--- Optimization Iteration {iteration + 1} ---")

        iter_dir = os.path.join(work_dir, f"iter_{iteration + 1:04d}")
        os.makedirs(iter_dir, exist_ok=True)

        write_alpha(alpha_file, alpha)
        write_alpha(os.path.join(work_dir, f"alpha_{iteration + 1:04d}.dat"), alpha)

        deformed_mesh = os.path.join(iter_dir, "deformed.msh")

        # 1. Deform mesh
        deform_xml = os.path.join(iter_dir, "deform.xml")
        modify_xml(base_xml, deform_xml, {
            "MeshFile": base_mesh_file,
        })
        ok = run_command([
            deform_mesh_exe, deform_xml,
            "--alpha", alpha_file,
            "--output", deformed_mesh
        ], "deform_mesh", cwd=iter_dir)
        if not ok:
            print("Mesh deformation failed!")
            break

        # 2. Forward solve (run in iter_dir so restart lands there)
        forward_xml = os.path.join(iter_dir, "forward.xml")
        modify_xml(base_xml, forward_xml, {
            "MeshFile": os.path.abspath(deformed_mesh),
        })
        ok = run_command([app2d_exe, forward_xml], "forward_solve", cwd=iter_dir)
        restart_path = os.path.join(iter_dir, "restart2d.bin")
        if not ok or not os.path.exists(restart_path):
            print("Forward solve failed or NaN detected!")
            print("  Halving step size and retrying...")
            step_size *= 0.5
            alpha = prev_alpha + step_size * prev_direction if prev_direction is not None else np.zeros(n_design)
            continue

        # 3. Adjoint solve (run in iter_dir, picks up restart2d.bin from there)
        adjoint_xml = os.path.join(iter_dir, "adjoint.xml")
        modify_xml(base_xml, adjoint_xml, {
            "MeshFile": os.path.abspath(deformed_mesh),
            "RestartFile": "restart2d.bin",
        })
        ok = run_command([adjoint_exe, adjoint_xml], "adjoint_solve", cwd=iter_dir)
        if not ok:
            print("Adjoint solve failed!")
            break

        # 4. Gradient computation (run in iter_dir)
        grad_xml = os.path.join(iter_dir, "gradient.xml")
        gradient_file = os.path.join(iter_dir, "gradient.dat")
        objective_file = os.path.join(iter_dir, "objective.dat")
        modify_xml(base_xml, grad_xml, {
            "MeshFile": os.path.abspath(deformed_mesh),
            "RestartFile": "restart2d.bin",
            "AdjointRestartFile": "discrete_adjoint_restart.bin",
            "BaseMeshFile": base_mesh_file,
        })
        ok = run_command([
            shape_grad_exe, grad_xml,
            "--alpha", alpha_file,
            "--gradient", os.path.abspath(gradient_file),
            "--objective", os.path.abspath(objective_file),
        ], "shape_grad", cwd=iter_dir)
        if not ok:
            print("Gradient computation failed!")
            break

        # Read results
        gradient = read_gradient(gradient_file)
        objective = read_objective(objective_file)
        grad_norm = np.linalg.norm(gradient)

        print(f"  Objective = {objective:.10e}")
        print(f"  |gradient| = {grad_norm:.6e}")

        # Log history
        csv_writer.writerow(
            [iteration + 1, objective, grad_norm, step_size]
            + list(alpha) + list(gradient)
        )
        history.flush()

        # Check convergence
        if grad_norm < args.tol:
            print(f"\nConverged! |gradient| = {grad_norm:.6e} < {args.tol}")
            break

        # Compute search direction
        if args.lbfgs and prev_grad is not None:
            s = alpha - prev_alpha
            y = gradient - prev_grad
            lbfgs.update(s, y)
            direction = lbfgs.direction(gradient)
        else:
            direction = -gradient

        # Normalize direction for step size
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction /= dir_norm

        # Update design variables
        prev_alpha = alpha.copy()
        prev_grad = gradient.copy()
        prev_direction = direction.copy()
        alpha = alpha + step_size * direction

        delta_alpha = np.linalg.norm(alpha - prev_alpha)
        print(f"  Step size = {step_size:.6e}")
        print(f"  |delta_alpha| = {delta_alpha:.6e}")
        print(f"  max|alpha| = {np.max(np.abs(alpha)):.6e}")

    history.close()

    # Write final results
    write_alpha(os.path.join(work_dir, "alpha_final.dat"), alpha)
    print(f"\nOptimization complete. Results in {work_dir}/")
    print(f"  Final alpha: {work_dir}/alpha_final.dat")
    print(f"  History:     {work_dir}/opt_history.csv")


if __name__ == "__main__":
    main()
