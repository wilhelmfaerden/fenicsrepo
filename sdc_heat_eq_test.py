from mpi4py import MPI
import argparse
import numpy as np
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
import ufl

from collocationNodes import compute_A_from_nodes  # same as in your ODE SISDC


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
n = 64
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(n, n),
    cell_type=mesh.CellType.triangle,
)


def solve_heat_sdc(num_time_steps: int, sweeps: int, endtime: float):
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Unknown and previous timestep solution
    u = fem.Function(V)
    u_old = fem.Function(V)
    v = ufl.TestFunction(V)

    # Time step
    dt = endtime / num_time_steps
    t = 0.0
    dtm = fem.Constant(msh, ScalarType(0.0))

    # Initial condition
    def u0(x):
        return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])
    u.interpolate(u0)
    u.x.scatter_forward()

    u_exact = fem.Function(V)

    xi = np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0])
    c_nodes = 0.5 * (xi + 1.0)

    # Integration matrix A[i, j] = ∫_0^{c_i} L_j(s) ds
    A = compute_A_from_nodes(c_nodes, nquad=8)

    # Substep lengths
    dtms = (c_nodes[1:] - c_nodes[:-1]) * dt
    M = len(dtms) + 1

    # Quadrature weights per subinterval
    qms_delta = A[1:, :] - A[:-1, :]

    u_new = [fem.Function(V) for _ in range(M)]
    u_stages = [fem.Function(V) for _ in range(M)]

    # UFL stage expressions
    u_s_curr = [u_new[j] for j in range(M)]
    u_s_prev = [u_stages[j] for j in range(M)]

    # f(u) = Δu in weak form ⇒  ∫ (-∇u · ∇v) dx
    F_stage_prev = [-ufl.dot(ufl.grad(u_s_prev[j]), ufl.grad(v)) * ufl.dx for j in range(M)]

    # Time-/stage-/sweep-dependent coefficients as Constants
    beta_u = [fem.Constant(msh, ScalarType(0.0)) for _ in range(M)]
    gamma_u = [fem.Constant(msh, ScalarType(0.0)) for _ in range(M)]
    delta_F = [fem.Constant(msh, ScalarType(0.0)) for _ in range(M)]

    # Build RHS(UFL) in terms of these Functions and Constants
    RHS_u = sum(beta_u[j] * u_s_curr[j] * v * ufl.dx for j in range(M))
    RHS_diff = sum(gamma_u[j] * ufl.dot(ufl.grad(u_s_prev[j]), ufl.grad(v)) * ufl.dx for j in range(M))
    RHS_quad = sum(delta_F[j] * F_stage_prev[j] for j in range(M))
    RHS = RHS_u + RHS_diff + RHS_quad

    # Fixed residual:
    #    ∫ (u*v + dtm ∇u·∇v) dx - RHS = 0
    F_res = (u*v + dtm*ufl.dot(ufl.grad(u), ufl.grad(v)))*ufl.dx - RHS

    # Single NonlinearProblem (the PDE is linear, but this keeps the same pattern)
    problem = NonlinearProblem(
        F_res,
        u,
        petsc_options_prefix="sdc_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_stol": 1e-3,
            "snes_atol": 0.0,
            "snes_rtol": 0.0,
            "snes_max_it": 20,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )


    for _ in range(num_time_steps):
        t += dt

        # Save previous timestep solution
        u_old.x.array[:] = u.x.array
        u_old.x.scatter_forward()

        # Initialise stage data for this timestep from u_old
        for j in range(M):
            u_new[j].x.array[:] = u_old.x.array
            u_new[j].x.scatter_forward()
            u_stages[j].x.array[:] = u_old.x.array
            u_stages[j].x.scatter_forward()

        # SDC sweeps
        for k in range(sweeps):
            # current sweep starts from previous sweep's stages
            u_new[0].x.array[:] = u_stages[0].x.array
            u_new[0].x.scatter_forward()

            # Node iteration over subintervals m
            for m in range(M - 1):
                # Set current substep size
                dtm.value = ScalarType(dtms[m])

                # Reset all coefficients
                for j in range(M):
                    beta_u[j].value = ScalarType(0.0)
                    gamma_u[j].value = ScalarType(0.0)
                    delta_F[j].value = ScalarType(0.0)

                # Predictor term: u_s_curr[m] ≈ u at stage m of current sweep
                beta_u[m].value = ScalarType(1.0)

                # Basic corrector term uses u at stage m+1 from previous sweep:
                # dtm * ∫ ∇u_s_prev[m+1] · ∇v dx
                gamma_u[m+1].value = ScalarType(dtm.value)

                # SDC correction: dt * ∑_j qms_delta[m,j] * F_stage_prev[j]
                for j in range(M):
                    delta_F[j].value = ScalarType(dt * qms_delta[m, j])

                u_new[m + 1].x.array[:] = problem.solve().x.array
                u_new[m + 1].x.scatter_forward()

            for j in range(M):
                u_stages[j].x.array[:] = u_new[j].x.array
                u_stages[j].x.scatter_forward()

        u.x.array[:] = u_new[-1].x.array
        u.x.scatter_forward()

    def u_exact_fun(x, t=t):
        return (np.cos(np.pi * x[0])* np.cos(np.pi * x[1])* np.exp(-2.0 * (np.pi**2) * t))
    u_exact.interpolate(u_exact_fun)

    return u, u_exact


N = 3
def main() -> None:
    parser = argparse.ArgumentParser(description="Time-convergence study for the biharmonic SDC test.")
    #parser.add_argument("--n", type=int, default=64, help="Mesh resolution in each direction.")
    parser.add_argument("--sweeps", type=int, default=N, help="Number of SDC sweeps per time step.")
    parser.add_argument("--endtime", type=float, default=0.03, help="Final time for the convergence test.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Number of time steps to test.",
    )
    args = parser.parse_args()

    dt_values = []
    errors = []

    for num_time_steps in args.steps:
        dt = args.endtime / num_time_steps

        u, u_exact = solve_heat_sdc(num_time_steps, args.sweeps, args.endtime)
        u_half, u_exact = solve_heat_sdc(2 * num_time_steps, args.sweeps, args.endtime)

        err = np.sqrt(fem.assemble_scalar(fem.form((u - u_half)**2 * ufl.dx)))
        dt_values.append(dt)
        errors.append(err)
        print(f"steps={num_time_steps:4d}  dt={dt:.6e}  ||u_N - u_2N||={err:.6e}")

    if len(errors) > 1:
        print(f"\nObserved orders for {args.sweeps - 1} correction sweeps:")
        for i in range(len(errors) - 1):
            p = np.log(errors[i] / errors[i + 1]) / np.log(dt_values[i] / dt_values[i + 1])
            print(f"  {args.steps[i]:4d} -> {args.steps[i+1]:4d} : {p:.4f}")


if __name__ == "__main__":
    main()

