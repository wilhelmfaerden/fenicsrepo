from mpi4py import MPI
import argparse
import numpy as np
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from basix.ufl import element, mixed_element
import ufl


def solve_biharmonic_sdc(num_time_steps: int, n: int, sweeps: int, endtime: float) -> float:
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(n, n),
        cell_type=mesh.CellType.triangle,
    )

    P1 = element("Lagrange", msh.basix_cell(), 1)
    P2 = element("Lagrange", msh.basix_cell(), 2)
    V = fem.functionspace(msh, mixed_element([P1, P1]))

    v, w = ufl.TestFunction(V)
    xi = fem.Function(V)
    xi_old = fem.Function(V)
    phi, mu = ufl.split(xi)
    phi_old, mu_old = ufl.split(xi_old)

    dt = endtime / num_time_steps
    t = 0.0

    phi0 = lambda x: (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    xi.sub(0).interpolate(phi0)
    xi.x.scatter_forward()

    xcoord = ufl.SpatialCoordinate(msh)
    V_phi, _ = V.sub(0).collapse()
    phi_exact_P1 = fem.Function(V_phi)

    dtms = [(0.5 - np.sqrt(5) / 10) * dt, (np.sqrt(5) / 5) * dt, (0.5 - np.sqrt(5) / 10) * dt]
    A = np.array(
        [
            [0, 0, 0, 0],
            [378 / 3427, 221 / 1165, -317 / 9349, 61 / 5922],
            [439 / 6011, 1609 / 3571, 574 / 2529, -256 / 9493],
            [1 / 12, 5 / 12, 5 / 12, 1 / 12],
        ]
    )
    qms_delta = A[1:, :] - A[:-1, :]

    for _ in range(num_time_steps):
        t += dt
        xi_old.x.array[:] = xi.x.array
        xi_old.x.scatter_forward()

        xi_stages = [fem.Function(V) for _ in range(len(dtms) + 1)]
        xi_stages[0].x.array[:] = xi_old.x.array
        xi_stages[0].x.scatter_forward()

        xi_old_k_new = [fem.Function(V) for _ in range(len(dtms) + 1)]
        F_stages_new = []

        for k in range(sweeps):
            F_stages = F_stages_new
            F_stages_new = []
            xi_old_k = xi_old_k_new
            xi_old_k_new = [fem.Function(V) for _ in range(len(dtms) + 1)]

            m = 0
            phi_old_m, muI_old_m = ufl.split(xi_stages[0])
            F_stages_new.append(-(ufl.dot(ufl.grad(muI_old_m), ufl.grad(v))) * ufl.dx)
            xi_old_k_new[0].x.array[:] = xi_stages[0].x.array
            xi_old_k_new[0].x.scatter_forward()

            for dtm, qm in zip(dtms, qms_delta):
                RHS_test = phi_old_m * v * ufl.dx

                if k > 0:
                    phi_old_k, muI_old_k = ufl.split(xi_old_k[m + 1])
                    phi_old_km, muI_old_km = ufl.split(xi_old_k[m])
                    RHS_test = phi_old_m * v * ufl.dx + dtm * (ufl.dot(ufl.grad(muI_old_k), ufl.grad(v))) * ufl.dx + dt * sum(qm[j] * (F_stages[j]) for j in range(len(F_stages)))

                F_phi = (phi * v + dtm * (ufl.dot(ufl.grad(mu), ufl.grad(v)))) * ufl.dx - RHS_test
                F_mu = (mu * w - ufl.dot(ufl.grad(phi), ufl.grad(w))) * ufl.dx
                F = F_phi + F_mu

                problem = NonlinearProblem(
                    F,
                    xi,
                    petsc_options_prefix="sdc_",
                    petsc_options={
                        "snes_type": "newtonls",
                        "snes_linesearch_type": "none",
                        "snes_stol": 1e-3,
                        "snes_atol": 0,
                        "snes_rtol": 0,
                        "snes_max_it": 100,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",
                        "ksp_error_if_not_converged": True,
                    },
                )

                xi_stages[m + 1].x.array[:] = problem.solve().x.array
                xi_stages[m + 1].x.scatter_forward()

                phi_old_m, muI_old_m = ufl.split(xi_stages[m + 1])
                F_stages_new.append(-(ufl.dot(ufl.grad(muI_old_m), ufl.grad(v))) * ufl.dx)
                xi_old_k_new[m + 1].x.array[:] = xi_stages[m + 1].x.array
                xi_old_k_new[m + 1].x.scatter_forward()
                m += 1

        xi.x.array[:] = xi_stages[-1].x.array
        xi.x.scatter_forward()

    #phi_exact_P1.interpolate(lambda x, t=t: np.exp(-t * ((2 * np.pi) ** 4)) * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    phi_exact = ufl.exp(-t*((2*np.pi)**4))*(ufl.cos(2*np.pi*xcoord[0]) + ufl.cos(2*np.pi*xcoord[1]))
    l2_err = np.sqrt(fem.assemble_scalar(fem.form((phi - phi_exact) ** 2 * ufl.dx)))
    return l2_err

N = 2
def main() -> None:
    parser = argparse.ArgumentParser(description="Time-convergence study for the biharmonic SDC test.")
    parser.add_argument("--n", type=int, default=14, help="Mesh resolution in each direction.")
    parser.add_argument("--sweeps", type=int, default=N, help="Number of SDC sweeps per time step.")
    parser.add_argument("--endtime", type=float, default=0.002, help="Final time for the convergence test.")
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
        error = solve_biharmonic_sdc(num_time_steps, args.n, args.sweeps, args.endtime)
        dt_values.append(dt)
        errors.append(error)
        print(f"steps={num_time_steps:4d}  dt={dt:.6e}  L2={error:.6e}")

    if len(errors) > 1:
        print(f"Observed orders for {N-1} corr. sweeps:")
        for i in range(len(errors) - 1):
            order = np.log(errors[i] / errors[i + 1]) / np.log(dt_values[i] / dt_values[i + 1])
            print(f"  {args.steps[i]} -> {args.steps[i + 1]} : {order:.4f}")


if __name__ == "__main__":
    main()