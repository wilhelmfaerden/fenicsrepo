import numpy as np


def compute_A_from_nodes(c, nquad=8):
    c = np.asarray(c, dtype=float)
    P = len(c)  # P = p+1 nodes
    A = np.zeros((P, P), dtype=float)

    # Gauss–Legendre quadrature on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(nquad)

    for i in range(P):
        a = 0.0
        b = c[i]
        if np.isclose(a, b):
            continue

        # Map [-1,1] → [a,b]
        s = 0.5 * (b - a) * x_gl + 0.5 * (b + a)
        w = 0.5 * (b - a) * w_gl

        # Lagrange basis L_j(s) at quadrature points
        L = np.ones((P, len(s)), dtype=float)
        for j in range(P):
            for k in range(P):
                if k == j:
                    continue
                L[j] *= (s - c[k]) / (c[j] - c[k])

        # Integrate each basis over [a,b]
        for j in range(P):
            A[i, j] = np.sum(w * L[j])

    return A


def solve_sisdc_scalar(lam, num_steps, sweeps, T, c_nodes, A):
    dt = T / num_steps
    c = np.asarray(c_nodes, dtype=float)

    # substep lengths Δt_m = (c_{m+1} - c_m) * Δt
    dtms = (c[1:] - c[:-1]) * dt
    M = len(dtms) + 1

    # subinterval quadrature weights q_{m,ℓ} = A_{m+1,ℓ} - A_{m,ℓ}
    qms_delta = A[1:, :] - A[:-1, :]

    # initial condition y(0) = 1
    y_n = 1.0

    for _ in range(num_steps):
        new_stages = np.full(M, y_n, dtype=float)
        y_stages = np.full(M, y_n, dtype=float)

        
        for k in range(sweeps):
            F_prev = lam * y_stages

            new_stages = np.zeros_like(y_stages)
            new_stages[0] = y_stages[0]

            for m in range(M-1):
                dtm = dtms[m]

                Imm = dt * np.dot(qms_delta[m, :], F_prev)
                rhs = new_stages[m] - dtm * lam * y_stages[m+1] + Imm # Found Waldo!! Changing new_stages to y_stages ruins convergence!

                # implicit solve (scalar):
                new_stages[m+1] = rhs / (1.0 - dtm * lam)

            y_stages[:] = new_stages

        y_n = y_stages[-1]

    return y_n


def convergence_test_sisdc_scalar(
    lam=-1,
    T=1.0,
    sweeps=1,
    steps_list=(1, 2, 4, 8, 16, 32, 64),
):
    # 4-point Gauss–Lobatto nodes on [0,1]
    xi = np.array([-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1.0])
    c_nodes = 0.5 * (xi + 1.0)

    # Build integration matrix A from nodes
    A = compute_A_from_nodes(c_nodes, nquad=8)

    dt_values = []
    errors = []

    for N in steps_list:
        dt = T / N
        yN = solve_sisdc_scalar(lam, N, sweeps, T, c_nodes, A)
        y2N = solve_sisdc_scalar(lam, 2 * N, sweeps, T, c_nodes, A)

        err = abs(yN - y2N)
        dt_values.append(dt)
        errors.append(err)
        print(f"steps={N:4d}  dt={dt:.6e}  |y_N - y_2N|={err:.6e}")

    if len(errors) > 1:
        print(f"\nObserved orders for {sweeps-1} correction sweeps:")
        for i in range(len(errors) - 1):
            p = np.log(errors[i] / errors[i + 1]) / np.log(
                dt_values[i] / dt_values[i + 1]
            )
            print(f"  {steps_list[i]:4d} -> {steps_list[i+1]:4d} : {p:.4f}")


if __name__ == "__main__":
    # Non-stiff test, λ = -1
    print("Non-stiff test, λ = -1, 0 correction sweeps (predictor only):")
    convergence_test_sisdc_scalar(lam=-1.0, sweeps=1)

    print("\nNon-stiff test, λ = -1, 1 correction sweep (should → order 2):")
    convergence_test_sisdc_scalar(lam=-1.0, sweeps=2)

    # Two correction sweep: sweeps=3
    print("\nNon-stiff test, λ = -1, 1 correction sweep (should increase order):")
    convergence_test_sisdc_scalar(lam=-1.0, sweeps=3)

    # Three correction sweep: sweeps=4
    print("\nNon-stiff test, λ = -1, 1 correction sweep (should increase order):")
    convergence_test_sisdc_scalar(lam=-1.0, sweeps=4)
