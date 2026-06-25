import numpy as np

def compute_A_from_nodes(c, nquad=8):
    """
    Compute the SDC integration matrix A_ij:
        A[i, j] = ∫_0^{c_i} L_j(s) ds,
    where L_j is the Lagrange basis associated with node c_j.

    This follows the standard SDC construction: represent the integral over [0, c_i]
    via interpolation of F(y) at the collocation nodes and precompute the integrals
    of the basis polynomials.
    """
    c = np.asarray(c, dtype=float)
    P = len(c)  # P = M+1 nodes
    A = np.zeros((P, P), dtype=float)

    # Gauss–Legendre quadrature on [-1,1]
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