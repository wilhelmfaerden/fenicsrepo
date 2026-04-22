from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io, la
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
import ufl
from ufl import grad, inner, div, nabla_grad, dot
import time
import pyvista
import pyvistaqt


# Mesh
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)

# Parameters
lid_speed = 3
dt = 0.003
num_time_steps = 5000
t = 0
eps = 0.025
sigma = 1
rho = 0.25
nu = 0.01
m = 1
f = fem.Constant(msh, ScalarType((0.0, 0.0)))
# advective CH test
uADV = fem.Constant(msh, ScalarType((-1.0, 0.0)))

# Function space CH
P1 = element("Lagrange", msh.basix_cell(), 1)
V = fem.functionspace(msh, mixed_element([P1,P1]))
# Function spaces NS
Q = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))
L = fem.functionspace(msh, ("Lagrange", 1))


# Test functions CH
v, w = ufl.TestFunction(V)
# Test functions NS
q = ufl.TestFunction(Q)
l = ufl.TestFunction(L)
# Trial funcs for LinearProblem
u_ = ufl.TrialFunction(Q)
p_ = ufl.TrialFunction(L)

# CH test functions
xi = fem.Function(V)
xi_old = fem.Function(V)
phi, mu = ufl.split(xi)
phi_old, mu_old = ufl.split(xi_old)
# NS test functions
u = fem.Function(Q)
u_old = fem.Function(Q)
p = fem.Function(L)
p_old = fem.Function(L)

# Driven lid cavity boundary conditions
def walls(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[0], 1.0)
    )

def lid(x):
    return np.isclose(x[1], 1.0)

u_lid = fem.Constant(msh, ScalarType((-lid_speed, 0.0)))
u_walls = fem.Constant(msh, ScalarType((0.0, 0.0)))

lid_dofs = fem.locate_dofs_geometrical(Q, lid)
wall_dofs = fem.locate_dofs_geometrical(Q, walls)
bc_lid = fem.dirichletbc(u_lid, lid_dofs, Q)
bc_walls = fem.dirichletbc(u_walls, wall_dofs, Q)
bc_u = [bc_lid,bc_walls]

p_zero = fem.Constant(msh, ScalarType(0))
p_dofs = fem.locate_dofs_geometrical(L, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
bc_p = [fem.dirichletbc(p_zero, p_dofs, L)]


def doublewell(p):
    return 1/4*(1-p**2)**2

def doublewell_prime(p):
    return -(1-p**2)*p


def implicit_euler():
    # Navier-Stokes weak form (not yet CHNS!)
    # Step 1: tentative velocity
    a1 = (dot(u_, q) + dt * nu*inner(grad(u_), grad(q)))*ufl.dx
    L1 = (dot(u_old, q) + dt * ( -dot(dot(u_old, nabla_grad(u_old)), q) - dot(grad(p_old), q) + dot(f, q)))*ufl.dx

    # Step 2: pressure
    a2 = dt * dot(grad(p_), grad(l))*ufl.dx
    L2 = -rho*div(u)*l*ufl.dx

    # Step 3: velocity correction
    a3 = rho*dot(u_, q)*ufl.dx
    L3 = (rho*dot(u, q) - dt * dot(grad(p), q))*ufl.dx


    # Advective Cahn-Hilliard weak form
    F_phi = ((phi - phi_old)*v + dt * dot(u,grad(phi))*v +  dt * eps*m*dot(grad(mu), grad(v)))*ufl.dx # extra eps factor?
    F_mu = (mu*w - sigma*doublewell_prime(phi)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
    return a1, L1, a2, L2, a3, L3, F_phi, F_mu, "Implicit Euler"

"""
def nonlin_convex_split():
    F_phi = (ufl.inner(phi, v)  + eps*m*dt * ufl.inner(ufl.grad(mu), ufl.grad(v)) - ufl.inner(phi_old, v))*ufl.dx
    F_mu = (ufl.inner(mu, w) - ufl.inner(doublewell_prime_con(phi) + doublewell_prime_exp(phi_old), w)/eps - ufl.inner(ufl.grad(phi), grad(w))*eps)*ufl.dx
    return F_phi, F_mu, "Nonlinear Convex Split"

def lin_convex_split():
    F_phi = (ufl.inner(phi, v)  + eps*m*dt * ufl.inner(ufl.grad(mu), ufl.grad(v)) - ufl.inner(phi_old, v))*ufl.dx
    F_mu = (ufl.inner(mu, w) - ufl.inner(2*phi + (phi_old)**3 - 3*phi_old, w)/eps - ufl.inner(ufl.grad(phi), grad(w))*eps)*ufl.dx
    return F_phi, F_mu, "Linear Convex Split"
"""

rng = np.random.default_rng(42)
xi.sub(0).interpolate(lambda x: rng.random(x.shape[1]).clip(-0.5,0.5))
xi.x.scatter_forward()

# Variational form
a1, L1, a2, L2, a3, L3, F_phi, F_mu, scheme = implicit_euler()

F = F_phi +F_mu

# Problems

problem_CH = NonlinearProblem(
    F,
    xi,
    petsc_options_prefix="ch_",
    petsc_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_stol": 1e-3,
        "snes_atol": 0,
        "snes_rtol": 0,
        "snes_max_it": 100,
        "snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)
"""
problem1 = LinearProblem(
    a1,
    L1,
    bcs=bc_u,
    petsc_options_prefix="ns_step1_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)

problem2 = LinearProblem(
    a2,
    L2,
    bcs=bc_p,
    petsc_options_prefix="ns_step2_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)

problem3 = LinearProblem(
    a3,
    L3,
    bcs=bc_u,
    petsc_options_prefix="ns_step3_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)
"""
problem1 = LinearProblem(
    a1, L1, bcs=bc_u,
    petsc_options_prefix="ns_step1_",
    petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_error_if_not_converged": True}
)

problem2 = LinearProblem(
    a2, L2, bcs=bc_p,
    petsc_options_prefix="ns_step2_",
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_error_if_not_converged": True}
)

problem3 = LinearProblem(
    a3, L3, bcs=bc_u,
    petsc_options_prefix="ns_step3_",
    petsc_options={"ksp_type": "cg", "pc_type": "sor", "ksp_error_if_not_converged": True}
)


# plotting
V_phi, dofs = V.sub(0).collapse()

cells, types, x = plot.vtk_mesh(V_phi)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["phi"] = xi.x.array[dofs].real
grid.set_active_scalars("phi")

plotter = pyvistaqt.BackgroundPlotter(title=f"CH Phase Plot: {scheme}", auto_update=True)
plotter.add_mesh(grid, clim =[-1, 1], scalar_bar_args={"title": f"$\phi$", "label_font_size": 22, "title_font_size": 28, "vertical": True, "position_x": 0.80, "position_y": 0.15, "height": 0.70})
plotter.view_xy(negative = True)
plotter.add_text(f"time: {t}", font_size=12, name="timelabel")


energies = []
for i in range(num_time_steps):
    t += dt

    # NS solve
    u_old.x.array[:] = u.x.array[:]
    p_old.x.array[:] = p.x.array[:]
    u_old.x.scatter_forward()
    p_old.x.scatter_forward()
    u.x.array[:] = problem1.solve().x.array
    u.x.scatter_forward()
    print(fem.assemble_scalar(fem.form(inner(u,u)*ufl.dx)))
    p.x.array[:] = problem2.solve().x.array
    p.x.scatter_forward()
    u.x.array[:] = problem3.solve().x.array
    u.x.scatter_forward()

    # CH solve
    xi_old.x.array[:] = xi.x.array
    xi_old.x.scatter_forward()
    xi = problem_CH.solve()
    xi.x.scatter_forward()

    # Add phi data to grid for plotting
    plotter.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
    grid.point_data["phi"] = xi.x.array[dofs].real
    plotter.app.processEvents()
    # time.sleep(0.1)


    if i > 0:
        E = fem.assemble_scalar(fem.form((sigma*doublewell(phi) + (sigma*(eps**2)/2)*inner(grad(phi),grad(phi)) + (rho/2)*inner(u,u))*ufl.dx)) # CHNS energy
        energies.append(E)

plotter.save_graphic("output/ch_plot.pdf")


t_vals = np.arange(dt, dt * num_time_steps, dt)
fig, ax = plt.subplots()
ax.plot(t_vals, energies)
ax.grid(True)
ax.set_xlim(0, dt*num_time_steps)
ax.set_ylim(np.min(energies)*0.9, np.max(energies))

ax.set_xlabel("t", fontsize=14)
ax.set_ylabel("E(t)", fontsize=14, rotation=0, labelpad=16)
ax.set_title(f"CH Free Energy: {scheme}", fontsize=16)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig("output/ch_energy.pdf", dpi=200)
plt.show()