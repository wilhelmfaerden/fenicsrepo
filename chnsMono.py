from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io, la
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
import ufl
from ufl import grad, inner, div, nabla_grad, dot, outer, sym
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
lid_speed = 0
dt = 0.003
num_time_steps = 1000
t = 0
eps = 0.025
sigma = 0.25
rho1 = 2 # "oil"
rho2 = 3 # "water"
nu = 0.005
m0 = 1
f = fem.Constant(msh, ScalarType((0.0, 0.0))) # body forces

# Function spaces (Taylor-Hood for NS)
P1 = element("Lagrange", msh.basix_cell(), 1)
P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
V = fem.functionspace(msh, mixed_element([P1,P1,P2,P1]))
Q, _ = V.sub(2).collapse()
L, _ = V.sub(3).collapse()

# Test functions CHNS
v, w, q, l = ufl.TestFunctions(V)

# CHNS functions
xi = fem.Function(V)
xi_old = fem.Function(V)
phi, mu, u, p = ufl.split(xi)
phi_old, mu_old, u_old, p_old = ufl.split(xi_old)


# Driven lid cavity markers
def walls(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[0], 1.0)
    )

def lid(x):
    return np.isclose(x[1], 1.0)


# Boundary conditions
u_lid = fem.Function(Q)
u_walls = fem.Function(Q)

u_lid.interpolate(lambda x: np.vstack((-lid_speed * np.ones(x.shape[1]), np.zeros(x.shape[1]))))
u_walls.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.zeros(x.shape[1]))))

lid_dofs = fem.locate_dofs_geometrical((V.sub(2), Q), lid)
wall_dofs = fem.locate_dofs_geometrical((V.sub(2), Q), walls)
bc_lid = fem.dirichletbc(u_lid, lid_dofs, V.sub(2))
bc_walls = fem.dirichletbc(u_walls, wall_dofs, V.sub(2))
bc_u = [bc_lid, bc_walls]

p_zero = fem.Constant(msh, ScalarType(0))
p_dofs = fem.locate_dofs_geometrical((V.sub(3), L), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))[0]
bc_p = [fem.dirichletbc(p_zero, p_dofs, V.sub(3))]

bcs = bc_u + bc_p


def doublewell(p):
    return 1/4*(1-p**2)**2

def doublewell_prime(p):
    return p**3 - p


J = (rho1-rho2)*eps*m0*grad(mu)/2
rho = (1-phi)*rho1/2 + (1+phi)*rho2/2

# CHNS weak form (no penalty method)
def implicit_euler():
    F_u = (rho*dot(u - u_old, q) + dt * (dot(dot(rho*u + J, nabla_grad(u)), q)
            + 2*nu*inner(sym(grad(u)), grad(q)) - dot(p, div(q)) - dot(f, q) + phi*dot(grad(mu), q)))*ufl.dx
    F_inc = div(u)*l*ufl.dx
    F_phi = ((phi - phi_old)*v - dt * phi*dot(u, grad(v)) + dt * eps*m0*dot(grad(mu), grad(v)))*ufl.dx # Conservative advection form, m = eps*m0 for Case II (AGG)
    F_mu = (mu*w - sigma*doublewell_prime(phi)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
    return F_u, F_inc, F_phi, F_mu, "Implicit Euler"

def nonlin_convex_split():
    F_u = (rho*dot(u - u_old, q) + dt * (dot(dot(rho*u + J, nabla_grad(u)), q)
            + 2*nu*inner(sym(grad(u)), grad(q)) - dot(p, div(q)) - dot(f, q) + phi*dot(grad(mu), q)))*ufl.dx
    F_inc = div(u)*l*ufl.dx
    F_phi = ((phi - phi_old)*v - dt * phi*dot(u, grad(v)) + dt * eps*m0*dot(grad(mu), grad(v)))*ufl.dx # Conservative advection form, m = eps*m0 for Case II (AGG)
    F_mu = (mu*w - sigma*(phi**3 - phi_old)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
    return F_u, F_inc, F_phi, F_mu, "Nonlinear Convex Split"

rng = np.random.default_rng(42)
xi.sub(0).interpolate(lambda x: rng.random(x.shape[1]).clip(-0.5,0.5))
xi.x.scatter_forward()

# Choose time stepping method for CH
F_u, F_inc, F_phi, F_mu, scheme = implicit_euler()

F = F_u + F_inc + F_phi + F_mu


# Monolithic problem
problem_CH = NonlinearProblem(
    F,
    xi,
    bcs=bcs,
    petsc_options_prefix="chns_",
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
        "pc_factor_mat_solver_type": "mumps",
        #"pc_factor_zero_pivot": 1.0e-12,
        "ksp_error_if_not_converged": True,
    },
)


# Plotting
V_phi, dofs = V.sub(0).collapse()

cells, types, x = plot.vtk_mesh(V_phi)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["phi"] = xi.x.array[dofs].real
grid.set_active_scalars("phi")

plotter = pyvistaqt.BackgroundPlotter(title=f"CH/NS Phase Plot: {scheme}", auto_update=True)
plotter.add_mesh(grid, clim =[-1, 1], scalar_bar_args={"title": f"$\phi$", "label_font_size": 22, "title_font_size": 28, "vertical": True, "position_x": 0.80, "position_y": 0.15, "height": 0.70})
plotter.view_xy(negative = True)
plotter.add_text(f"time: {t}", font_size=12, name="timelabel")


energies = []
for i in range(num_time_steps):
    t += dt

    # CHNS solve
    xi_old.x.array[:] = xi.x.array
    xi_old.x.scatter_forward()
    xi = problem_CH.solve()
    xi.x.scatter_forward()

    # Add phi data to grid for plotting
    plotter.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
    grid.point_data["phi"] = xi.x.array[dofs].real
    plotter.app.processEvents()
    # time.sleep(0.1)

    # phi conservation test
    print(fem.assemble_scalar(fem.form(phi*ufl.dx)), i)

    if i > 0:
        E = fem.assemble_scalar(fem.form(((sigma/eps)*doublewell(phi) + (sigma*eps/2)*inner(grad(phi),grad(phi)) + (rho/2)*inner(u,u))*ufl.dx)) # CHNS energy
        energies.append(E)

plotter.save_graphic("output/ch_plot.pdf")


# Latex rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

t_vals = np.arange(dt, dt * num_time_steps, dt)
fig, ax = plt.subplots()
ax.plot(t_vals, energies)
ax.grid(True)
ax.set_xlim(0, dt*num_time_steps)
ax.set_ylim(np.min(energies)*0.9, np.max(energies))

ax.set_xlabel(r"$t$", fontsize=14)
ax.set_ylabel(r"$\mathcal{E}(t)$", fontsize=14, rotation=0, labelpad=16)
ax.set_title(f"CHNS Free Energy: {scheme}", fontsize=16)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig("output/ch_energy.pdf", dpi=200)
plt.show()