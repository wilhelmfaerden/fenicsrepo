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
    n=(48, 48),
    cell_type=mesh.CellType.triangle,
)

# Parameters
lid_speed = 0
endtime = 6
num_time_steps = 2000
dt = endtime/num_time_steps
t = 0
eps = 0.025
sigma = 1
rho1 = 2 # "oil"
rho2 = 3 # "water"
nu = 0.005
m0 = 1
f = fem.Constant(msh, ScalarType((0.0, 0.0))) # body forces
# advective CH test
uADV = fem.Constant(msh, ScalarType((0.0, 0.0)))

# Function space CH
P1 = element("Lagrange", msh.basix_cell(), 1)
V = fem.functionspace(msh, mixed_element([P1,P1]))
# Function spaces NS (Taylor-Hood)
Q = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))
L = fem.functionspace(msh, ("Lagrange", 1))


# Test functions CH
v, w = ufl.TestFunction(V)
# Test functions NS
q = ufl.TestFunction(Q)
l = ufl.TestFunction(L)
# Trial funcs for LinearProblem
xi_ = ufl.TrialFunction(V)  # For linearized CH
u_ = ufl.TrialFunction(Q)
p_ = ufl.TrialFunction(L)

# CH test functions
xi = fem.Function(V)
xi_old = fem.Function(V)
phi, mu = ufl.split(xi)
phi_,mu_ = ufl.split(xi_)
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
p_dofs = fem.locate_dofs_geometrical(L, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)) # One point fixed pressure in (0,0) corner
bc_p = [fem.dirichletbc(p_zero, p_dofs, L)]


def doublewell(p):
    return 1/4*(1-p**2)**2

def doublewell_prime(p):
    return p**3 - p



# CH-coupled Navier-Stokes weak form, linear IPCS method
# Step 1: tentative velocity
J = (rho1-rho2)*eps*m0*grad(mu)/2
rho = (1-phi)*rho1/2 + (1+phi)*rho2/2
a1 = (rho*dot(u_, q) + dt * (dot(dot(rho*u_old + J, nabla_grad(u_)), q) + 2*nu*inner(sym(grad(u_)), grad(q))))*ufl.dx
L1 = (rho*dot(u_old, q) + dt * (dot(p_old, div(q)) + dot(f, q) + mu*dot(grad(phi), q)))*ufl.dx # w/ updated Korteweg capillarity term

# Step 2: pressure
a2 = dt * (1/rho)*dot(grad(p_), grad(l))*ufl.dx
L2 = (dt * (1/rho)*dot(grad(p_old), grad(l)) - div(u)*l)*ufl.dx

# Step 3: velocity correction
a3 = rho*dot(u_, q)*ufl.dx
L3 = (rho*dot(u, q) + dt * dot(p - p_old, div(q)))*ufl.dx

# Advective Cahn-Hilliard weak form
def implicit_euler():
    F_phi = ((phi - phi_old)*v - dt * phi*dot(u, grad(v)) + dt * eps*m0*dot(grad(mu), grad(v)))*ufl.dx # Conservative advection form, m = eps*m0 for Case II (AGG)
    F_mu = (mu*w - sigma*doublewell_prime(phi)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
    return F_phi, F_mu, "Implicit Euler", False

def nonlin_convex_split():
    F_phi = ((phi - phi_old)*v - dt * phi*dot(u, grad(v)) + dt * eps*m0*dot(grad(mu), grad(v)))*ufl.dx # Conservative advection form, m = eps*m0 for Case II (AGG)
    F_mu = (mu*w - sigma*(phi**3 - phi_old)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
    return F_phi, F_mu, "Nonlinear Convex Split", False

def lin_convex_split():
    F_phi = ((phi_ - phi_old)*v - dt * phi_*dot(u, grad(v)) + dt * eps*m0*dot(grad(mu_), grad(v)))*ufl.dx # Conservative advection form, m = eps*m0 for Case II (AGG)
    F_mu = (mu_*w - sigma*(2*phi_ + (phi_old)**3 - 3*phi_old)*w/eps - sigma*eps*dot(grad(phi_), grad(w)))*ufl.dx
    return F_phi, F_mu, "Linear Convex Split", True

def lin_convex_split_newt():
    F_phi = ((phi - phi_old)*v - dt * phi*dot(u, grad(v)) + dt * eps*m0*dot(grad(mu), grad(v)))*ufl.dx # Conservative advection form, m = eps*m0 for Case II (AGG)
    F_mu = (mu*w - sigma*(2*phi + (phi_old)**3 - 3*phi_old)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
    return F_phi, F_mu, "Linear Convex Split", False

rng = np.random.default_rng(42)
xi.sub(0).interpolate(lambda x: rng.random(x.shape[1]).clip(-0.5,0.5))
xi.x.scatter_forward()

# Choose time stepping method for CH
F_phi, F_mu, scheme, linear = implicit_euler()

F = F_phi + F_mu


# Problems
if linear == False:
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
            "pc_factor_mat_solver_type": "mumps",
            #"pc_factor_zero_pivot": 1.0e-12,
            "ksp_error_if_not_converged": True,
        },
    )
else:
    problem_CH = LinearProblem(
        ufl.lhs(F), ufl.rhs(F),
        petsc_options_prefix="ch_lin_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    )



problem1 = LinearProblem(
    a1, L1, bcs=bc_u,
    petsc_options_prefix="ns_step1_",
    petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_error_if_not_converged": True}
)

problem2 = LinearProblem(
    a2, L2, bcs=bc_p,
    petsc_options_prefix="ns_step2_",
    petsc_options={"ksp_type": "cg", "pc_type": "ilu", "ksp_error_if_not_converged": True}
)

problem3 = LinearProblem(
    a3, L3, bcs=bc_u,
    petsc_options_prefix="ns_step3_",
    petsc_options={"ksp_type": "cg", "pc_type": "ilu", "ksp_error_if_not_converged": True}
)


# Plotting
V_phi, dofs_phi = V.sub(0).collapse()
V_mu, dofs_mu = V.sub(1).collapse()

cells, types, x = plot.vtk_mesh(V_phi)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["phi"] = xi.x.array[dofs_phi].real
grid.set_active_scalars("phi")

plotter = pyvistaqt.BackgroundPlotter(title=f"CH/NS Phase Plot: {scheme}", auto_update=True)
plotter.add_mesh(grid, clim =[-1, 1], scalar_bar_args={"title": r"$\phi$", "label_font_size": 22, "title_font_size": 28, "vertical": True, "position_x": 0.80, "position_y": 0.15, "height": 0.70})
plotter.view_xy(negative = True)
plotter.add_text(f"time: {t}", font_size=12, name="timelabel")


start = time.perf_counter()
energies = []
glob_phi = []


for i in range(num_time_steps):
    t += dt

    # NS solve
    u_old.x.array[:] = u.x.array[:]
    p_old.x.array[:] = p.x.array[:]
    u_old.x.scatter_forward()
    p_old.x.scatter_forward()
    u.x.array[:] = problem1.solve().x.array
    u.x.scatter_forward()
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
    grid.point_data["phi"] = xi.x.array[dofs_phi].real
    plotter.app.processEvents()
    # time.sleep(0.1)

    # phi conservation test
    print(fem.assemble_scalar(fem.form(phi*ufl.dx)), i)

    if i > 0:
        E = fem.assemble_scalar(fem.form(((sigma/eps)*doublewell(phi) + (sigma*eps/2)*inner(grad(phi),grad(phi)) + (rho/2)*inner(u,u))*ufl.dx)) # CHNS energy
        energies.append(E)
        int_phi = fem.assemble_scalar(fem.form(phi*ufl.dx))
        glob_phi.append(int_phi)

end = time.perf_counter()
print(f"Simulation time: {end-start:.6f}")
np.savetxt(f"output/stagCHNS_energies_{scheme}.txt", energies, header=scheme)

plotter.save_graphic(f"output/chns_plot_t={t:.2f}.pdf")

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
ax.set_title(f"Staggered: {scheme}", fontsize=16)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(f"output/stagCHNS_energy_{scheme}.pdf", dpi=200)
plt.show()



fig, ax = plt.subplots()
ax.plot(t_vals, glob_phi)
ax.grid(True)
ax.set_xlim(0, dt*num_time_steps)
ax.set_ylim(np.min(glob_phi), np.max(glob_phi))

ax.set_xlabel(r"$t$", fontsize=14)
ax.set_ylabel(r"$\int_{\Omega} \phi(t) \, dx$", fontsize=14, labelpad=16)
ax.set_title(f"Staggered: {scheme}", fontsize=16)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(f"output/stagCHNS_phi_{scheme}.pdf", dpi=200)
plt.show()