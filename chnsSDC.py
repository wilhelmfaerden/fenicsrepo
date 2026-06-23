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
endtime = 6.0
num_time_steps = 50
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

# comp. variables
J = (rho1-rho2)*eps*m0*grad(mu)/2
rho = (1-phi)*rho1/2 + (1+phi)*rho2/2

# Driven lid cavity boundary conditions
def walls(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[0], 1.0)
    )

def lid(x):
    return np.isclose(x[1], 1.0)


def doublewell(p):
    return 1/4*(1-p**2)**2


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


rng = np.random.default_rng(42)
xi.sub(0).interpolate(lambda x: rng.random(x.shape[1]).clip(-0.5,0.5))
xi.x.scatter_forward()

# Plotting
V_phi, dofs_phi = V.sub(0).collapse()
V_mu, dofs_mu = V.sub(1).collapse()
phi_exact_P1 = fem.Function(V_phi)

cells, types, x = plot.vtk_mesh(V_phi)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["phi"] = xi.x.array[dofs_phi].real
grid.set_active_scalars("phi")

plotter = pyvistaqt.BackgroundPlotter(title=f"SDC CHNS Phase Field", auto_update=True)
plotter.add_mesh(grid, clim =[-1, 1], scalar_bar_args={"title": r"$\phi$", "label_font_size": 22, "title_font_size": 28, "vertical": True, "position_x": 0.80, "position_y": 0.15, "height": 0.70})
plotter.view_xy(negative = True)
plotter.add_text(f"time: {t}", font_size=12, name="timelabel")


start = time.perf_counter()
energies = []
glob_phi = []


for i in range(num_time_steps):
    t += dt

    """
    # NS solve
    u_old.x.array[:] = u.x.array[:]
    p_old.x.array[:] = p.x.array[:]
    u_old.x.scatter_forward()
    p_old.x.scatter_forward()
    u.x.array[:] = problem1.solve().x.array
    u.x.scatter_forward()
    #print(fem.assemble_scalar(fem.form(inner(u,u)*ufl.dx)))
    p.x.array[:] = problem2.solve().x.array
    p.x.scatter_forward()
    u.x.array[:] = problem3.solve().x.array
    u.x.scatter_forward()
    """

    # SDC CH solve
    xi_old.x.array[:] = xi.x.array
    xi_old.x.scatter_forward()

    # 4 node Gauss-Lobato
    dtms = [(0.5 - np.sqrt(5)/10)*dt, (np.sqrt(5)/5)*dt, (0.5 - np.sqrt(5)/10)*dt]
    A = np.array([
            [0, 0, 0, 0],
            [378/3427, 221/1165, -317/9349, 61/5922],
            [439/6011, 1609/3571, 574/2529, -256/9493],
            [1/12, 5/12, 5/12, 1/12],
        ])
    qms_delta = A[1:, :] - A[:-1, :]
    

    xi_stages = [fem.Function(V) for _ in range(len(dtms)+1)]
    xi_stages[0].x.array[:] = xi_old.x.array
    xi_stages[0].x.scatter_forward()

    xi_old_k_new = [fem.Function(V) for _ in range(len(dtms)+1)]
    F_stages_new = []

    sweeps = 4
    for k in range(sweeps):
        F_stages = F_stages_new
        F_stages_new = []
        xi_old_k = xi_old_k_new
        xi_old_k_new = [fem.Function(V) for _ in range(len(dtms)+1)]

        m = 0

        phi_old_m, muI_old_m = ufl.split(xi_stages[0])
        muE = sigma*(-phi_old_m)/eps
        mu_old_m = muI_old_m + muE

        F_stages_new.append(-(eps*m0*dot(grad(mu_old_m),grad(v)))*ufl.dx + phi_old_m*dot(u, grad(v))*ufl.dx)
        xi_old_k_new[0].x.array[:] = xi_stages[0].x.array 
        xi_old_k_new[0].x.scatter_forward()

        for dtm, qm in zip(dtms, qms_delta):
            
            # Predictor
            RHS = phi_old_m*v*ufl.dx - dtm*(eps*m0*dot(grad(muE),grad(v)))*ufl.dx
            
            # Corrector
            if k > 0:
                phi_old_k, muI_old_k = ufl.split(xi_old_k[m+1])
                phi_old_km, muI_old_km = ufl.split(xi_old_k[m])
                muE_diff = dtm*(eps*m0*dot(grad(sigma*(-phi_old_m + phi_old_km)/eps),grad(v)))*ufl.dx
                RHS = phi_old_m*v*ufl.dx - muE_diff - dtm*phi_old_k*dot(uADV, grad(v))*ufl.dx + dtm*(eps*m0*dot(grad(muI_old_k),grad(v)))*ufl.dx + dt*sum(qm[j]*(F_stages[j]) for j in range(len(F_stages)))

            # Weak form
            F_phi = (phi*v - dtm*phi*dot(uADV, grad(v)) + dtm*(eps*m0*(dot(grad(mu), grad(v)))))*ufl.dx - RHS
            F_mu = (mu*w - sigma*(phi**3)*w/eps - sigma*eps*dot(grad(phi), grad(w)))*ufl.dx
            
            F = F_phi + F_mu

            problem_CH_SDC = NonlinearProblem(
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

            # Solve and store stage solution, and F form
            xi_stages[m+1].x.array[:] = problem_CH_SDC.solve().x.array
            xi_stages[m+1].x.scatter_forward()

            # add F_stage and phi
            phi_old_m, muI_old_m = ufl.split(xi_stages[m+1])
            muE = sigma*(-phi_old_m)/eps
            mu_old_m = muI_old_m + muE

            F_stages_new.append(-(eps*m0*dot(grad(mu_old_m),grad(v)))*ufl.dx + phi_old_m*dot(u, grad(v))*ufl.dx)
            xi_old_k_new[m+1].x.array[:] = xi_stages[m+1].x.array 
            xi_old_k_new[m+1].x.scatter_forward()

            m += 1
    
    # Take final stage as CH solve output
    xi.x.array[:] = xi_stages[len(dtms)].x.array
    #xi.x.array[dofs_mu] += -(sigma/eps)*xi.x.array[dofs_phi] # adding muE back to muI to output mu = muE + muI
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
np.savetxt(f"output/SDC_CHNS_energies.txt", energies)

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
ax.set_title(f"SDC", fontsize=16)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(f"output/SDC_CHNS_energy.pdf", dpi=200)
plt.show()



fig, ax = plt.subplots()
ax.plot(t_vals, glob_phi)
ax.grid(True)
ax.set_xlim(0, dt*num_time_steps)
ax.set_ylim(np.min(glob_phi), np.max(glob_phi))

ax.set_xlabel(r"$t$", fontsize=14)
ax.set_ylabel(r"$\int_{\Omega} \phi(t) \, dx$", fontsize=14, labelpad=16)
ax.set_title(f"SDC", fontsize=16)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(f"output/SDC_CHNS_phi.pdf", dpi=200)
plt.show()