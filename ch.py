# Fix MPI/OFI finalization errors on macOS test
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"


from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io, la

from basix.ufl import element, mixed_element


from dolfinx.fem.petsc import NonlinearProblem

import ufl
from ufl import grad, inner, dx
import time

import pyvista
import pyvistaqt


# Mesh
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(64, 64),
    cell_type=mesh.CellType.triangle,
)

dt = 0.001
num_time_steps = 100
t = 0
eps = 0.01
m = 1

# Function space
P1 = element("Lagrange", msh.basix_cell(), 1)

V = fem.functionspace(msh, mixed_element([P1,P1]))

v, w = ufl.TestFunction(V)

u = fem.Function(V)
u_old = fem.Function(V)
phi, mu = ufl.split(u)
phi_old, mu_old = ufl.split(u_old)

def doublewell(p):
    return 1/4*(1-p**2)**2

def doublewell_prime(p):
    return -(1-p**2)*p

rng = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: rng.random(x.shape[1]).clip(-0.5,0.5))
u.x.scatter_forward()

# Variational form
F_phi = (ufl.inner(phi, v)  + eps*m*dt * ufl.inner(ufl.grad(mu), ufl.grad(v)) - ufl.inner(phi_old, v))*ufl.dx
F_mu = (ufl.inner(mu, w) - ufl.inner(doublewell_prime(phi), w)/eps - ufl.inner(ufl.grad(phi), grad(w))*eps)*ufl.dx

F = F_phi +F_mu

# Problem
problem = NonlinearProblem(
    F,
    u,
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


###############################
# Plotting with pyvista
###############################

V_phi, dofs = V.sub(0).collapse()

cells, types, x = plot.vtk_mesh(V_phi)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["phi"] = u.x.array[dofs].real
grid.set_active_scalars("phi")

p = pyvistaqt.BackgroundPlotter(title="phi", auto_update=True)
p.add_mesh(grid, clim =[0, 1])
p.view_xy(negative = True)
p.add_text(f"time: {t}", font_size=12, name="timelabel")


for i in range(num_time_steps):
    t += dt  # Update the time constant
    u_old.x.array[:] = u.x.array
    u_old.x.scatter_forward()
    u = problem.solve()
    u.x.scatter_forward() 
    # Add scalar data to grid
    p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
    grid.point_data["phi"] = u.x.array[dofs].real
    p.app.processEvents()
    # time.sleep(0.1)