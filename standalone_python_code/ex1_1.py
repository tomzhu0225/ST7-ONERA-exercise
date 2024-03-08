# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

# +
import numpy as np

from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

import ufl
from ufl import TrialFunction, TestFunction
from ufl import inner, dot, grad, dx, ds

import matplotlib.pyplot as plt
import time
# create a rectangular mesh [0,1]x[0,1] with n subdivisions (ex. n=(16,16) means 16 elements in x-direction and 16 elements in y-direction) of triangles
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(16, 16),
    cell_type=mesh.CellType.triangle,
)

# Create a finite element FunctionSpace $V$ on the mesh.
# The first argument is the mesh created above
# The second argument is a tuple `(family, degree)`, where
# `family` is the finite element family, and `degree` specifies the
# polynomial degree. In this case `V` is a space of continuous Lagrange
# finite elements of degree 1.
V = fem.functionspace(msh, ("Lagrange", 1))

# To apply the Dirichlet boundary conditions, we find the edges of the mesh (called facets)
# that lie on the boundary $\partial \Gamma$ using
# <dolfinx.mesh.locate_entities_boundary>`. The function is provided
# with a 'marker' function that returns `True` for points `x` on the
# boundary and `False` otherwise.
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))),
)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using <dolfinx.fem.locate_dofs_topological>
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# And set Dirichlet boundary conditions on the boundary degrees of freedom with
# Here, the first argument specifies the value to set on the boundary (in this case 0)
# the second argument are the degrees of freedom and the third is the function space
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Next, the variational problem is defined:
u = TrialFunction(V)
v = TestFunction(V)

#ufl expression
x = ufl.SpatialCoordinate(msh)
u_e = ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])
f = 2*ufl.pi**2*u_e

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# A {py:class}object is
# created that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case an LU solver is used. The {py:func}`solve
# <dolfinx.fem.petsc.LinearProblem.solve>` computes the solution.

# We collect the variational problem and its boundary conditions
# into a `LinearProblem <dolfinx.fem.petsc.LinearProblem>`
# and we specify the solver
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# The solution can be written to a XDMF File
# for visualization with ParaView or VisIt:

# +
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
# -

# and displayed using [pyvista](https://docs.pyvista.org/).
comm = uh.function_space.mesh.comm

eh = uh - u_e

error = fem.form((eh)**2 * dx)
E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
if comm.rank == 0:
    print(f"L2-error: {E:.2e}")


error_H1 = fem.form(dot(grad(eh), grad(eh)) * dx + (eh)**2 * dx)
E_H1 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_H1), op=MPI.SUM))
if comm.rank == 0:
    print(f"H01-error: {E_H1:.2e}")


def solve_poisson(N=16, degree=1):

    msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(msh, ("Lagrange", degree))

    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

    u = TrialFunction(V)
    v = TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    u_e = ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])
    f = 2*ufl.pi**2*u_e

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh, u_e



def error_l2(eh):
    error = fem.form((eh)**2 * dx)
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    return E

def error_h1(eh):
    error_H1 = fem.form(dot(grad(eh), grad(eh)) * dx + (eh)**2 * dx)
    E_H1 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_H1), op=MPI.SUM))
    return E_H1
    

Ns = np.arange(2, 100, 10)
Es_l2 = np.zeros(len(Ns), dtype=default_scalar_type)
Es_h1 = np.zeros(len(Ns), dtype=default_scalar_type)

hs = np.zeros(len(Ns), dtype=np.float64)


def u_ex(mod):
    return lambda x: mod.sin(mod.pi * x[0]) * mod.cos(mod.pi * x[1])

u_numpy = u_ex(np)

for i, N in enumerate(Ns):
    uh, u_e = solve_poisson(N, degree=1)
    comm = uh.function_space.mesh.comm
    # One can send in either u_numpy or u_ex
    # For L2 error estimations it is reccommended to send in u_numpy
    # as no JIT compilation is required
    Es_l2[i] = error_l2((uh - u_e))
    Es_h1[i] = error_h1((uh - u_e))
    hs[i] = 1. / Ns[i]


#plt.figure()
#plt.plot(np.log(hs), np.log(Es_l2))
#plt.plot(np.log(hs), np.log(Es_h1))
#plt.legend(["l2 norm", "h1 norm"])
#plt.title("degree=1")
#plt.show()

Ns = np.arange(2, 100, 10)
Es_l2_2 = np.zeros(len(Ns), dtype=default_scalar_type)
Es_h1_2 = np.zeros(len(Ns), dtype=default_scalar_type)
hs = np.zeros(len(Ns), dtype=np.float64)


for i, N in enumerate(Ns):
    uh, u_e = solve_poisson(N, degree=2)
    comm = uh.function_space.mesh.comm
    # One can send in either u_numpy or u_ex
    # For L2 error estimations it is reccommended to send in u_numpy
    # as no JIT compilation is required
    Es_l2_2[i] = error_l2((uh - u_e))
    Es_h1_2[i] = error_h1((uh - u_e))
    hs[i] = 1. / Ns[i]


#plt.figure()
#plt.plot(np.log(hs), np.log(Es_l2_2))
#plt.plot(np.log(hs), np.log(Es_h1_2))
#plt.legend(["l2 norm", "h1 norm"])
#plt.title("degree=2")
#plt.show()


plt.figure()
plt.plot(np.log(hs), np.log(Es_l2))
plt.plot(np.log(hs), np.log(Es_h1))
plt.plot(np.log(hs), np.log(Es_l2_2))
plt.plot(np.log(hs), np.log(Es_h1_2))
plt.legend(["l2 d=1", "h1 d=1", "l2 d=2", "h1 d=2"])
plt.title("comparison between Lagrange's degree 1 and 2")
plt.show()


Ns = [5, 10, 20, 40]
n = np.zeros(len(Ns), dtype=np.float64)
t = np.zeros(len(Ns), dtype=default_scalar_type)
for i, N in enumerate(Ns):
    t0 = time.time()
    uh, u_e = solve_poisson(N, degree=1)
    t[i] = time.time() - t0
    n[i] = N



plt.figure()
plt.plot(n, t)
plt.title("time x degrees of freedom")
plt.show()  

"""
# +

try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# -
"""