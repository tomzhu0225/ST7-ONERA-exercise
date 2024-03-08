# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.
from mpi4py import MPI
from dolfinx import io
from dolfinx import fem
import ufl
from basix.ufl import element , mixed_element
from ufl import TrialFunctions , TestFunctions
from ufl import inner, dot, grad, dx, ds
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
import numpy as np

file = io.XDMFFile(MPI.COMM_WORLD, "mesh_out.xdmf", "r")
msh = file.read_mesh()
tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1 , tdim )
facet_tags = file.read_meshtags(msh , "Facet tags")  # try PhysicalNames



V_el = element ("Lagrange", msh.basix_cell() , 1)
Q_el = element ("Lagrange", msh.basix_cell() , 1)
VQ_el = mixed_element ([V_el , Q_el ])
W = fem.functionspace (msh , VQ_el )



(u , p) = TrialFunctions(W)
(v , q) = TestFunctions(W)


#ufl expression
x = ufl.SpatialCoordinate(msh)
u_e1 = ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])
f1 = 2*ufl.pi**2*u_e1

#u_e2 = 1*x[0]**2 * x[1]**2
#f2 = -2 * (x[1]**2 + x[0]**2)
f2 = 1

a = (inner(grad(u), grad(v)) + inner(grad(p), grad(q))) * dx
L = (inner(f1, v) + inner(f2, q) )* dx



fdim = msh.topology.dim - 1
u_D = ScalarType(0.)


# Applying Dirichlet boundary conditions u=0 in the Obstacle 
marker_id = 4 # Obstacle
dofs4_0 = fem.locate_dofs_topological(W.sub(0 ), fdim, facet_tags.find( marker_id ))
bc4_0 = fem.dirichletbc( u_D , dofs4_0 , W.sub(0))

dofs4_1 = fem.locate_dofs_topological(W.sub(1), fdim, facet_tags.find( marker_id ))
bc4_1 = fem.dirichletbc( u_D , dofs4_0 , W.sub(1))

# Applying Dirichlet boundary conditions u=0 in the Walls 
marker_id = 3 
dofs3_0 = fem.locate_dofs_topological(W.sub(0 ), fdim, facet_tags.find( marker_id ))
bc3_0 = fem.dirichletbc( u_D , dofs3_0 , W.sub(0))

dofs3_1 = fem.locate_dofs_topological(W.sub(1), fdim, facet_tags.find( marker_id ))
bc3_1 = fem.dirichletbc( u_D , dofs3_1 , W.sub(1))

# Applying Dirichlet boundary conditions u=0 in the Out Flow 
marker_id = 2 # Out Flow
dofs2_0 = fem.locate_dofs_topological(W.sub(0 ), fdim, facet_tags.find( marker_id ))
bc2_0 = fem.dirichletbc( u_D , dofs2_0 , W.sub(0))

dofs2_1 = fem.locate_dofs_topological(W.sub(1), fdim, facet_tags.find( marker_id ))
bc2_1 = fem.dirichletbc( u_D , dofs2_1 , W.sub(1))

# Applying Dirichlet boundary conditions u=1 in the In Flow 
marker_id = 1 # In Flow
dofs1_0 = fem.locate_dofs_topological(W.sub(0 ), fdim, facet_tags.find( marker_id ))
bc1_0 = fem.dirichletbc( ScalarType(1.) , dofs1_0 , W.sub(0))

dofs1_1 = fem.locate_dofs_topological(W.sub(1), fdim, facet_tags.find( marker_id ))
bc1_1 = fem.dirichletbc( ScalarType(1.) , dofs1_1 , W.sub(1))




problem = LinearProblem(a , L , bcs=[bc4_0, bc3_0, bc2_0, bc1_0, bc4_1, bc3_1, bc2_1, bc1_1])
wh = problem.solve()
(uh , ph ) = wh.split()


# +
with io.XDMFFile(msh.comm, "out_ex_1_2/result.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
    file.write_function(ph)
# -

try:
    import pyvista

    cells, types, x = plot.vtk_mesh(W.sub(1).collapse()[0])  # you can change this sub(1) with sub(1)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = np.split(ph.x.array.real, 2)[1]  # you can change ph with uh, and also [1] with [0]
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
