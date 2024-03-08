from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from dolfinx import cpp as _cpp

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner
from dolfinx import io
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem

# Create mesh
file = io.XDMFFile(MPI.COMM_WORLD, "worksheet2/mesh_out.xdmf", "r")
msh = file.read_mesh()
tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1 , tdim )
facet_tags = file.read_meshtags(msh , "Facet tags")  # try PhysicalNames


P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.basix_cell(), 1)


# Create the Taylot-Hood function space
TH = mixed_element([P2, P1])
W = functionspace(msh, TH)

# No slip boundary condition
W0, _ = W.sub(0).collapse()



fdim = msh.topology.dim - 1
noslip = Function(W0)

# Applying Dirichlet boundary conditions u=(0, 0) in the Obstacle 

marker_id = 4 # Obstacle
dofs4 = fem.locate_dofs_topological((W.sub(0), W0), fdim, facet_tags.find( marker_id ))
bc4 = dirichletbc(noslip, dofs4, W.sub(0))

# Applying Dirichlet boundary conditions u=(0, 0) in the Walls 

marker_id = 3 # Walls
dofs3 = fem.locate_dofs_topological((W.sub(0), W0), fdim, facet_tags.find( marker_id ))
bc3 = dirichletbc(noslip, dofs3,  W.sub(0))

def lid_velocity_expression(x):
    return np.stack((-1/24*(x[1, :]- 6)*(x[1, :] + 6), np.zeros(x.shape[1])))
    # return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

lid_velocity = Function(W0)
lid_velocity.interpolate(lid_velocity_expression)

marker_id = 1 # In Flow
dofs1 = fem.locate_dofs_topological((W.sub(0), W0), fdim, facet_tags.find( marker_id ))
bc1 = dirichletbc(lid_velocity, dofs1, W.sub(0))


bcs = [bc4, bc3, bc1]

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(W0)
a = form((inner(grad(u), grad(v)) + inner(p, div(v)) + inner(div(u), q)) * dx)
L = form(inner(f, v) * dx)


# Assemble LHS matrix and RHS vector
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L)

fem.petsc.apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


# Set Dirichlet boundary condition values in the RHS
fem.petsc.set_bc(b, bcs)

# Create and configure solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")

# Configure MUMPS to handle pressure nullspace
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)


# Compute the solution
U = Function(W)
try:
    ksp.solve(b, U.vector)
except PETSc.Error as e:
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e
    



# Split the mixed solution and collapse
u, p = U.sub(0).collapse(), U.sub(1).collapse()

# Compute norms
norm_u, norm_p = u.x.norm(), p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print(f"(D) Norm of velocity coefficient vector (monolithic, direct): {norm_u}")
    print(f"(D) Norm of pressure coefficient vector (monolithic, direct): {norm_p}")


with XDMFFile(MPI.COMM_WORLD, "worksheet2/out_stokes/velocity.xdmf", "w") as ufile_xdmf:
        u.x.scatter_forward()
        P1 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
        u1 = Function(functionspace(msh, P1))
        u1.interpolate(u)
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u1)

with XDMFFile(MPI.COMM_WORLD, "worksheet2/out_stokes/pressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p)