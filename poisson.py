from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import ufl
import pyvista

domain = mesh.create_unit_square(MPI.COMM_WORLD, 100, 100, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 10 * (abs(x[0]-0.5) < 0.2))
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

k=fem.Constant(domain, default_scalar_type(6e-8))
mu=fem.Constant(domain, default_scalar_type(0.5))

p = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type(0))
a = -ufl.dot(k/mu * ufl.grad(p), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
root = 0
assert root < domain.comm.size, f"Cannot gather on process {root} of {domain.comm.size}"
pieces = domain.comm.gather(grid, root=root)
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_clim = [
    domain.comm.reduce(uh.x.array.real.min(), op=MPI.MIN, root=root),
    domain.comm.reduce(uh.x.array.real.max(), op=MPI.MAX, root=root),
]
u_pieces = domain.comm.gather(u_grid, root=root)
if u_pieces is not None:
    merged_u_grid = pyvista.merge(u_pieces)
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(merged_u_grid, show_edges=True, clim=u_clim)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()