from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import ufl
import pyvista
from basix.ufl import element
from dolfinx.plot import vtk_mesh

# TODO: Solve volumetric flux field using velocity field and area of each control volume
# TODO: VOF or FAN method
# TODO: Anisotropic (non-scalar) K

domain = mesh.create_unit_square(MPI.COMM_WORLD, 100, 100, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

tdim = domain.topology.dim
fdim = tdim - 1

inlet_position = (0.40, 0.40)
outlet_position = (0.60, 0.60)
tube_radius = 0.05

def is_tube_boundary(positions, tube_position, radius):
    return np.where(np.sqrt((positions[0] - tube_position[0]) ** 2 + 
                            (positions[1] - tube_position[1]) ** 2) < radius, True, False)

inlet_pressure = fem.Function(V)
inlet_pressure.interpolate(lambda x: x[1]*0 + 10.0)
dofs_inlet = fem.locate_dofs_geometrical(V, lambda x: is_tube_boundary(x, inlet_position, tube_radius))
bc_inlet = fem.dirichletbc(inlet_pressure, dofs_inlet)

outlet_pressure = fem.Function(V)
outlet_pressure.interpolate(lambda x: x[1]*0 + 0.0)
dofs_outlet = fem.locate_dofs_geometrical(V, lambda x: is_tube_boundary(x, outlet_position, tube_radius))
bc_oulet = fem.dirichletbc(outlet_pressure, dofs_outlet)

k = fem.Constant(domain, default_scalar_type(6e-8))
mu = fem.Constant(domain, default_scalar_type(0.5))

p = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = -ufl.dot(k/mu * ufl.grad(p), ufl.grad(v)) * ufl.dx
x = ufl.SpatialCoordinate(domain)
g = fem.Constant(domain, default_scalar_type(0))
f = fem.Constant(domain, default_scalar_type(0))
L = f * v * ufl.dx - g * v * ufl.ds

problem = LinearProblem(
    a,
    L,
    bcs=[bc_inlet, bc_oulet],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

W = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,)))
velocity_expression = fem.Expression(-ufl.grad(uh) * k/mu * 10000, W.element.interpolation_points)
velocity_field = fem.Function(W)
velocity_field.interpolate(velocity_expression)


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




topology, cell_types, geometry = vtk_mesh(W)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, : len(velocity_field)] = velocity_field.x.array.real.reshape((geometry.shape[0], len(velocity_field)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)

# Create a pyvista-grid for the mesh
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")