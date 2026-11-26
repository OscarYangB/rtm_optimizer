"""Darcy pressure solve (dolfinx, P1 Lagrange).

Strong form solved (Okabe Eq. 3 with an optional source for MMS):
        -div( (K/mu) grad p ) = f              in Omega
                            p = p_D            on Gamma_D   (gate, vent, flow front)
              (K/mu) grad p . n = g            on Gamma_N   (g = 0 on the mould walls)
Darcy velocity: u = -(K/mu) grad p   [m/s].  Note g = -u.n, i.e. g is the INFLOW flux.

`solve_pressure` is the stand-alone entry point for the team's MMS study: it takes the mesh,
K, mu, an optional source f and Dirichlet/Neumann data, and returns p_h.  The FAN filling
loop (fan.py) uses `assemble_cv_operator`, which assembles the SAME bilinear form once.
"""
import numpy as np
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


def _as_tensor(msh, K):
    """K: scalar, 2x2 nested list/array, or a UFL expression -> UFL 2x2 tensor [m^2]."""
    if isinstance(K, ufl.core.expr.Expr):
        return K if K.ufl_shape == (2, 2) else K * ufl.Identity(2)
    K = np.asarray(K, dtype=float)
    if K.ndim == 0:
        K = K * np.eye(2)
    return fem.Constant(msh, default_scalar_type(K))


def _as_coeff(msh, V, value):
    if value is None or isinstance(value, (ufl.core.expr.Expr, fem.Function)):
        return value
    if callable(value):
        fn = fem.Function(V)
        fn.interpolate(value)
        return fn
    return fem.Constant(msh, default_scalar_type(value))


def darcy_bilinear_form(V, K, mu):
    """a(p, v) = int (K/mu) grad p . grad v dx   (units: m^3/(Pa s) * Pa -> m^2/s per m)."""
    msh = V.mesh
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    Kt = _as_tensor(msh, K)
    mu_c = mu if isinstance(mu, ufl.core.expr.Expr) else fem.Constant(msh, default_scalar_type(mu))
    return ufl.inner((Kt / mu_c) * ufl.grad(p), ufl.grad(v)) * ufl.dx, Kt, mu_c


def solve_pressure(msh, K, mu, dirichlet, f=None, neumann=None, facet_tags=None,
                   degree=1, petsc_options=None):
    """Solve the Darcy pressure problem and return p_h (fem.Function).

    msh        : dolfinx mesh (coordinates in m)
    K          : permeability [m^2] - scalar, 2x2 array, or UFL expression (anisotropic OK)
    mu         : viscosity [Pa s] - float or UFL expression
    dirichlet  : list of (where, value). `where` is a facet-tag int (needs facet_tags), an array
                 of dofs, or a callable marker x -> bool (geometric). `value` is a float,
                 callable x -> values, UFL expression or fem.Function.  [Pa]
    f          : optional source [1/s] (float, callable, or UFL expr) - for MMS
    neumann    : optional list of (facet_tag, g); g = (K/mu) grad p . n [m/s]. Unlisted
                 boundaries are natural no-flux walls (g = 0).
    """
    V = fem.functionspace(msh, ("Lagrange", degree))
    fdim = msh.topology.dim - 1
    a, _, _ = darcy_bilinear_form(V, K, mu)
    v = ufl.TestFunction(V)
    f_c = _as_coeff(msh, V, f if f is not None else 0.0)
    L = f_c * v * ufl.dx
    if neumann:
        ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
        for tag, g in neumann:
            L += _as_coeff(msh, V, g) * v * ds(tag)

    bcs = []
    for where, value in dirichlet:
        if isinstance(where, (int, np.integer)):
            dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(where))
        elif callable(where):
            dofs = fem.locate_dofs_geometrical(V, where)
        else:
            dofs = np.asarray(where, dtype=np.int32)
        if isinstance(value, ufl.core.expr.Expr) and not isinstance(value, fem.Function):
            g = fem.Function(V)
            g.interpolate(fem.Expression(value, V.element.interpolation_points))
            value = g
        val = _as_coeff(msh, V, value)
        bcs.append(fem.dirichletbc(val, dofs) if isinstance(val, fem.Function)
                   else fem.dirichletbc(val, dofs, V))

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options=petsc_options or {"ksp_type": "preonly", "pc_type": "lu"},
                            petsc_options_prefix="darcy_")
    return problem.solve()


def darcy_velocity(p_h, K, mu):
    """u = -(K/mu) grad p as a DG0 vector field [m/s] (exact for P1 pressure)."""
    msh = p_h.function_space.mesh
    Kt = _as_tensor(msh, K)
    W = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
    u = fem.Function(W)
    u.interpolate(fem.Expression(-(Kt / mu) * ufl.grad(p_h), W.element.interpolation_points))
    return u


def assemble_cv_operator(V, K, mu):
    """Return (A, area) for the FE/CV (median-dual) front tracker.

    A    : scipy CSR stiffness of a(p, v) WITHOUT boundary conditions [m^2/(Pa s)]
    area : median-dual control-volume area of each P1 node, |CV_i| = int phi_i dx [m^2]

    For P1 triangles, (A p)_i equals the net Darcy OUTFLOW through the boundary of the
    median-dual CV of node i (per unit thickness, m^2/s), so Q_i = -(A p)_i is a flux that
    is exactly conservative across CV faces (sum over all CVs telescopes to the boundary flux).
    """
    a, _, _ = darcy_bilinear_form(V, K, mu)
    A = fem.assemble_matrix(fem.form(a)).to_scipy().tocsr()
    v = ufl.TestFunction(V)
    area = fem.assemble_vector(fem.form(v * ufl.dx)).array.copy()
    return A, area
