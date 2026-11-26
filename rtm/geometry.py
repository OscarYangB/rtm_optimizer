"""Gmsh mesh of the rectangular preform with semicircular gate/vent cut-outs.

Facet tags: INLET = 1 (gate arc), OUTLET = 2 (vent arc), WALL = 3 (all other outer edges).
Cell tags : BOTTOM = 1 (y < W/2), TOP = 2 (y > W/2). For the two-region permeability case the
            surface is split along y = W/2 so the K jump lies exactly on element edges.
All coordinates in metres.
"""
import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

INLET, OUTLET, WALL = 1, 2, 3
BOTTOM, TOP = 1, 2
TAG_NAMES = {INLET: "inlet (gate)", OUTLET: "outlet (vent)", WALL: "wall (∂p/∂n = 0)"}


def build_mesh(cfg, comm=MPI.COMM_WORLD):
    """Return (mesh, cell_tags, facet_tags)."""
    L, W = cfg.L, cfg.W
    rg, rv = cfg.gate_diameter / 2, cfg.vent_diameter / 2
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("preform")
    if comm.rank == 0:
        occ = gmsh.model.occ
        rect = occ.addRectangle(0, 0, 0, L, W)
        gate = occ.addDisk(0, W / 2, 0, rg, rg)
        vent = occ.addDisk(L, W / 2, 0, rv, rv)
        out, _ = occ.cut([(2, rect)], [(2, gate), (2, vent)])
        if cfg.two_region:
            # interface line gate-arc -> vent-arc along y = W/2 splits the surface in two
            p1, p2 = occ.addPoint(rg, W / 2, 0), occ.addPoint(L - rv, W / 2, 0)
            line = occ.addLine(p1, p2)
            out, _ = occ.fragment(out, [(1, line)])
            out = [e for e in out if e[0] == 2]
        occ.synchronize()

        bottom, top = [], []
        for _, s in out:
            (top if occ.getCenterOfMass(2, s)[1] > W / 2 else bottom).append(s)
        inlet, outlet, wall = [], [], []
        for _, c in gmsh.model.getBoundary(out, combined=True, oriented=False):
            x, y, _ = occ.getCenterOfMass(1, c)
            if abs(x) < rg and abs(y - W / 2) < rg:
                inlet.append(c)
            elif abs(x - L) < rv and abs(y - W / 2) < rv:
                outlet.append(c)
            else:
                wall.append(c)
        if bottom:
            gmsh.model.addPhysicalGroup(2, bottom, BOTTOM, name="bottom")
        if top:
            gmsh.model.addPhysicalGroup(2, top, TOP, name="top")
        gmsh.model.addPhysicalGroup(1, inlet, INLET, name="inlet")
        gmsh.model.addPhysicalGroup(1, outlet, OUTLET, name="outlet")
        gmsh.model.addPhysicalGroup(1, wall, WALL, name="wall")

        # size field: fine at the ports, coarse elsewhere
        f = gmsh.model.mesh.field
        d = f.add("Distance")
        f.setNumbers(d, "CurvesList", inlet + outlet)
        f.setNumber(d, "Sampling", 200)
        t = f.add("Threshold")
        f.setNumber(t, "InField", d)
        f.setNumber(t, "SizeMin", cfg.h_port)
        f.setNumber(t, "SizeMax", cfg.h_far)
        f.setNumber(t, "DistMin", 0.0)
        f.setNumber(t, "DistMax", cfg.refine_radius)
        f.setAsBackgroundMesh(t)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay: well-shaped triangles
        gmsh.model.mesh.generate(2)
    data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return data.mesh, data.cell_tags, data.facet_tags
