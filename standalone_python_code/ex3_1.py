
import gmsh
import sys
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import io
from mpi4py import MPI



def rotate(x, y, theta):
    """Rotate points around the origin by theta degrees."""
    theta_rad = np.radians(theta)
    xr = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    yr = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    return xr, yr

def naca4_full(number, chord=0.1, n=50, angle_of_attack=0):
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0
    
    x = np.linspace(0, 1, n)
    x = (0.5 * (1 - np.cos(np.pi * x))) * chord
    
    yt = 5 * t * chord * (0.2969 * np.sqrt(x/chord) - 0.1260 * (x/chord) - 0.3516 * (x/chord)**2 + 0.2843 * (x/chord)**3 - 0.1015 * (x/chord)**4)
    
    yc = np.where(x < p * chord, m * (x / np.power(p, 2)) * (2 * p - (x / chord)), m * ((chord - x) / np.power(1-p, 2)) * (1 + (x / chord) - 2 * p))
    dyc_dx = np.where(x < p * chord, 2*m / np.power(p, 2) * (p - x / chord), 2*m / np.power(1-p, 2) * (p - x / chord))
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Rotate coordinates for angle of attack
    xu, yu = rotate(xu, yu, angle_of_attack)
    xl, yl = rotate(xl, yl, angle_of_attack)
    
    # Combine upper and lower surfaces into one loop without duplicating the trailing edge
    x_full = np.concatenate([xu[:-1], xl[::-1]])[:-1]
    y_full = np.concatenate([yu[:-1], yl[::-1]])[:-1]
    
    return x_full, y_full






def create_NACA_mesh(theta , camber: str):
    """
    theta: angle in degrees
    camber: string of 4 digits
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("mesh")

    # Define the geometry
    lc = 0.1  # characteristic length for mesh size

    # Add points for the rectangle
    p1 = gmsh.model.geo.addPoint(-1, -1, 0, lc)
    p2 = gmsh.model.geo.addPoint(2, -1, 0, lc)
    p3 = gmsh.model.geo.addPoint(2, 2, 0, lc)
    p4 = gmsh.model.geo.addPoint(-1, 2, 0, lc)



    # Add lines for the rectangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    center_x=0.7
    center_y=0.4
    x,y= naca4_full(camber,chord=0.7, angle_of_attack=theta)  # 6312, 10
    x,y=x+center_x,y+center_y

    airfoil_points = []
    for xi, yi in zip(x, y):
        pid = gmsh.model.geo.addPoint(xi, yi, 0, lc/10)
        airfoil_points.append(pid)

    # Add lines for the airfoil
    airfoil_lines = []
    for i in range(len(airfoil_points) - 1):
        lid = gmsh.model.geo.addLine(airfoil_points[i], airfoil_points[i + 1])
        airfoil_lines.append(lid)
    # Close the airfoil loop by connecting the last point back to the first
    airfoil_lines.append(gmsh.model.geo.addLine(airfoil_points[-1], airfoil_points[0]))





    # Create curve loops and plane surfaces
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    wing_loop = gmsh.model.geo.addCurveLoop(airfoil_lines)
    plane_surface = gmsh.model.geo.addPlaneSurface([outer_loop, wing_loop])


    # Synchronize necessary before meshing
    gmsh.model.geo.synchronize()

    # Define Physical Groups for boundaries
    fluid = gmsh.model.addPhysicalGroup(2, [plane_surface], 0)  # Fluid
    gmsh.model.setPhysicalName(1, fluid, "Fluid")

    inflow = gmsh.model.addPhysicalGroup(1, [l4], 1)  # Inflow
    gmsh.model.setPhysicalName(1, inflow, "Inflow")

    outflow = gmsh.model.addPhysicalGroup(1, [l2], 2)  # Outflow
    gmsh.model.setPhysicalName(1, outflow, "Outflow")

    walls = gmsh.model.addPhysicalGroup(1, [l3, l1], 3)  # Walls
    gmsh.model.setPhysicalName(1, walls, "Walls")

    obstacle = gmsh.model.addPhysicalGroup(1, airfoil_lines, 4)
    gmsh.model.setPhysicalName(1, obstacle, "Obstacle")

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1 , tdim )



    gmsh.finalize()

    return mesh, facet_tags



if __name__=="__main__":
    create_NACA_mesh(10, '6312')