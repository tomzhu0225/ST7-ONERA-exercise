from ex3_1 import create_NACA_mesh
from compute_stokes import solve_stokes
import pandas as pd

cambers = ["2412", "6112", "9912", "0012", "8812"]

angles = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70,
          80, 90]

# angles = [0, 20, 40, 60, 100, 140, 170]

camber_values = []
angle_values = []
j_values = []
fd_values = []
fl_values = []


for camber in cambers:
    for angle in angles:
        mesh, facets = create_NACA_mesh(angle, camber)

        j, fd, fl, u, p = solve_stokes(mesh, facets)
        camber_values.append(camber)
        angle_values.append(angle)
        j_values.append(j)
        fd_values.append(fd)
        fl_values.append(fl)
        print(f"{camber} - {angle}")


dict_to_pandas = {'camber': camber_values, 'angle': angle_values, 'j': j_values, 'fl': fl_values, 'fd': fd_values}


df = pd.DataFrame(dict_to_pandas)
df.to_csv("results_ex3_2.csv")







