import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from extract_vtu_data import extract_vtu_data

"""
File to evaluate the velocity field of a flow over a flat plate

"""
############################################################################################################
###  Import file and extract data
############################################################################################################
# define boundary conditions 
U_INF = 1.0
MU_CONSTANT= 1.83463e-05
RHO = 1.13235
NU = MU_CONSTANT / RHO

# define files 
filename = "Inc_Laminar_Flat_Plate/flow.vtu"
data = extract_vtu_data(filename)

# extract data
WIDTH = 65      # hard coded from mesh topology
HEIGHT = 65    # hard coded from mesh topology

coordinate = data['Coordinates']
pressure = data['Pressure']
velocity = data['Velocity']
temperature = data['Temperature']
pressure_coefficient = data['Pressure_Coefficient']
density = data['Density']
laminar_viscosity = data['Laminar_Viscosity']
heat_capacity = data['Heat_Capacity']
thermal_conductivity = data['Thermal_Conductivity']
skin_friction_coefficient = data['Skin_Friction_Coefficient']
heat_flux = data['Heat_Flux']
y_plus = data['Y_Plus']


# Blasius function derivative
def blasius_derivative(eta):
    return 0.332 * np.sqrt(eta) - 0.4167 * eta + 0.0833 / (1 + eta)**(1.5)


############################################################################################################
###  Evaluate the velocity field at specific x location 
############################################################################################################

# define the index of x loc for the velocity field extraction 

x_index =  30
if x_index > WIDTH:
    print(f"Index out of range, max index is {WIDTH}")
    sys.exit(1)


# interpolate the node index of given x_index
column_start = x_index * HEIGHT
column_end = column_start + HEIGHT

# extract the velocity field at the x_index
# for Blasius boundary layer we are only interested in u
u = velocity[column_start:column_end, 0]
v = velocity[column_start:column_end, 1]
w = velocity[column_start:column_end, 2]

print(f"Extracting data at x = {coordinate[column_start, 0]}")
y_val = coordinate[column_start:column_end, 1]
x_val = coordinate[column_start:column_end, 0]
# plot the coordinate
plt.figure()
plt.plot(x_val, y_val, 'k.')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Flat Plate Mesh Verify')
plt.show()

# plot the velocity field
plt.figure()
plt.plot(u, y_val, 'k.')
plt.xlabel('U')
plt.ylabel('Y')
plt.title('Velocity')
plt.show()


# compute Blasius analytical solution
eta = y_val / np.sqrt(NU * x_val[1]/ U_INF)
u_blasius = U_INF*blasius_derivative(eta)


# Plotting
plt.figure()
plt.plot(u, y_val, 'k.', label='Numerical')
plt.plot(u_blasius, y_val, 'r-', label='Blasius')
plt.xlabel('U')
plt.ylabel('Y')
plt.title('Velocity Profile')
plt.legend()
plt.grid(True)
plt.show()