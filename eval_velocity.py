import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from extract_vtu_data import extract_vtu_data
from scipy.integrate import solve_ivp

"""
File to evaluate the velocity field of a flow over a flat plate

"""

def blasius_derivative(eta):
    return 0.332 * np.sqrt(eta) - 0.4167 * eta + 0.0833 / (1 + eta)**(1.5)

def eta(y, x, u_inf, nu):
    return y * np.sqrt(u_inf/nu/x)

def Re(x, u_inf, nu):
    return u_inf * x / nu

def blasius_expansion(eta):
    k = 1/2
    sigma = 0.33206
    f = sigma/np.math.factorial(2) * eta**2 - k*sigma**2/np.math.factorial(5) * eta**5 + \
        11*(k**2)*sigma**3/np.math.factorial(8) * eta**8 - \
        375*(k**3)*sigma**4/np.math.factorial(11) * eta**11
    # result is  u = df/deta
    u = np.diff(f) / np.diff(eta)
    # append the last value to make the length same as eta
    u = np.append(u, 0)
    return f


def return_velocity(WIDTH, HEIGHT, u_inf, mu, rho, vtu_data, x_index):
    """
    Function to extract the velocity field at a specific x location from the vtu data

    Args:
    u_inf (float): free stream velocity
    mu (float): dynamic viscosity
    rho (float): density
    vtu_data (dict): extracted vtu data
    x_index (int): index of x location for extraction

    Returns:
    u (np.array): velocity field at the x_index
    y (np.array): y coordinate of the velocity field
    x (np.array): x coordinate of the velocity field
    """

    # extract data
    coordinate = vtu_data['Coordinates']
    velocity = vtu_data['Velocity']

    nu = mu / rho
    # interpolate the node index of given x_index
    column_start = x_index * HEIGHT
    column_end = column_start + HEIGHT

    # extract the velocity field at the x_index
    # for Blasius boundary layer we are only interested in u
    u = velocity[column_start:column_end, 0] / u_inf
    v = velocity[column_start:column_end, 1]
    w = velocity[column_start:column_end, 2]

    y = coordinate[column_start:column_end, 1]
    x = coordinate[column_start:column_end, 0]

    my_eta = eta(y, x[1], u_inf, nu)
    
    # find portion that eta is less than 20
    mask = my_eta < 20
    u = u[mask]
    my_eta = my_eta[mask]
    
    my_eta = my_eta / 20
    # divide by 10 to convert to scale 
    #my_eta = my_eta / 500

    re_x = Re(x[1], u_inf, nu)

    return u, my_eta, re_x

if __name__ == "__main__":
    ############################################################################################################
    ###  Import file and extract data
    ############################################################################################################
    # define boundary conditions 
    U_INF = 69.16444
    MU_CONSTANT= 1.83463e-05
    RHO = 1.13235
    NU = MU_CONSTANT / RHO

    # define files 
    filename = "Inc_Laminar_Flat_Plate/flow_comp.vtu"
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
    #heat_capacity = data['Heat_Capacity']
    #thermal_conductivity = data['Thermal_Conductivity']
    skin_friction_coefficient = data['Skin_Friction_Coefficient']
    heat_flux = data['Heat_Flux']
    y_plus = data['Y_Plus']


    # Blasius function derivative

    ############################################################################################################
    ###  Evaluate the velocity field at specific x location 
    ############################################################################################################

    # define the index of x loc for the velocity field extraction 

    x_index =  64
    if x_index > WIDTH:
        print(f"Index out of range, max index is {WIDTH}")
        sys.exit(1)


    # interpolate the node index of given x_index
    column_start = x_index * HEIGHT
    column_end = column_start + HEIGHT

    # extract the velocity field at the x_index
    # for Blasius boundary layer we are only interested in u
    u = velocity[column_start:column_end, 0]/U_INF
    v = velocity[column_start:column_end, 1]
    w = velocity[column_start:column_end, 2]

    print(f"Extracting data at x = {coordinate[column_start, 0]}")
    y_val = coordinate[column_start:column_end, 1]
    x_val = coordinate[column_start:column_end, 0]
    # plot the coordinate
    plt.figure()
    plt.plot(x_val, eta(y_val,x_val[1], U_INF, NU), 'k.')
    plt.xlabel('X')
    plt.ylabel('$\eta$')
    plt.title('Flat Plate Mesh Verify')
    plt.show()

    # plot the velocity field
    plt.figure()
    plt.plot(u, eta(y_val,x_val[1], U_INF, NU), 'k.')
    print(Re(x_val[1], U_INF, NU))
    x_index =  25
    if x_index > WIDTH:
        print(f"Index out of range, max index is {WIDTH}")
        sys.exit(1)


    # interpolate the node index of given x_index
    column_start = x_index * HEIGHT
    column_end = column_start + HEIGHT

    # extract the velocity field at the x_index
    # for Blasius boundary layer we are only interested in u
    u = velocity[column_start:column_end, 0]/U_INF
    v = velocity[column_start:column_end, 1]
    w = velocity[column_start:column_end, 2]

    print(f"Extracting data at x = {coordinate[column_start, 0]}")
    y_val = coordinate[column_start:column_end, 1]
    x_val = coordinate[column_start:column_end, 0]
    plt.plot(u, eta(y_val,x_val[1], U_INF, NU), 'r.')

    # plot analytical solution 
    eta_val = np.linspace(0, 9, 100)
    u = blasius_expansion(eta_val)
    plt.plot(u, eta_val, 'b-')
    plt.xlabel('U')
    plt.ylabel('$\eta$')
    plt.ylim(0, 10)
    plt.xlim(0, 1)
    plt.title('Velocity')
    plt.show()

    print(eta(y_val,x_val[1], U_INF, NU))
    print(u)
    print(Re(x_val[1], U_INF, NU))