import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# import CANN, extract_vtu_data
from CANN import CANN
from extract_vtu_data import extract_vtu_data


# define the path to the vtu file
incomp_path = "Inc_Laminar_Flat_Plate/flow.vtu"
comp_path = "Inc_Laminar_Flat_Plate/flow_comp.vtu"

# define constants 
U_INF_INCOMP = 1
U_INF_COMP = 69.1687
MU_CONSTANT= 1.83463e-05
RHO = 1.13235
NU = MU_CONSTANT / RHO
PR_LAM = 0.72
PR_TURB = 0.9
GAMMA = 1.4
R = 287.05
T_INF = 297.62
MA_COMP = 0.2
MA_INCOMP = 0.195324

# mesh constants 
WIDTH = 65      # hard coded from mesh topology
HEIGHT = 65    # hard coded from mesh topology

# define index of x location for extraction 
x_index = list(range(26,64))

# extract the data from the vtu file
in_comp_data = []
comp_data = []

incomp_data= extract_vtu_data(incomp_path)
comp_data = extract_vtu_data(comp_path)

# pack extracted data into dict for CANN 
for idx, val in enumerate(x_index):
