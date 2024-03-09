import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# import CANN, extract_vtu_data
from CANN import CANN, BoundaryLayerDataset, custom_collate_fn, pad_sequence
from extract_vtu_data import extract_vtu_data
from eval_velocity import return_velocity

import torch

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
x_index = list(range(30,64))

# extract the data from the vtu file
all_incomp_data = list()
all_comp_data = list()

incomp_data= extract_vtu_data(incomp_path)
comp_data = extract_vtu_data(comp_path)
max_eta = 0
# pack extracted data into dict for CANN 
for idx, val in enumerate(x_index):
    u_incomp, eta_incomp, re_x_incomp = return_velocity(WIDTH, HEIGHT, U_INF_INCOMP, MU_CONSTANT, RHO, incomp_data, val)
    dp_dx = 1e-3
    incomp_cur_data = (re_x_incomp, dp_dx, 1e-3, PR_LAM, eta_incomp, u_incomp)
    all_incomp_data.append(incomp_cur_data)
    u_comp, eta_comp, re_x_comp = return_velocity(WIDTH, HEIGHT, U_INF_COMP, MU_CONSTANT, RHO, comp_data, val)
    cur_eta_max1 = max(eta_comp)
    cur_eta_max2 = max(eta_incomp)
    cur_eta_max = max(cur_eta_max1, cur_eta_max2)
    if cur_eta_max > max_eta:
        max_eta = cur_eta_max
    comp_cur_data = (re_x_comp, dp_dx, MA_COMP, PR_LAM, eta_comp, u_comp)
    all_comp_data.append(comp_cur_data)

# merge two database
input_data = all_incomp_data + all_comp_data
input_data = all_comp_data
dataset = BoundaryLayerDataset(input_data)

print(f"maximum eta is {max_eta}")
    

############################################################################################################
###  Train the model
############################################################################################################

model = CANN()
los_history = model.train_model(dataset)

trained_weights, trained_biases = model.get_weights()
print('Trained weights:')
print(trained_weights)
print('Trained biases:')
print(trained_biases)

# Plotting the loss history
plt.plot(los_history)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()

eta_orig = [0, 1, 1.5, 2, 2.3, 3, 4, 5, 7, 13]  # Example eta values
eta = [x / 500 for x in eta_orig]  # Divide each element by 10

f_eta = [0, 0.2, 0.3, 0.4, 0.64, 0.8, 0.92, 0.98, 0.989, 0.99999]
predicted_dy_deta = model.eval_prediction(re_x_incomp, dp_dx, 0.001, PR_LAM, torch.tensor(eta, dtype=torch.float32).unsqueeze(0))
plt.plot(eta_orig, f_eta, label='True')
plt.plot(eta_orig, predicted_dy_deta.cpu().numpy().squeeze(), label='Predicted')
plt.xlabel('$\eta$')
plt.ylabel('$f\'(\eta)$')
plt.legend()
plt.show()