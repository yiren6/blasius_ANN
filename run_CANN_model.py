import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
# import CANN, extract_vtu_data
from CANN import CANN, BoundaryLayerDataset, custom_collate_fn, pad_sequence
from extract_vtu_data import extract_vtu_data
from eval_velocity import return_velocity
from extract_comp_info import extract_farfield_conditions
import torch
import datetime

"""
# 20240313-003320
num_cv_folds = 9
epochs = 5000
learning_rate = 10e-1
step_size = 500
gamma = 0.9
prune_iter = 500
to_prune = True
alpha=20e-3
beta = 1
prune_threshold_min=1e-3
prune_threshold_max=1e2

self.network = nn.Sequential(
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(16*4, 32), # Adjusted to take the 36 custom outputs as input
            nn.Tanh(),
            nn.Linear(32, 5),   # Outputting the 5 coefficients a1 to a5
            #nn.Linear(9, 5)     # Outputting the 5 coefficients a1 to a5
        ).to(device)

COMMENTS: weights are quite sparse, mainly e-x weights, loss is ~0.34
"""

# define the path to the vtu file
current_path = os.getcwd()
incomp_path = "Inc_Laminar_Flat_Plate/flow.vtu"
comp_folder = "all_comp"
comp_path = glob.glob(os.path.join(comp_folder, "*.vtu"))

# extract farfield conditions for compressible cases 
comp_output_folder = "all_comp/output/"
comp_farfield_conditions = extract_farfield_conditions(comp_output_folder, comp_path)

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
DOMAIN_LENGTH = 0.06096 + 0.3048

# mesh constants 
WIDTH = 65      # hard coded from mesh topology
HEIGHT = 65    # hard coded from mesh topology

# define index of x location for extraction 
x_index = list(range(30,64))

# extract the data from the vtu file
all_incomp_data = list()
all_comp_data = list()

incomp_data= extract_vtu_data(incomp_path)
max_eta_1 = 0
max_eta_2 = 0
# pack extracted data into dict for CANN 
for idx, val in enumerate(x_index):
    u_incomp, eta_incomp, re_x_incomp = return_velocity(WIDTH, HEIGHT, U_INF_INCOMP, MU_CONSTANT, RHO, incomp_data, val)
    dp_dx = 1e-3
    incomp_cur_data = (re_x_incomp, dp_dx, dp_dx, PR_LAM, eta_incomp, u_incomp)
    all_incomp_data.append(incomp_cur_data)
    
    
    cur_eta_max = max(eta_incomp)
    
    if cur_eta_max > max_eta_1:
        max_eta_1 = cur_eta_max


# loop over the compressible case 
for idx, cur_path in enumerate(comp_path):
    # get farfield conditions from comp_farfield_conditions
    ff = comp_farfield_conditions[idx]
    u_inf = ff["Velocity-X"]
    rho = ff["Density"]
    mach = ff["Mach"]
    back_pres = ff["BackPres"]
    static_pres = ff["Static Pressure"]
    total_energy = ff["Total Energy"]
    dp_dx = abs((back_pres - static_pres)/DOMAIN_LENGTH)

    comp_data = extract_vtu_data(cur_path)

    for idx, val in enumerate(x_index):
        u_comp, eta_comp, re_x_comp = return_velocity(WIDTH, HEIGHT, u_inf, MU_CONSTANT, rho, comp_data, val)
        cur_eta_max = max(eta_comp)
        if cur_eta_max > max_eta_2:
            max_eta_2 = cur_eta_max
        comp_cur_data = (re_x_comp, dp_dx, mach, PR_LAM, eta_comp, u_comp)
        all_comp_data.append(comp_cur_data)

# merge two database
input_data = all_incomp_data + all_comp_data
input_data = all_comp_data
dataset = BoundaryLayerDataset(input_data)

max_eta = max(max_eta_1, max_eta_2)
print(f"maximum eta is {max_eta}*20, make sure smaller than 20")
    

############################################################################################################
###  Train the model
############################################################################################################

# define training params 
num_cv_folds = 9
epochs = 5000
learning_rate = 10e-1
step_size = 500
gamma = 0.9
prune_iter = 500
to_prune = True

# define model parameters 
alpha=20e-3
beta = 1
prune_threshold_min=1e-3
prune_threshold_max=1e2


model = CANN(alpha=alpha, beta=beta, prune_threshold_min=prune_threshold_min, prune_threshold_max=prune_threshold_max)
# train the model with cross validation (k-fold cross validation
los_history, train_ave_loss, val_ave_loss = model.train_with_cross_validation(dataset, num_folds=num_cv_folds, \
                                                                               epochs=epochs, learning_rate=learning_rate, \
                                                                                step_size=step_size, gamma=gamma, \
                                                                                to_prune=to_prune, prune_iter=prune_iter)

trained_weights, trained_biases = model.get_weights()
print('Trained weights:')
print(trained_weights)
print('Trained biases:')
print(trained_biases)
# save the model
cur_date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f"trained_model_{cur_date_time}.pth")
# save the loss history
np.save(f"loss_history_{cur_date_time}.npy", los_history)
np.save(f"train_ave_loss_{cur_date_time}.npy", train_ave_loss)
np.save(f"val_ave_loss_{cur_date_time}.npy", val_ave_loss)
# save the weights and biases
np.save(f"trained_weights_{cur_date_time}.npy", trained_weights)
np.save(f"trained_biases_{cur_date_time}.npy", trained_biases)
# save dataset
np.save(f"dataset.npy", dataset)

# flatting loss history into 1d array 
los_history_flatten = [item for sublist in los_history for item in sublist]

# Plotting the loss history
plt.plot(los_history_flatten)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.savefig(f"loss_history_{cur_date_time}.png")
plt.show()

# Plotting the average training and validation loss
plt.plot(train_ave_loss, label='Training Loss')
plt.plot(val_ave_loss, label='Validation Loss')
plt.title('Average Training and Validation Loss')
plt.xlabel('fold')
plt.ylabel('Average Loss')
plt.legend()
plt.savefig(f"train_val_loss_{cur_date_time}.png")
plt.show()

# randomly select some 5 data from the dataset to evaluate the model
num_data = 5
random_indices = np.random.choice(len(dataset), num_data, replace=False)
for idx in random_indices:
    re_x, dp_dx, mach, pr_lam, eta, u = dataset[idx]
    predicted_dy_deta = model.eval_prediction(re_x, dp_dx, mach, pr_lam, torch.tensor(eta, dtype=torch.float32).unsqueeze(0))
    plt.plot(u, eta, label=f"Ma={mach}", linestyle='--')
    plt.plot(predicted_dy_deta.cpu().detach().numpy().squeeze(), eta, label=f"Ma={mach}", linestyle='-')
plt.xlabel('$\eta$')
plt.ylabel('$u$')
plt.legend()
plt.savefig(f"eval_{cur_date_time}.png")
plt.show()    




eta_orig = [0, 1, 1.5, 2, 2.3, 3, 4, 5, 7, 13]  # Example eta values
eta = [x / 13 for x in eta_orig]  # Divide each element by 10

f_eta = [0, 0.2, 0.3, 0.4, 0.64, 0.8, 0.92, 0.98, 0.989, 0.99999]
predicted_dy_deta = model.eval_prediction(re_x_incomp, dp_dx, 0.001, PR_LAM, torch.tensor(eta, dtype=torch.float32).unsqueeze(0))
plt.plot(eta_orig, f_eta, label='True')
plt.plot(eta_orig, predicted_dy_deta.cpu().numpy().squeeze(), label='Predicted')
plt.xlabel('$\eta$')
plt.ylabel('$f\'(\eta)$')
plt.legend()
plt.show()