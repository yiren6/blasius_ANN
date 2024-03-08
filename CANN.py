import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt


# Assuming your data is structured as a list of tuples [(Re_x, dp_dx, Ma, Pr, eta_values, dy_deta_values), ...]
class BoundaryLayerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Re_x, dp_dx, Ma, Pr, eta_values, dy_deta_values = self.data[idx]
        physical_params = torch.tensor([Re_x, dp_dx, Ma, Pr], dtype=torch.float32)
        eta = torch.tensor(eta_values, dtype=torch.float32)
        dy_deta = torch.tensor(dy_deta_values, dtype=torch.float32)
        return physical_params, eta, dy_deta

class CANN(nn.Module):
    def __init__(self):
        super(CANN, self).__init__()
        # Define the subsequent parts of the network
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(36, 12), # Adjusted to take the 36 custom outputs as input
            nn.ReLU(),
            nn.Linear(12, 6)   # Outputting the 5 coefficients a1 to a5
        )

    def evaluate(self, x):
        return self.network(x)    

    def power_series_transformation(self, x):
        # x is the input tensor with shape [batch_size, 4] ([Re_x, dp_dx, Ma, Pr] for each sample)
        powers = torch.tensor([0, 1/2, 2, 1/3, 3, 1/4, 4, 1/5, 5], dtype=torch.float32, device=x.device)
        transformed = torch.cat([x[:, i:i+1].pow(powers) for i in range(4)], dim=1)
        return transformed
    
    def forward(self, physical_params, eta):
        transformed_params = self.power_series_transformation(physical_params)
        mean = transformed_params.mean()
        std = transformed_params.std()

        # Normalize the tensor
        norm_params = (transformed_params - mean) / std
        coefficients = self.network(norm_params)
        eta = eta.t()
        powers = torch.tensor([1, 2, 5, 8, 11, 14], dtype=torch.float32, device=eta.device)
        dy_deta = (coefficients * powers) @ ((eta.pow(powers - 1)).t())
        return dy_deta

    def train_model(self, dataset, epochs=1000, learning_rate=1e-1, step_size=100, gamma=0.5):
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        loss_fn = nn.L1Loss()
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            for physical_params, eta, true_dy_deta in dataloader:
                optimizer.zero_grad()
                predicted_dy_deta = self(physical_params, eta)
                loss = loss_fn(predicted_dy_deta, true_dy_deta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()  # Decay the learning rate
            average_loss = total_loss / len(dataloader)
            loss_history.append(average_loss) 
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        return loss_history
    
    def eval_prediction(self, Re_x, dp_dx, Ma, Pr, eta):
        # Assuming Re_x, dp_dx, Ma, Pr, and eta are already tensors. If not, you should convert them.
        # true_dy_deta should also be a tensor of true values.
        
        physical_params = torch.stack((Re_x, dp_dx, Ma, Pr), dim=-1)
        
        # Disable gradient computation for prediction
        with torch.no_grad():
            predicted_dy_deta = self(physical_params, eta)
        return predicted_dy_deta    
# Example usage
# data = [...]  # Your dataset here
# dataset = BoundaryLayerDataset(data)
# model = CANN()
# model.train_model(dataset)


# Example input tensors
re_x = torch.tensor([300.0])  # Replace with actual value
dp_dx = torch.tensor([1.0])  # Replace with actual value
Ma = torch.tensor([0.1])  # Replace with actual value
Pr = torch.tensor([0.71])  # Replace with actual value

eta_orig = [0, 1, 1.5, 2, 2.3, 3, 4, 5, 7]  # Example eta values
eta = [x / 10 for x in eta_orig]  # Divide each element by 10

f_eta = [0, 0.2, 0.3, 0.4, 0.64, 0.8, 0.92, 0.98, 0.999]

input_data = [(re_x, dp_dx, Ma, Pr, eta, f_eta)]
dataset = BoundaryLayerDataset(input_data)
model = CANN()
los_history = model.train_model(dataset)

# Plotting the loss history
plt.plot(los_history)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()

# plot the prediction vs true 
predicted_dy_deta = model.eval_prediction(re_x, dp_dx, Ma, Pr, torch.tensor(eta, dtype=torch.float32).unsqueeze(0))
plt.plot(eta_orig, f_eta, label='True')
plt.plot(eta_orig, predicted_dy_deta.cpu().numpy().squeeze(), label='Predicted')
plt.xlabel('$\eta$')
plt.ylabel('$f\'(\eta)$')
plt.legend()
plt.show()
