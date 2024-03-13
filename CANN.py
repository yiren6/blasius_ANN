import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.prune as prune
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

def pad_sequence(sequences):
    """
    Pad sequences to the same length with zeros.
    """
    max_length = max([s.size(0) for s in sequences])
    padded_sequences = torch.zeros(len(sequences), max_length)
    for i, sequence in enumerate(sequences):
        padded_sequences[i, :sequence.size(0)] = sequence
    return padded_sequences

def custom_collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    """
    # Unzip the batch
    physical_params, eta, dy_deta = zip(*batch)
    # Pad the eta and dy_deta sequences
    padded_eta = pad_sequence(eta)
    padded_dy_deta = pad_sequence(dy_deta)
    # Stack the physical_params tensors
    physical_params_stacked = torch.stack(physical_params)
    return physical_params_stacked, padded_eta, padded_dy_deta

class CANN(nn.Module):
    def __init__(self, alpha=20e-3, beta = 1, prune_threshold_min=1e-3, prune_threshold_max=1e2):

        super(CANN, self).__init__()
        self.alpha = alpha  # L1 regularization strength
        self.beta = beta # L2 loss strength
        self.prune_threshold = prune_threshold_min  # Threshold for pruning weights
        self.prune_threshold_max = prune_threshold_max
        # access the current CUDA enviroment 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # Define the subsequent parts of the network
        self.network = nn.Sequential(
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(16*4, 32), # Adjusted to take the 36 custom outputs as input
            nn.Tanh(),
            nn.Linear(32, 9),   # Outputting the 5 coefficients a1 to a5
            nn.Linear(9, 5)     # Outputting the 5 coefficients a1 to a5
        ).to(device)

    def evaluate(self, x):
        return self.network(x)    
    @staticmethod
    def create_threshold_mask(weights, threshold_min, threshold_max):
        """
        Create a mask for weights that are above a certain absolute value threshold.

        Parameters:
        weights (Tensor): The weight tensor from a neural network layer.

        Returns:
        Tensor: A boolean mask where True indicates the weight should be kept, and False indicates it should be pruned.
        """
        
        mask = torch.logical_and(torch.abs(weights) > threshold_min, torch.abs(weights) < threshold_max)
        return mask

    
    def apply_threshold_pruning(self):
        """
        apply threshold pruning to the weights and biases of the network
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                mask_weights = self.create_threshold_mask(weights, self.prune_threshold, self.prune_threshold_max)
                bias = module.bias.data
                mask_bias = self.create_threshold_mask(bias, self.prune_threshold, self.prune_threshold_max)
                prune.custom_from_mask(module, name='weight', mask=mask_weights)
                prune.custom_from_mask(module, name='bias', mask=mask_bias)
                with torch.no_grad():
                    module.weight.data[torch.abs(module.weight) < self.prune_threshold] = 0
                    module.bias.data[torch.abs(module.bias) < self.prune_threshold] = 0


    def power_series_transformation(self, x):
        """
        Apply a power series transformation to the input tensor.
        """
        # x is the input tensor with shape [batch_size, 4] ([Re_x, dp_dx, Ma, Pr] for each sample)
        powers = torch.tensor([-5, -4, -3, -2, -3/2, -1, -1/2, 0, 1/2, 2, 1/3, 3, 1/4, 4, 1/5, 5], dtype=torch.float32, device=x.device)
        transformed = torch.cat([x[:, i:i+1].pow(powers) for i in range(4)], dim=1)
        return transformed
    
    def forward(self, physical_params, eta):
        """
        Forward pass of the network.
        """
        transformed_params = self.power_series_transformation(physical_params).to(device=self.device)
        mean = transformed_params.mean()
        std = transformed_params.std()

        # Normalize the tensor
        norm_params = (transformed_params - mean) / std
        coefficients = self.network(norm_params)
        powers = torch.tensor([2, 5, 8, 11, 14], dtype=torch.float32, device=self.device)
        eta_samples, eta_length = eta.size()
        dy_deta = torch.zeros(eta_samples, eta_length, device=self.device)
        for i in range(eta_samples):
            cur_eta = eta[i, :].unsqueeze(0)
            cur_eta = cur_eta.t()
            cur_coefficients = coefficients[i, :].unsqueeze(0)
            cur_dy_deta = (cur_coefficients * powers) @ ((cur_eta.pow(powers - 1)).t())
            dy_deta[i, :] = cur_dy_deta
        # append all dy_deta to tensor 
        
        return dy_deta

    def train_model(self, dataloader, epochs=10000, learning_rate=10e-1, step_size=500, gamma=0.4, prune_iter = 2000, to_prune=True):
        """
        Train the model on the given dataset as inner loop of k-fold cross validation.
        """

        print(f"Training on {self.device}")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        loss_fn1 = nn.L1Loss()
        loss_fn2 = nn.MSELoss()
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (physical_params, eta, true_dy_deta) in enumerate(iter(dataloader)):

                physical_params, eta, true_dy_deta = physical_params.to(self.device), eta.to(self.device), true_dy_deta.to(self.device)
                optimizer.zero_grad()
                predicted_dy_deta = self(physical_params, eta)
                loss = loss_fn1(predicted_dy_deta, true_dy_deta) + self.beta * loss_fn2(predicted_dy_deta, true_dy_deta)
                l1_reg = sum(param.abs().sum() for param in self.parameters())
                reg_loss = loss + self.alpha * l1_reg
                reg_loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()  # Decay the learning rate

            # check pruning req 
            if epoch % prune_iter == 0 and to_prune:
                self.apply_threshold_pruning()
            average_loss = total_loss / len(dataloader)
            loss_history.append(average_loss) 
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        return loss_history, average_loss
    
    def train_with_cross_validation(self, dataset, num_folds=5, epochs=10000, learning_rate=10e-1, step_size=500, gamma=0.4, prune_iter=2000, to_prune=True):
        """
        Train the model using k-fold cross validation.
        """
        print(f"Running on {self.device}")
        fold_losses = []
        train_average_losses = []
        val_average_losses = []
        loss_fn1 = nn.L1Loss()
        loss_fn2 = nn.MSELoss()
        # Split dataset into folds
        fold_size = len(dataset) // num_folds
        folds = [torch.utils.data.Subset(dataset, range(i * fold_size, (i + 1) * fold_size)) for i in range(num_folds)]

        for fold in range(num_folds):
            print(f"Fold {fold+1}/{num_folds}")

            # Prepare data loaders for train and validation sets
            train_indices = [i for j, fold_data in enumerate(folds) if j != fold for i in fold_data.indices]
            val_indices = folds[fold].indices
            train_set = torch.utils.data.Subset(dataset, train_indices)
            val_set = torch.utils.data.Subset(dataset, val_indices)
            
            # Create DataLoader for train set and validation set 
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
            
            # first 2 folds do not prune 
            if fold < 2 and to_prune == True:
                cur_to_prune = False
            elif fold > 2 and to_prune == True:
                cur_to_prune = True
            else:
                cur_to_prune = False
            # Train the model
            fold_epoches = int(np.ceil(epochs/num_folds))
            fold_learning_rate = learning_rate * (gamma ** ((fold_epoches*fold)/step_size)) # based on total epoches of all folds
            loss_history, avg_train_loss = self.train_model(train_loader, epochs=fold_epoches, learning_rate=fold_learning_rate, step_size=step_size, gamma=gamma, prune_iter=prune_iter, to_prune=cur_to_prune)
            fold_losses.append(loss_history)

            # Validate the model
            val_losses = []
            for physical_params, eta, true_dy_deta in val_loader:
                physical_params, eta, true_dy_deta = physical_params.to(self.device), eta.to(self.device), true_dy_deta.to(self.device)
                predicted_dy_deta = self(physical_params, eta)
                loss = loss_fn1(predicted_dy_deta, true_dy_deta) + loss_fn2(predicted_dy_deta, true_dy_deta)
                val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Validation Loss: {avg_val_loss}")
            train_average_losses.append(avg_train_loss)
            val_average_losses.append(avg_val_loss)

        return fold_losses, train_average_losses, val_average_losses
    
    def get_weights(self):
        weights = []
        biases = []
        for layer in self.network:
            if hasattr(layer, 'weight'):  # Check if the layer has the 'weight' attribute
                weights.append(layer.weight.data.cpu().numpy())
                biases.append(layer.bias.data.cpu().numpy())
        return weights, biases
    
    def eval_prediction(self, Re_x, dp_dx, Ma, Pr, eta):
        # Assuming Re_x, dp_dx, Ma, Pr, and eta are already tensors. 
        # true_dy_deta should also be a tensor of true values.
        
        # making sure Re_x dp_dx Ma Pr are tensors
        Re_x = torch.tensor(Re_x, dtype=torch.float32).unsqueeze(0)
        dp_dx = torch.tensor(dp_dx, dtype=torch.float32).unsqueeze(0)
        Ma = torch.tensor(Ma, dtype=torch.float32).unsqueeze(0)
        Pr = torch.tensor(Pr, dtype=torch.float32).unsqueeze(0)

        physical_params = torch.stack((Re_x, dp_dx, Ma, Pr), dim=-1).to(self.device)
        eta = eta.to(self.device)
        # Disable gradient computation for prediction
        with torch.no_grad():
            predicted_dy_deta = self(physical_params, eta)
        return predicted_dy_deta    


if __name__ == "__main__":

    # debug script 
    re_x = torch.tensor([300.0])  
    dp_dx = torch.tensor([1.0])  
    Ma = torch.tensor([0.1])  
    Pr = torch.tensor([0.71])  

    eta_orig = [0, 1, 1.5, 2, 2.3, 3, 4, 5, 7]  # Example eta values
    eta = [x / 10 for x in eta_orig]  # Divide each element by 10

    f_eta = [0, 0.2, 0.3, 0.4, 0.64, 0.8, 0.92, 0.98, 0.999]

    input_data = [(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta)]
    dataset = BoundaryLayerDataset(input_data)
    model = CANN()
    #los_history = model.train_model(dataset)
    los_history, train_average_losses, val_average_losses = model.train_with_cross_validation(dataset, num_folds=2, epochs=1000, learning_rate=10e-1, step_size=50, gamma=0.4, prune_iter=2, to_prune=False)
    print(train_average_losses)
    print(val_average_losses)

    trained_weights, trained_biases = model.get_weights()
    print('Trained weights:')
    print(trained_weights)
    print('Trained biases:')
    print(trained_biases)

    # Plotting the loss history
    for fold in range(2):
        plt.plot(los_history[fold])
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
