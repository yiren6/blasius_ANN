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
    def __init__(self, alpha=100e-3, beta = 10, prune_threshold_min=0.01, prune_threshold_max=1e4):

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
            nn.ReLU(),
        #    nn.Tanh(),
            nn.Linear(16*4, 32), # Adjusted to take the 36 custom outputs as input
            nn.Linear(32, 5)   # Outputting the 5 coefficients a1 to a5
        #    nn.Linear(9, 5)     # Outputting the 5 coefficients a1 to a5
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

    def train_model(self, dataloader, epochs=100, learning_rate=10e-1, step_size=50, gamma=0.5, prune_iter = 50, to_prune=True):
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

        # randomly shuffle dataset 
        np.random.seed(3407)
        np.random.shuffle(dataset.data)

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
            if fold < 3 and to_prune == True:
                cur_to_prune = False
            elif fold > 3 and to_prune == True:
                cur_to_prune = True
            else:
                cur_to_prune = False
            # Train the model
            fold_epoches = int(np.ceil(epochs/num_folds))
            fold_learning_rate = learning_rate * (gamma ** ((fold_epoches*fold)/step_size)) # based on total epoches of all folds
            print(f"Learning rate: {fold_learning_rate}")
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
    re_x = 30000
    dp_dx = 0.001
    Ma = 0.1
    Pr = 0.71

    eta_orig = [0,0.18861613758858,0.377232275177161,0.565848412765741,0.754464550354321,0.943080687942901,1.13169682553148,1.32031296312006,1.50892910070864,1.69754523829722,1.8861613758858,2.07477751347438,2.26339365106296,2.45200978865154,2.64062592624012,2.8292420638287,3.01785820141728,3.20647433900586,3.39509047659444,3.58370661418303,3.77232275177161,3.96093888936019,4.14955502694877,4.33817116453735,4.52678730212593,4.71540343971451,4.90401957730309,5.09263571489167,5.28125185248025,5.46986799006883,5.65848412765741,5.84710026524599,6.03571640283457,6.22433254042315,6.41294867801173,6.60156481560031,6.79018095318889,6.97879709077747,7.16741322836605,7.35602936595463,7.54464550354321,7.73326164113179,7.92187777872037,8.11049391630895,8.29911005389753,8.48772619148611,8.67634232907469,8.86495846666327,9.05357460425185,9.24219074184043,9.43080687942901,9.61942301701759,9.80803915460617,9.99665529219475,10.1852714297833,10.3738875673719,10.5625037049605,10.7511198425491,10.9397359801377,11.1283521177262,11.3169682553148,11.5055843929034,11.694200530492,11.8828166680806,12.0714328056691,12.2600489432577,12.4486650808463,12.6372812184349,12.8258973560235,13.014513493612,13.2031296312006,13.3917457687892,13.5803619063778,13.7689780439664,13.9575941815549,14.1462103191435,14.3348264567321,14.5234425943207,14.7120587319093,14.9006748694978,15.0892910070864,15.277907144675,15.4665232822636,15.6551394198522,15.8437555574407,16.0323716950293,16.2209878326179,16.4096039702065,16.5982201077951,16.7868362453836,16.9754523829722,17.1640685205608,17.3526846581494,17.541300795738,17.7299169333265,17.9185330709151,18.1071492085037,18.2957653460923,18.4843814836809,18.6729976212695,18.861613758858]
    eta_orig = eta_orig[0:45]
      # Example eta values
    eta = [x / 9.5 for x in eta_orig]  # Divide each element by 10

    f_eta = [0,0.061717,0.12343,0.18505,0.24642,0.3073,0.36743,0.42646,0.48402,0.53957,0.59282,0.64337,0.69085,0.73509,0.77564,0.81152,0.84392,0.87351,0.89702,0.91842,0.93589,0.95037,0.9623,0.97165,0.97899,0.985,0.98897,0.99289,0.99472,0.99655,0.99788,0.99857,0.99926,0.99966,0.99985,1.0001,1.0002,1.0002,1.0002,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003,1.0003]
    f_eta = f_eta[0:45]

    input_data = [(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta),(re_x, dp_dx, Ma, Pr, eta, f_eta)]
    dataset = BoundaryLayerDataset(input_data)
    model = CANN()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    los_history, average_loss = model.train_model(train_loader)
    plt.plot(los_history)
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()
    print(average_loss)
    #los_history, train_average_losses, val_average_losses = model.train_with_cross_validation(dataset, num_folds=2, epochs=1000, learning_rate=10e-1, step_size=50, gamma=0.4, prune_iter=2, to_prune=False)
    #print(train_average_losses)
    #print(val_average_losses)
    # Plotting the loss history
    #for fold in range(2):
    #    plt.plot(los_history[fold])
    #plt.title('Loss History')
    #plt.xlabel('Epoch')
    #plt.ylabel('Average Loss')
    #plt.show()

    trained_weights, trained_biases = model.get_weights()
    print('Trained weights:')
    print(trained_weights)
    print('Trained biases:')
    print(trained_biases)

    

    # plot the prediction vs true 
    predicted_dy_deta = model.eval_prediction(re_x, dp_dx, Ma, Pr, torch.tensor(eta, dtype=torch.float32).unsqueeze(0))
    plt.plot(eta_orig, f_eta, label='True')
    plt.plot(eta_orig, predicted_dy_deta.cpu().numpy().squeeze(), label='Predicted')
    plt.xlabel('$\eta$')
    plt.ylabel('$f\'(\eta)$')
    plt.legend()
    plt.show()

    # plot the trained weights
    for i, layer_weights in enumerate(trained_weights):
        plt.figure()
        plt.hist(layer_weights.flatten(), bins=50)
        plt.title(f'Layer {i+1} Weights')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()

    for i, layer_weights in enumerate(trained_weights):
        weights_map = np.vstack(layer_weights)

        plt.figure()
        plt.imshow(weights_map, cmap='viridis')
        plt.title(f'Layer {i+1} Weights')
        plt.xlabel('Input Neuron')
        plt.ylabel('Output Neuron')
        plt.colorbar()
        plt.show()
    weights_map_1 = np.vstack(trained_weights[0])
    weights_map_2 = np.vstack(trained_weights[1])
    effective_weights = weights_map_2 @ weights_map_1

    plt.figure()
    plt.imshow(effective_weights, cmap='viridis')
    plt.title('Effective Weights')
    plt.xlabel('Input Neuron')
    plt.ylabel('Output Neuron')
    plt.colorbar()
    plt.show()

