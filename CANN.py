import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CANN(nn.Module):
    def __init__(self, output_size, re_x, dp_dx, eta_array, learning_rate, loss_type):
        super(CANN, self).__init__()
        # Define the input size based on feature the number of features we will create from the inputs
        self.input_size = 2  #  re_x and dp_dx
        self.hidden = nn.Linear(self.input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.re_x_powers = torch.tensor([-2, -1, -2/3, -1/2, -1/5, 0, 1/5, 1/2, 2/3, 1, 2])
        self.dp_dx_powers = torch.tensor([-2, -1, -2/3, -1/2, -1/5, 0, 1/5, 1/2, 2/3, 1, 2])
        self.hidden_layer_size =  self.re_x_powers.size()[0] + self.dp_dx_powers.size()[0]    #size of sel.re_x_powers
        self.hidden = nn.Linear(self.input_size, self.hidden_layer_size)
        self.eta_powers = torch.tensor([-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12])
        self.eta_array = eta_array 
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.rex = re_x
        self.dp_dx = dp_dx
        # define training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
        self.f = None
        if self.loss_type == 'L1':
            self.loss_function = nn.L1Loss()
        elif self.loss_type == 'RES':
            self.loss_function = self.gradient_loss(self.f, self.eta_array)
        else:
            print("Invalid loss type, please choose from 'L1' or 'RES', using MSE loss as default")
            self.loss_function = nn.MSELoss()

    
    def forward(self, re_x, dp_dx, eta):
        # Calculate the powers for Re_x and dp_dx
        re_x_powers = torch.stack([re_x**i for i in self.re_x_powers]).t()  # .t() for transpose
        dp_dx_powers = torch.stack([dp_dx**i for i in self.dp_dx_powers]).t()

        # Combine the two input vectors
        combined_input = torch.cat((re_x_powers, dp_dx_powers), dim=1)

        # Pass the input through the hidden layer with an activation function, e.g., ReLU
        hidden_output = F.relu(self.hidden(combined_input))

        # Pass the output of the hidden layer to the output layer
        output = self.output(hidden_output)
        
        # Reshape eta to be able to perform broadcasting in the multiplication
        eta = torch.tensor(eta).unsqueeze(1)  # Adding an extra dimension for broadcasting

        # Assume the output represents coefficients for the polynomial terms of eta
        # Now multiply the outputs by the eta powers
        eta_powers = torch.stack([eta**i for i in self.eta_powers], dim=2)  # Creating a tensor of eta powers
        final_output = (output.unsqueeze(1) * eta_powers).sum(dim=2)  # Multiply and sum across the polynomial terms
        
        return final_output
    
    def train_model(self, epochs, true_velocity_profiles, learning_rate=0.001, weight_decay=1e-5):
        # Convert the input data to tensors
        re_x_data = torch.tensor(self.rex, dtype=torch.float32)
        dp_dx_data = torch.tensor(self.dp_dx, dtype=torch.float32)
        eta_data = torch.tensor(self.eta_array, dtype=torch.float32)
        
        for epoch in range(epochs):
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_velocity = self.forward(re_x_data, dp_dx_data, eta_data)
            
            # Compute loss
            loss = self.loss_function(predicted_velocity, true_velocity_profiles)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Optional: Print loss
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: train loss: {loss.item()}')


    def gradient_loss(self, f, eta):
        # Make sure that `f` requires gradient
        f.requires_grad_(True)
        
        # We assume that `f` is the result of CANN network and `eta` is the corresponding input tensor
        # Compute the second derivative f'' with respect to eta
        f_prime = torch.autograd.grad(f, eta, grad_outputs=torch.ones(f.shape), create_graph=True)[0]
        f_double_prime = torch.autograd.grad(f_prime, eta, grad_outputs=torch.ones(f_prime.shape), create_graph=True)[0]
        
        # Compute the third derivative f''' with respect to eta
        f_triple_prime = torch.autograd.grad(f_double_prime, eta, grad_outputs=torch.ones(f_double_prime.shape), create_graph=True)[0]
        
        # Now compute the custom loss term: 2*f''' + f*f''
        loss_term = 2 * f_triple_prime + f * f_double_prime
        
        # You can use a standard loss function, like MSE, to compute the loss between the loss_term and zeros
        # This effectively measures the magnitude of the loss_term, which we want to minimize
        loss = torch.mean(loss_term ** 2)
        
        return loss
    


# Number of neurons in the hidden layer and output size
hidden_size = 50  # Adjust as necessary
output_size = 13  # This should match the highest power of eta you wish to consider

# Instantiate the network
cann_network = CANN(hidden_size, output_size)

# Example input tensors
re_x = torch.tensor([1.0])  # Replace with actual value
dp_dx = torch.tensor([-0.1])  # Replace with actual value
eta = [0, 1, 1.5, 2, 2.3, 3, 4, 5]  # Example eta values

# Get the predicted boundary layer velocity profile
predicted_velocity = cann_network(re_x, dp_dx, eta)
print(predicted_velocity)
