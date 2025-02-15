import torch
import torch.nn as nn
import numpy as np

class MulticlassNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, criterion):
        super(MulticlassNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.criterion = criterion
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        
        self.classes_ = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    def fit(self, X_train_tensor, y_train_tensor, num_epochs, validation_data: tuple = None, optimizer = None):
        """
        Trains the model on the given training data for the specified number of epochs.

        Args:
            X_train_tensor (torch.Tensor): The input training data tensor.
            y_train_tensor (torch.Tensor): The target training data tensor.
            num_epochs (int): The number of epochs to train the model.
            validation_data (tuple, optional): A tuple containing the validation data tensors (X_test_tensor, y_test_tensor).
                                               Defaults to None.

        Returns:
            None
        """
        self.classes_ = np.unique(y_train_tensor)
        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.forward(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if validation_data is not None:
                X_test_tensor = validation_data[0]
                y_test_tensor = validation_data[1]
                # Test Set
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    outputs = self.forward(X_test_tensor)
                    test_loss = self.criterion(outputs, y_test_tensor)
                
                # Print the loss for every 10 epochs
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

            # Print the loss for every 10 epochs
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        outputs = self.forward(X)
        probas = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probas, 1)
        return predicted.numpy()
    
    def predict_proba(self, X):
        outputs = self.forward(X)
        return torch.softmax(outputs, dim=1).detach().numpy()

