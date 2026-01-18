import torch.nn as nn
import torch
import numpy as np

class FlexibleMLP(nn.Module):
    def __init__(self, n_hidden_layers, neurons_per_layer, activation="ReLU"):
        super().__init__()

        # Map string to activation function
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU()
        }
        act_fn = activations.get(activation, nn.ReLU())

        # Build layers dynamically
        layers = []

        # Input layer -> first hidden layer
        layers.append(nn.Linear(2, neurons_per_layer))
        layers.append(act_fn)

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(act_fn)

        # Output layer (binary classification)
        layers.append(nn.Linear(neurons_per_layer, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_neuron_activations(self, X):
        """Get mean activation for each neuron given input data"""
        layer_outputs = {}
        x = torch.tensor(X) if not isinstance(X, torch.Tensor) else X

        layer_idx = 0
        with torch.no_grad():
            for layer in self.network:
                x = layer(x)
                if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                    layer_outputs[f"layer_{layer_idx}"] = x.numpy()  # or x.cpu().numpy()
                    layer_idx += 1

        return layer_outputs

def create_model(n_hidden_layers, neurons_per_layer, activation, learning_rate):
    model = FlexibleMLP(
        n_inputs=2,  # 2D input data
        n_hidden_layers=n_hidden_layers,
        neurons_per_layer=neurons_per_layer,
        activation=activation
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

def get_weights_from_model(model):
    """Extract weights as nested list for Svelte component"""
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Shape: [to_neurons, from_neurons]
            w = param.detach().numpy().tolist()
            weights.append(w)
    return weights


def get_prediction_grid(model, resolution=200):
    """Generate a 2D grid of model predictions for the decision boundary."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    # Flatten to (resolution^2, 2) for model input
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    # Get predictions
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points)
        predictions = model(grid_tensor).numpy()

    # Reshape back to 2D grid
    prediction_grid = predictions.reshape(resolution, resolution).tolist()

    return prediction_grid


def get_data_points(X, y):
    """Convert training data to format expected by DecisionBoundary component."""
    data_points = [
        {"x": float(X[i, 0]), "y": float(X[i, 1]), "label": int(y[i])}
        for i in range(len(X))
    ]
    return data_points