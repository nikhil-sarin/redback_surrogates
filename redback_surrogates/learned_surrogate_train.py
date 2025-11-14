"""Basic training for LearnedSurrogateModels using a neural network and pytorch."""

import numpy as np
import torch
import torch.nn as nn

import inspect

from tqdm import tqdm

from redback_surrogates.learned_surrogate import (
    assert_safe_param_names,
    LearnedSurrogateModel,
)


# Define a simple sigmoid neural network
class MultilevelSigmoid(nn.Module):
    """This is the simple neural network architecture used to train the surrogate model. It
    consists of several hidden layers with sigmoid activations and a final linear layer to
    produce the output grid.

    Parameters
    ----------
    input_params : list or array-like
        The names of the input parameters.
    hidden_sizes : int or list of int
        The size(s) of each of the hidden layers.
    output_shape : tuple
        The shape of the output grid (e.g., (num_times, num_wavelengths)).
    """

    def __init__(self, input_size, hidden_sizes, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        if np.isscalar(hidden_sizes):
            hidden_sizes = [hidden_sizes]

        # Create each of the hidden layers.
        self.lin_layers = []
        self.sigmoid_layers = []
        prev_size = self.input_size
        for curr_size in hidden_sizes:
            self.lin_layers.append(nn.Linear(prev_size, curr_size))
            self.sigmoid_layers.append(nn.Sigmoid())
            prev_size = curr_size

        # Create the output layer.
        self.out_layer = nn.Linear(prev_size, output_shape[0] * output_shape[1])

    def forward(self, *params):
        x = torch.column_stack(params)
        for lin, sigmoid in zip(self.lin_layers, self.sigmoid_layers):
            x = sigmoid(lin(x))
        x = self.out_layer(x)
        x = x.view(-1, *self.output_shape)
        return x


def train_pytorch_model(
    dataset,
    hidden_sizes=[64, 64],
    training_epochs=100,
):
    """Trains a simple neural network surrogate model using PyTorch.

    Parameters
    ----------
    dataset : LearnedSurrogateDataset
        The dataset containing the training data.
    training_epochs : int, optional
        The number of epochs to train the model. Default is 100.
    hidden_sizes : int or list of int, optional
        The size(s) of the hidden layers in the neural network.
        Default is a pair of 64 node layers.

    Returns
    -------
    LearnedSurrogateModel
        The trained surrogate model.
    """
    torch.set_default_dtype(torch.float64)

    # Get the input and outputs and convert to tensors.
    input_raw = dataset.get_input()
    input = [
        torch.tensor(input_raw[:, i], dtype=torch.float64)
        for i in range(input_raw.shape[1])
    ]
    output = torch.tensor(dataset.get_output(), dtype=torch.float64)

    # Configure the model and training.
    model = MultilevelSigmoid(
        input_size=len(dataset.parameter_names),
        hidden_sizes=hidden_sizes,
        output_shape=output.shape[1:],
    )
    print(list(model.parameters()))
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("Starting training...")
    for _ in tqdm(range(training_epochs)):
        # Forward pass
        outputs = model(*input)
        loss = criterion(outputs, output)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Create a LearnedSurrogateModel from the trained PyTorch model.
    input_example = tuple(input[i][0] for i in range(len(input)))
    onnx_program = torch.onnx.export(
        model,
        input_example,
        input_names=dataset.parameter_names,
        dynamo=True,
    )
    surrogate_model = LearnedSurrogateModel(
        onnx_program.model_proto,
        times=dataset.times,
        wavelengths=dataset.wavelengths,
        param_names=dataset.parameter_names,
    )
    return surrogate_model
