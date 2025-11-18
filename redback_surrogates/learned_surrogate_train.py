"""Basic training for LearnedSurrogateModels using a neural network and pytorch."""

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from redback_surrogates.learned_surrogate import LearnedSurrogateModel


class NormalizedMultilevelSigmoid(nn.Module):
    """Wrapper that combines input normalization with the neural network."""

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_shape,
        min_vals=None,
        max_vals=None,
    ):
        """Initialize the model.

        :param input_size: The number of input parameters.
        :param hidden_sizes: The size(s) of the hidden layers.
        :param output_shape: The shape of the output (e.g., (num_times, num_wavelengths)).
        :param min_vals: Optional tensor of minimum values for each input parameter for normalization.
        :param max_vals: Optional tensor of maximum values for each input parameter for normalization.
        """
        super().__init__()

        # Save the network configuration.
        self.output_shape = output_shape
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if np.isscalar(hidden_sizes):
            hidden_sizes = [hidden_sizes]

        # If we have range information, compute the scaling factors for normalization.
        if min_vals is not None and max_vals is not None:
            if len(min_vals) != input_size or len(max_vals) != input_size:
                raise ValueError(
                    f"Length of min_vals ({len(min_vals)}) and max_vals ({len(max_vals)}) "
                    f"must match input_size ({input_size})."
                )
            min_vals_tensor = torch.clone(min_vals)
            max_vals_tensor = torch.clone(max_vals)

            # Compute scale for each parameter, avoiding division by zero
            ranges = max_vals_tensor - min_vals_tensor
            ranges = torch.maximum(ranges, torch.tensor(1e-8, dtype=torch.float64))
            scales = 1.0 / ranges
        else:
            # If no range information is provided, use identity normalization.
            min_vals_tensor = torch.zeros(input_size, dtype=torch.float64)
            scales = torch.ones(input_size, dtype=torch.float64)

        self.register_buffer("input_shift", min_vals_tensor)
        self.register_buffer("input_scale", scales)

        # Create the multilevel sigmoid network.
        self.lin_layers = []
        self.sigmoid_layers = []
        prev_size = self.input_size
        for curr_size in hidden_sizes:
            self.lin_layers.append(nn.Linear(prev_size, curr_size))
            self.sigmoid_layers.append(nn.Sigmoid())
            prev_size = curr_size
        self.out_layer = nn.Linear(prev_size, output_shape[0] * output_shape[1])

    def forward(self, *params):
        x = torch.column_stack(params)
        # Do normalization (no-op if no min/max provided)
        x = (x - self.input_shift) * self.input_scale

        # Propagate through each layer of the network
        for lin, sigmoid in zip(self.lin_layers, self.sigmoid_layers):
            x = sigmoid(lin(x))
        x = self.out_layer(x)
        x = x.view(-1, *self.output_shape)
        return x


def train_pytorch_model(
    dataset,
    hidden_sizes=[64, 64],
    training_epochs=100,
    scale_output=True,
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

    # Get the input, its min/max bounds, and the outputs.
    input_vals = torch.tensor(dataset.get_input(), dtype=torch.float64).T
    min_vals = torch.min(input_vals, dim=1).values
    max_vals = torch.max(input_vals, dim=1).values
    output = torch.tensor(dataset.get_output(), dtype=torch.float64)

    # Configure the model and training.
    model = NormalizedMultilevelSigmoid(
        input_size=len(dataset.parameter_names),
        hidden_sizes=hidden_sizes,
        output_shape=output.shape[1:],
        min_vals=min_vals,
        max_vals=max_vals,
    )
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("Starting training...")
    for _ in tqdm(range(training_epochs)):
        # Forward pass
        outputs = model(*input_vals)
        loss = criterion(outputs, output)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Create a LearnedSurrogateModel from the trained PyTorch model.
    input_example = tuple(input_vals[:, 0])
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
