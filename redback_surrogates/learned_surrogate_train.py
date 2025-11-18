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
    scale_output : bool, optional
        Whether to scale the output values to be in the range [0, 1] during training.
        Default is True.

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

    # Scale the output values to be in the range [0, 1] for training.
    if scale_output:
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_scale = output_max - output_min
        if output_scale == 0:
            output_scale = 1.0
        output_shift = output_min
        output = (output - output_shift) / output_scale
    else:
        output_scale = 1.0
        output_shift = 0.0

    # Configure the model and training.
    model = MultilevelSigmoid(
        input_size=len(dataset.parameter_names),
        hidden_sizes=hidden_sizes,
        output_shape=output.shape[1:],
    )
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
    ).model_proto

    # Add layers to the ONNX model to undo the scaling.
    if scale_output:
        onnx_program = add_output_scaling_and_shift_layers(
            onnx_program,
            scaling_factor=output_scale.item(),
            shift=output_shift.item(),
        )

    surrogate_model = LearnedSurrogateModel(
        onnx_program,
        times=dataset.times,
        wavelengths=dataset.wavelengths,
        param_names=dataset.parameter_names,
    )
    return surrogate_model


def add_output_scaling_and_shift_layers(onnx_model, scaling_factor, shift=0.0):
    """Add a scaling and shift operation to the output of an ONNX model.
    
    The transformation applied is: output_scaled = (output * scaling_factor) + shift
    
    Parameters
    ----------
    onnx_model : onnx.ModelProto
        The ONNX model to modify.
    scaling_factor : float
        The scaling factor to multiply the output by.
    shift : float, optional
        The additive shift to apply after scaling. Default is 0.0.
    
    Returns
    -------
    onnx.ModelProto
        The modified ONNX model with scaling and shift applied to the output.
    """
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as err:
        raise ImportError(
            "The onnx package is required to modify the ONNX model. "
            "Please install it using 'pip install onnx'."
        ) from err

    # Get the graph
    graph = onnx_model.graph
    
    # Get the original output name and use it to derive two intermediate output names
    original_output_name = graph.output[0].name
    unscaled_output_name = original_output_name + "_unscaled"
    scaled_output_name = original_output_name + "_scaled"

    # Rename the original output to the intermediate name
    for node in graph.node:
        for i, output_name in enumerate(node.output):
            if output_name == original_output_name:
                node.output[i] = unscaled_output_name

    # Create a proto for constant tensor for the scaling factor
    scale_tensor_proto = numpy_helper.from_array(
        np.array([scaling_factor], dtype=np.float64),
        name="output_scaling_factor"
    )
    graph.initializer.append(scale_tensor_proto)

    # Create a Mul node to multiply the output by the scaling factor
    mul_node = helper.make_node(
        'Mul',
        inputs=[unscaled_output_name, 'output_scaling_factor'],
        outputs=[scaled_output_name],
        name='output_scaling'
    )
    graph.node.append(mul_node)

    # Create a constant tensor for the shift
    shift_tensor_proto = numpy_helper.from_array(
        np.array([shift], dtype=np.float64),
        name="output_shift"
    )
    graph.initializer.append(shift_tensor_proto)

    # Create an Add node to add the shift
    add_node = helper.make_node(
        'Add',
        inputs=[scaled_output_name, 'output_shift'],
        outputs=[original_output_name],
        name='output_shift_add'
    )
    graph.node.append(add_node)

    # Update the graph output to use the final output name
    graph.output[0].name = original_output_name

    # Run ONNX checker to validate the modified model
    onnx.checker.check_model(onnx_model)

    return onnx_model
