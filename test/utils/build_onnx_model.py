"""This is a helper script to train and save a simple ONNX model for testing. It
was used to generate the ../data/test_model.onnx file.

Note that the script uses multiple packages not installed by default. You will need to
run:
    pip install torch onnx onnxscript tqdm
in order to run this script.
"""

import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from redback_surrogates.learned_surrogate import LearnedSurrogateModel


torch.set_default_dtype(torch.float64)


# Define a simple sigmoid neural network
class SigmoidModel(nn.Module):
    """This is the simple neural network architecture used in the test ONNX model."""

    def __init__(self, input_size, hidden_size, output_shape):
        super(SigmoidModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size, output_shape[0] * output_shape[1])
        self.output_shape = output_shape

    def forward(self, frequency, amplitude, center, width):
        x = torch.stack([frequency, amplitude, center, width]).T
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer for regression
        x = x.view(-1, *self.output_shape)
        return x


def _test_function(
    time,
    wave,
    *,
    frequency=1.0,
    amplitude=1.0,
    center=None,
    width=None,
):
    """
    A test function that generates a Gaussian-modulated sine wave.

    Parameters:
    - time: array-like, Length T array of time values
    - wave: array-like, Length W array of wave values
    - frequency: float or array-like, frequency of the sine wave
    - amplitude: float or array-like, amplitude of the sine wave
    - center: float or array-like, center frequency of the Gaussian envelope
    - width: float or array-like, width of the Gaussian envelope

    Returns:
    - result: array-like, A T x W array of generated values.
    """
    if center is not None and width is not None:
        gaussian_envelope = np.exp(-((wave - center) ** 2) / (2 * width**2))
    else:
        gaussian_envelope = np.ones_like(wave)

    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    result = gaussian_envelope[None, :] * sine_wave[:, None]
    return result


def _build_testing_data():
    waves = np.array([1000.0, 1500.0, 2000.0])
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create fake training data.
    num_samples = 1000
    frequency = np.random.uniform(low=0.5, high=2.0, size=num_samples)
    amplitude = np.random.uniform(low=10.0, high=20.0, size=num_samples)
    center = np.random.uniform(low=1000.0, high=2000.0, size=num_samples)
    width = np.random.uniform(low=100.0, high=500.0, size=num_samples)
    y_vals = [
        _test_function(
            times,
            waves,
            frequency=frequency[idx],
            amplitude=amplitude[idx],
            center=center[idx],
            width=width[idx],
        )
        for idx in range(num_samples)
    ]

    # Convert everything to torch tensors
    frequency = torch.tensor(frequency)
    amplitude = torch.tensor(amplitude)
    center = torch.tensor(center)
    width = torch.tensor(width)
    y_tensor = torch.tensor(y_vals)

    # Configure the model and training.
    model = SigmoidModel(
        4,  # Input parameters: frequency, amplitude, center, width
        64,  # Hidden layer size
        (
            len(times),
            len(waves),
        ),  # Output shape
    )
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 1000

    print("Starting training...")
    for _ in tqdm(range(num_epochs)):
        # Forward pass
        outputs = model(frequency, amplitude, center, width)
        loss = criterion(outputs, y_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    input_example = (
        frequency[0],
        amplitude[0],
        center[0],
        width[0],
    )
    onnx_program = torch.onnx.export(model, input_example, dynamo=True)

    surrogate_model = LearnedSurrogateModel(
        onnx_program.model_proto,
        times=times,
        wavelengths=waves,
    )
    surrogate_model.to_onnx_file("../data/test_model.onnx", overwrite=True)


if __name__ == "__main__":
    _build_testing_data()
