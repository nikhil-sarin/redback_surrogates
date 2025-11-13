"""This is a helper script to train and save a simple ONNX model for testing. It
was used to generate the ../data/test_model.onnx file.

Note that the script uses multiple packages not installed by default. You will need to
run:
    pip install torch onnx onnxscript tqdm
in order to run this script.
"""

import numpy as np
import torch

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from skl2onnx import to_onnx
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

    def forward(self, freq, amp, center, width):
        x = torch.column_stack([freq, amp, center, width])
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer for regression
        x = x.view(-1, *self.output_shape)
        return x


def _test_function(
    time,
    wave,
    *,
    freq=1.0,
    amp=1.0,
    center=None,
    width=None,
):
    """
    A test function that generates a Gaussian-modulated sine wave.

    Parameters:
    - time: array-like, Length T array of time values
    - wave: array-like, Length W array of wave values
    - freq: float or array-like, freq of the sine wave
    - amp: float or array-like, amp of the sine wave
    - center: float or array-like, center freq of the Gaussian envelope
    - width: float or array-like, width of the Gaussian envelope

    Returns:
    - result: array-like, A T x W array of generated values.
    """
    if center is not None and width is not None:
        gaussian_envelope = np.exp(-((wave - center) ** 2) / (2 * width**2))
    else:
        gaussian_envelope = np.ones_like(wave)

    sine_wave = amp * np.sin(2 * np.pi * freq * time)
    result = gaussian_envelope[None, :] * sine_wave[:, None]
    return result


def _build_testing_data():
    waves = np.array([1000.0, 1500.0, 2000.0])
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create fake training data.
    num_samples = 1000
    freq = np.random.uniform(low=0.5, high=2.0, size=num_samples)
    amp = np.random.uniform(low=10.0, high=20.0, size=num_samples)
    center = np.random.uniform(low=1000.0, high=2000.0, size=num_samples)
    width = np.random.uniform(low=100.0, high=500.0, size=num_samples)
    y_vals = [
        _test_function(
            times,
            waves,
            freq=freq[idx],
            amp=amp[idx],
            center=center[idx],
            width=width[idx],
        )
        for idx in range(num_samples)
    ]
    y_vals = np.array(y_vals)
    return waves, times, freq, amp, center, width, y_vals


def _train_pytorch_model(waves, times, freq, amp, center, width, y_vals):
    # Convert everything to torch tensors
    freq = torch.tensor(freq)
    amp = torch.tensor(amp)
    center = torch.tensor(center)
    width = torch.tensor(width)
    y_tensor = torch.tensor(y_vals)

    # Configure the model and training.
    model = SigmoidModel(
        4,  # Input parameters: freq, amp, center, width
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
        outputs = model(freq, amp, center, width)
        loss = criterion(outputs, y_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    input_example = (
        freq[0],
        amp[0],
        center[0],
        width[0],
    )
    onnx_program = torch.onnx.export(model, input_example, dynamo=True)

    surrogate_model = LearnedSurrogateModel(
        onnx_program.model_proto,
        times=times,
        wavelengths=waves,
    )
    surrogate_model.add_parameter_info("freq", "The freq of the sine wave in Hz.")
    surrogate_model.add_parameter_info("amp", "The amp of the sine wave.")
    surrogate_model.add_parameter_info("center", "The center freq of the Gaussian envelope in Angstroms.")
    surrogate_model.add_parameter_info("width", "The width of the Gaussian envelope in Angstroms.")
    surrogate_model.to_onnx_file("../data/test_pytorch_model.onnx", overwrite=True)


def _train_scikit_model(waves, times, frequency, amplitude, center, width, y_vals):
    x = np.column_stack([frequency, amplitude, center, width])
    y_vals = np.array([y.flatten() for y in y_vals])
    model = MultiOutputRegressor(LinearRegression())
    model.fit(x, y_vals)
    onnx_model = to_onnx(model, x[:1])

    surrogate_model = LearnedSurrogateModel(
        onnx_model,
        times=times,
        wavelengths=waves,
        param_names=["frequency", "amplitude", "center", "width"],
    )
    surrogate_model.to_onnx_file("../data/test_scikit_model.onnx", overwrite=True)


def _train_flat_scikit_model():
    x1, x2 = np.meshgrid(
        np.linspace(0.5, 2.0, 10),  # frequency
        np.linspace(10.0, 20.0, 10),  # amplitude
        indexing="ij",
    )
    x = np.column_stack([x1.ravel(), x2.ravel()])
    num_points = x.shape[0]

    y_vals = np.tile([1, 5, 2, 10, 3, 15], num_points).reshape(num_points, 6)
    model = MultiOutputRegressor(LinearRegression())
    model.fit(x, y_vals)
    onnx_model = to_onnx(model, x[:1])

    surrogate_model = LearnedSurrogateModel(
        onnx_model,
        times=np.array([0.1, 0.2, 0.3]),
        wavelengths=np.array([1000.0, 2000.0]),
        param_names=["frequency", "amplitude"],
    )
    surrogate_model.to_onnx_file("../data/test_flat_scikit_model.onnx", overwrite=True)


def _train_and_save_models():
    waves, times, frequency, amplitude, center, width, y_vals = _build_testing_data()
    _train_pytorch_model(
        waves,
        times,
        frequency,
        amplitude,
        center,
        width,
        y_vals,
    )
    _train_scikit_model(
        waves,
        times,
        frequency,
        amplitude,
        center,
        width,
        y_vals,
    )
    _train_flat_scikit_model()

if __name__ == "__main__":
    _train_and_save_models()
