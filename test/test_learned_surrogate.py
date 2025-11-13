import numpy as np
import unittest
import warnings

from pathlib import Path

from redback_surrogates.learned_surrogate import (
    assert_safe_param_names,
    LearnedSurrogateModel,
)

try:
    import torch
    import torch.nn as nn

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


class TestLearnedSurrogateModel(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = Path(__file__).parent / "data"

    def tearDown(self) -> None:
        pass

    def test_assert_safe_param_names(self):
        """Test that assert_safe_param_names works as expected."""
        # Valid names should not raise an error.
        valid_names = ["param1", "param_2", "Param3", "_param4"]
        assert_safe_param_names(valid_names)

        # Invalid names should raise an error.
        invalid_names = [
            "1param",
            "param-2",
            "param 3",
            "param$4",
            "param;x",
            "param)x",
            "a b",
            "a.b",
        ]
        for name in invalid_names:
            with self.assertRaises(ValueError):
                assert_safe_param_names([name])

    def test_learned_surrogate_from_pytorch_onnx_file(self):
        """Test that we can load a surrogate model from a file."""
        model = LearnedSurrogateModel.from_onnx_file(
            self.data_dir / "test_pytorch_model.onnx"
        )
        assert model.times is not None
        assert model.wavelengths is not None
        assert np.array_equal(model.param_names, ["freq", "amp", "center", "width"])

        # Test that we can use the dynamically created predict method to get outputs.
        output = model.predict_spectra_grid(
            freq=1.0, amp=10.0, center=1500.0, width=100.0
        )
        assert output.shape == (len(model.times), len(model.wavelengths))

        # Test that we can read the __repr__ output.
        repr_str = repr(model)
        assert "LearnedSurrogateModel with 4 parameters" in repr_str
        assert "Times Dimension: 5 steps [0.1, 0.5]" in repr_str
        assert "Wavelengths Dimension: 3 steps [1000.0, 2000.0]" in repr_str
        assert "freq: The freq of the sine wave in Hz" in repr_str
        assert "amp: The amp of the sine wave." in repr_str
        assert (
            "center: The center freq of the Gaussian envelope in Angstroms." in repr_str
        )
        assert "width: The width of the Gaussian envelope in Angstroms." in repr_str

        # Test that we can overwrite and retrieve parameter info.
        model.add_parameter_info("freq", "This is a freq param.")
        repr_str = repr(model)
        assert "freq: This is a freq param." in repr_str

        self.assertRaises(
            ValueError, model.add_parameter_info, "nonexistent_param", "Info"
        )

    def test_learned_surrogate_from_scikit_onnx_file(self):
        """Test that we can load a surrogate model from a file."""
        model = LearnedSurrogateModel.from_onnx_file(
            self.data_dir / "test_scikit_model.onnx"
        )
        assert model.times is not None
        assert model.wavelengths is not None
        assert np.array_equal(
            model.param_names, ["frequency", "amplitude", "center", "width"]
        )

        # Test that we can use the dynamically created predict method to get outputs.
        output = model.predict_spectra_grid(
            frequency=1.0, amplitude=10.0, center=1500.0, width=100.0
        )
        assert output.shape == (len(model.times), len(model.wavelengths))

    def test_learned_surrogate_from_flat_scikit_onnx_file(self):
        """Test that we can load a surrogate model from a file."""
        model = LearnedSurrogateModel.from_onnx_file(
            self.data_dir / "test_flat_scikit_model.onnx"
        )
        assert model.times is not None
        assert len(model.times) == 3
        assert model.wavelengths is not None
        assert len(model.wavelengths) == 2
        assert np.array_equal(model.param_names, ["frequency", "amplitude"])

        # Test that we can use the dynamically created predict method to get outputs.
        output = model.predict_spectra_grid(
            frequency=1.0, amplitude=10.0, center=1500.0, width=100.0
        )
        assert output.shape == (len(model.times), len(model.wavelengths))
        assert np.allclose(output, np.array([[1, 5], [2, 10], [3, 15]]))

    @unittest.skipUnless(TORCH_INSTALLED, "PyTorch is not installed")
    def test_pytorch_model(self):
        torch.set_default_dtype(torch.float64)

        # Define a simple sigmoid neural network architecture for testing.
        class SigmoidModel(torch.nn.Module):
            """This is the simple neural network architecture used in the test ONNX model."""

            def __init__(self, input_size, hidden_size, output_shape):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.sigmoid1 = nn.Sigmoid()
                self.fc2 = nn.Linear(hidden_size, output_shape[0] * output_shape[1])
                self.output_shape = output_shape

            def forward(self, freq, amp):
                x = torch.column_stack([freq, amp])
                x = self.sigmoid1(self.fc1(x))
                x = self.fc2(x)  # No activation on final layer for regression
                x = x.view(-1, *self.output_shape)
                return x

        # Configure the model and run a single forward pass to initialize weights.
        model = SigmoidModel(
            2,  # Input size (2 parameters)
            8,  # Hidden layer size
            (2, 3),  # Output shape (2x3)
        )
        model.forward(torch.tensor(1.1), torch.tensor(1.1))

        # Create the surrogate model from the trained PyTorch model.
        times = np.array([0.1, 0.2])
        wavelengths = np.array([1000.0, 1500.0, 2000.0])
        model = LearnedSurrogateModel.from_pytorch_model(model, times, wavelengths)

        assert np.allclose(model.times, times)
        assert np.allclose(model.wavelengths, wavelengths)
        assert np.array_equal(model.param_names, ["freq", "amp"])
        assert model.output_shape == (len(times), len(wavelengths))

        # Test that we can use the dynamically created predict method to get outputs.
        output = model.predict_spectra_grid(freq=1.0, amp=10.0)
        assert output.shape == (len(model.times), len(model.wavelengths))
