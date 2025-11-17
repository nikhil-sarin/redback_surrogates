import numpy as np
import unittest

from astropy.table import Table

from redback_surrogates.learned_surrogate import LearnedSurrogateModel
from redback_surrogates.learned_surrogate_data import LearnedSurrogateDataset
from redback_surrogates.learned_surrogate_train import train_pytorch_model

try:
    import torch

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


class TestLearnedSurrogateTrain(unittest.TestCase):

    @unittest.skipUnless(TORCH_INSTALLED, "PyTorch is not installed")
    def test_learned_surrogate_train_basic(self):
        """Test that we can train a basic surrogate model from a data set."""
        torch.set_default_dtype(torch.float64)

        def _toy_function(x, y, z):
            return np.array(
                [
                    [x, y, z],
                    [0.1 * z, 0.1 * y, 0.1 * x],
                ]
            )

        num_samples = 1_000
        np.random.seed(42)  # Set the seed for reproducibility
        x_vals = 0.5 * np.random.rand(num_samples)
        y_vals = 0.5 * np.random.rand(num_samples)
        z_vals = np.random.rand(num_samples) - 0.5
        params = {
            "x": x_vals,
            "y": y_vals,
            "z": z_vals,
            "output": [
                _toy_function(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)
            ],
        }
        data = Table(params)

        times = np.array([0.1, 0.2])
        wavelengths = np.array([1000.0, 2000.0, 3000.0])
        dataset = LearnedSurrogateDataset(
            data,
            output_column="output",
            times=times,
            wavelengths=wavelengths,
        )

        # Learn a model with a single hidden layer with 8 nodes.
        surrogate_model = train_pytorch_model(
            dataset, hidden_sizes=8, training_epochs=1000
        )
        assert isinstance(surrogate_model, LearnedSurrogateModel)
        assert np.allclose(surrogate_model.times, times)
        assert np.allclose(surrogate_model.wavelengths, wavelengths)
        assert surrogate_model.param_names == ["x", "y", "z"]

        # Test that the surrogate model produces outputs of the correct shape.
        grid = surrogate_model.predict_spectra_grid(x=0.2, y=0.1, z=-0.1)
        print(grid)
        print(_toy_function(0.2, 0.1, -0.1))
        assert grid.shape == (2, 3)
        assert np.allclose(grid, _toy_function(0.2, 0.1, -0.1), atol=0.1)
