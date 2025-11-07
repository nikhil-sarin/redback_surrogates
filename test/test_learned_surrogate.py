import numpy as np
import unittest

from pathlib import Path

from redback_surrogates.learned_surrogate import (
    assert_safe_param_names,
    LearnedSurrogateModel,
)


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
        model = LearnedSurrogateModel.from_onnx_file(self.data_dir / "test_pytorch_model.onnx")
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

    def test_learned_surrogate_from_scikit_onnx_file(self):
        """Test that we can load a surrogate model from a file."""
        model = LearnedSurrogateModel.from_onnx_file(self.data_dir / "test_scikit_model.onnx")
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
        model = LearnedSurrogateModel.from_onnx_file(self.data_dir / "test_flat_scikit_model.onnx")
        assert model.times is not None
        assert len(model.times) == 3
        assert model.wavelengths is not None
        assert len(model.wavelengths) == 2
        assert np.array_equal(
            model.param_names, ["frequency", "amplitude"]
        )

        # Test that we can use the dynamically created predict method to get outputs.
        output = model.predict_spectra_grid(
            frequency=1.0, amplitude=10.0, center=1500.0, width=100.0
        )
        assert output.shape == (len(model.times), len(model.wavelengths))
        assert np.allclose(output, np.array([[1, 5], [2, 10], [3, 15]]))
