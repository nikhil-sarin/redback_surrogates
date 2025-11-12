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
        print(repr_str)
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
