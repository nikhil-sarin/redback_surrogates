import inspect
import numpy as np
import types
import unittest

from pathlib import Path

from redback_surrogates.learned_surrogate import LearnedSurrogateModel


class TestLearnedSurrogateModel(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = Path(__file__).parent / "data"

    def tearDown(self) -> None:
        pass

    def test_learned_surrogate_from_onnx_file(self):
        """Test that we can load a surrogate model from a file."""
        model = LearnedSurrogateModel.from_onnx_file(self.data_dir / "test_model.onnx")
        assert model.times is not None
        assert model.wavelengths is not None
        assert np.array_equal(
            model.param_names, ["frequency", "amplitude", "center", "width"]
        )

        # Test that we correctly created a dynamic predict method.
        assert hasattr(model, "predict")
        assert isinstance(model.predict, types.MethodType)
        signature = inspect.getfullargspec(model.predict)
        assert signature.args == ["self", "frequency", "amplitude", "center", "width"]

        # Test that we can use the dynamically created predict method to get outputs.
        output = model.predict(
            frequency=1.0, amplitude=10.0, center=1500.0, width=100.0
        )[0]
        assert output.shape == (1, len(model.times), len(model.wavelengths))
