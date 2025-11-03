import unittest

from pathlib import Path

from redback_surrogates.learned_surrogate import LearnedSurrogateModel


class TestLearnedSurrogateModel(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = Path(__file__).parent / "data"

    def tearDown(self) -> None:
        pass

    def test_learned_surrogate_from_onnx_file(self):
        model = LearnedSurrogateModel.from_onnx_file(self.data_dir / "test_model.onnx")
