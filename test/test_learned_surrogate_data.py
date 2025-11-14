import numpy as np
import tempfile
import unittest

from astropy.table import Table
from pathlib import Path

from redback_surrogates.learned_surrogate_data import LearnedSurrogateDataset


class TestLearnedSurrogateDataset(unittest.TestCase):

    def test_learned_surrogate_dataset_basic(self):
        """Test that we can create a basic LearnedSurrogateDataset."""
        data = Table(
            {
                "param1": [1, 2, 3],
                "param2": [4, 5, 6],
                "grids": [
                    np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]]),
                    np.array([[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]]),
                    np.array([[0.9, 1.0, 1.1], [1.1, 1.2, 1.3]]),
                ],
            }
        )
        dataset = LearnedSurrogateDataset(data, output_column="grids")
        assert len(dataset) == 3
        assert dataset.parameter_names == ["param1", "param2"]
        assert dataset._output_column == "grids"
        assert dataset.times is None
        assert dataset.wavelengths is None

        # We can access the input parameters.
        assert np.array_equal(dataset.get_input(), np.array([[1, 4], [2, 5], [3, 6]]))
        assert np.array_equal(dataset.get_input(0), np.array([1, 4]))
        assert np.array_equal(dataset.get_input([0, 2]), np.array([[1, 4], [3, 6]]))

        # We can access the output grids.
        assert dataset.get_output().shape == (3, 2, 3)
        assert np.array_equal(
            dataset.get_output(0), np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]])
        )
        assert np.array_equal(
            dataset.get_output([0, 2]),
            np.array(
                [[[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]], [[0.9, 1.0, 1.1], [1.1, 1.2, 1.3]]]
            ),
        )

        # We fail if the grid column isn't in the data.
        with self.assertRaises(ValueError):
            LearnedSurrogateDataset(data, output_column="nonexistent_column")

        # We can add metadata.
        times = np.array([0.1, 0.5])
        wavelengths = np.array([1000.0, 2000.0, 3000.0])
        dataset = LearnedSurrogateDataset(
            data, output_column="grids", times=times, wavelengths=wavelengths
        )
        assert np.array_equal(dataset.times, times)
        assert np.array_equal(dataset.wavelengths, wavelengths)

    def test_learned_surrogate_dataset_to_from_file(self):
        """Test that we can save and load a LearnedSurrogateDataset."""
        data = Table(
            {
                "param1": [1, 2, 3],
                "param2": [4, 5, 6],
                "grids": [
                    np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]]),
                    np.array([[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]]),
                    np.array([[0.9, 1.0, 1.1], [1.1, 1.2, 1.3]]),
                ],
            }
        )
        times = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9])
        wavelengths = np.array([1000.0, 2000.0, 3000.0])
        dataset = LearnedSurrogateDataset(
            data,
            output_column="grids",
            times=times,
            wavelengths=wavelengths,
        )
        assert dataset.in_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "dataset.ecsv"

            # The data set doesn't exist yet.
            assert not filename.exists()
            with self.assertRaises(FileNotFoundError):
                LearnedSurrogateDataset.from_file(filename)

            # Save the dataset and check that it exists.
            dataset.save(filename)
            assert filename.exists()

            # Load the dataset and check that it matches the original, including the metadata.
            loaded_dataset = LearnedSurrogateDataset.from_file(filename)
            assert loaded_dataset.in_memory
            assert len(loaded_dataset) == 3
            assert loaded_dataset.parameter_names == ["param1", "param2"]
            assert loaded_dataset._output_column == "grids"
            assert np.array_equal(loaded_dataset.times, times)
            assert np.array_equal(loaded_dataset.wavelengths, wavelengths)

            # We can access slices of the input parameters.
            assert np.array_equal(loaded_dataset.get_input(0), np.array([1, 4]))
            assert np.array_equal(
                loaded_dataset.get_input([0, 2]), np.array([[1, 4], [3, 6]])
            )

            # We can access the output grids.
            assert np.array_equal(
                loaded_dataset.get_output(0),
                np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]]),
            )
            assert np.array_equal(
                loaded_dataset.get_output([0, 2]),
                np.array(
                    [
                        [[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]],
                        [[0.9, 1.0, 1.1], [1.1, 1.2, 1.3]],
                    ]
                ),
            )

            # We fail if we try to write again with overwrite=False.
            with self.assertRaises(FileExistsError):
                dataset.save(filename, overwrite=False)

    def test_learned_surrogate_dataset_to_from_separate_files(self):
        """Test that we can save and load a LearnedSurrogateDataset using separate files."""
        data = Table(
            {
                "param1": [1, 2, 3],
                "param2": [4, 5, 6],
                "grids": [
                    np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]]),
                    np.array([[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]]),
                    np.array([[0.9, 1.0, 1.1], [1.1, 1.2, 1.3]]),
                ],
            }
        )
        times = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9])
        wavelengths = np.array([1000.0, 2000.0, 3000.0])
        dataset = LearnedSurrogateDataset(
            data,
            output_column="grids",
            times=times,
            wavelengths=wavelengths,
        )
        assert dataset.in_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "dataset_sep.ecsv"
            dataset.save(filename, separate_files=True)

            # The main data set file and each output file should exist.
            assert filename.exists()
            for idx in range(len(dataset)):
                assert (filename.parent / f"{filename.stem}_{idx}.npy").exists()

            # Load the dataset and check that it matches the original, including the metadata.
            loaded_dataset = LearnedSurrogateDataset.from_file(filename)
            assert not loaded_dataset.in_memory
            assert len(loaded_dataset) == 3
            assert loaded_dataset.parameter_names == ["param1", "param2"]
            assert loaded_dataset._output_column == "grids"
            assert np.array_equal(loaded_dataset.times, times)
            assert np.array_equal(loaded_dataset.wavelengths, wavelengths)

            # We can access slices of the input parameters.
            assert np.array_equal(loaded_dataset.get_input(0), np.array([1, 4]))
            assert np.array_equal(
                loaded_dataset.get_input([0, 2]), np.array([[1, 4], [3, 6]])
            )

            # We can access the output grids.
            assert np.array_equal(
                loaded_dataset.get_output(0),
                np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]]),
            )
            assert np.array_equal(
                loaded_dataset.get_output([0, 2]),
                np.array(
                    [
                        [[0.1, 0.2, 0.3], [0.3, 0.4, 5.0]],
                        [[0.9, 1.0, 1.1], [1.1, 1.2, 1.3]],
                    ]
                ),
            )

    def test_learned_surrogate_dataset_split(self):
        """Test that we can split a LearnedSurrogateDataset."""
        data = Table(
            {
                "param1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "param2": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                "param3": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "filename": [f"file_{i}.npy" for i in range(10)],
            }
        )
        times = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9])
        wavelengths = np.array([1000.0, 2000.0, 3000.0])

        dataset = LearnedSurrogateDataset(
            data, output_column="filename", times=times, wavelengths=wavelengths
        )

        assert len(dataset) == 10
        assert dataset.parameter_names == ["param1", "param2", "param3"]
        assert dataset._output_column == "filename"
        assert np.array_equal(dataset.times, times)
        assert np.array_equal(dataset.wavelengths, wavelengths)

        train_dataset, test_dataset = dataset.split(0.2)
        assert len(train_dataset) == 8
        assert len(test_dataset) == 2
        assert train_dataset.parameter_names == ["param1", "param2", "param3"]
        assert test_dataset.parameter_names == ["param1", "param2", "param3"]
        assert train_dataset._output_column == "filename"
        assert test_dataset._output_column == "filename"
        assert np.array_equal(train_dataset.times, times)
        assert np.array_equal(test_dataset.times, times)
        assert np.array_equal(train_dataset.wavelengths, wavelengths)
        assert np.array_equal(test_dataset.wavelengths, wavelengths)

        # The data sets should not overlap. We just check the first parameter.
        train_inputs = set([row[0] for row in train_dataset.get_input()])
        test_inputs = set([row[0] for row in test_dataset.get_input()])
        assert train_inputs.isdisjoint(test_inputs)
