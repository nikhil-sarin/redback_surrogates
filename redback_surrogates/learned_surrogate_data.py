"""This model defines a wrapper data set class for training LearnedSurrogateModels.

These models hold a table of parameters and associated grid data, which can either
be stored directly in the table or as references to external files (e.g., NumPy files).
"""

from astropy.table import Table
from pathlib import Path

import numpy as np


class LearnedSurrogateDataset:
    """A base class for datasets representing functions where the input is a set of
    parameters and the output is a grid of values.

    The dataset is defined by a parameter table (e.g., CSV, Parquet) where each row
    corresponds to a unique set of parameters and a grid of values associated with those
    parameters. The grid data can be stored directly in the table or as references to
    external files (e.g., NumPy files).

    :param data: An astropy Table containing the parameter sets and associated grid data.
    :param output_column: An optional string with the name of the Table column that contains
        the grid data or references to it.
    :param times: An optional array-like object representing the time points associated with
        the grid data.
    :param wavelengths: An optional array-like object representing the wavelength points
        associated with the grid data.
    """

    def __init__(self, data, output_column="filename", times=None, wavelengths=None):
        self.data = data
        self._output_column = output_column
        if output_column not in data.colnames:
            raise ValueError(f"Grid column '{output_column}' not found in the dataset.")
        self._in_files = np.issubdtype(data[output_column].dtype, np.str_)
        self.parameter_names = [col for col in data.colnames if col != output_column]
        self.times = times
        self.wavelengths = wavelengths

    def __len__(self):
        return len(self.data)

    @property
    def in_memory(self):
        """Returns True if the output data is stored directly in the table, False if it is
        stored as file references.
        """
        return not self._in_files

    @classmethod
    def from_file(cls, filename: str | Path):
        """Creates a LearnedSurrogateDataset from a file.

        :param filename: A string or Path specifying the path to the dataset file.
        :param output_column: An optional string with the name of the Table column that contains
            the grid data or references to it.
        :return: An instance of LearnedSurrogateDataset.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Unable to find the data file: {filename}")
        data = Table.read(filename)

        # Load load the metadata that is present in the file, if it exists.
        output_column = data.meta.get("output_column", "filename")
        times = data.meta.get("times", None)
        if times is not None:
            times = np.asarray(times)
        wavelengths = data.meta.get("wavelengths", None)
        if wavelengths is not None:
            wavelengths = np.asarray(wavelengths)
        return cls(
            data, output_column=output_column, times=times, wavelengths=wavelengths
        )

    def get_input(self, idx=None):
        """Retrieves the input parameters for a given index.

        :param idx: An integer or array specifying the index/indices of the desired data.
            If None, returns all input parameters.
        :return: A numpy array containing the parameter values as columns (in the same order
            as the parameter names).
        """
        if idx is None:
            idx = np.arange(len(self.data))
        elif np.isscalar(idx):
            return np.asarray([self.data[param][idx] for param in self.parameter_names])
        return np.column_stack(
            [self.data[param][idx] for param in self.parameter_names]
        )

    def get_output(self, idx=None):
        """Retrieves the output (grid) data for a given index or indices.

        :param idx: An integer or array specifying the index/indices of the desired data.
            If None, returns all output data.
        :return: A NumPy array containing the output values.
        """
        if idx is None:
            idx = np.arange(len(self.data))
        if np.isscalar(idx):
            row_data = [self.data[self._output_column][idx]]
            single_index = True
        else:
            row_data = self.data[self._output_column][idx]
            single_index = False

        # If the output column gives file names, load the data from those files.
        if self._in_files:
            filenames = row_data
            row_data = []
            for idx, filename in enumerate(filenames):
                row_data.append(np.load(filename))

        # Return a single array if the input index is a scalar, otherwise return a
        # multi-dimensional array with all the data.
        if single_index:
            return np.asarray(row_data[0])
        return np.asarray(row_data)

    def load_output_to_memory(self):
        """Loads all output data into memory if it is currently stored as file references.

        This is useful if the output data is stored as file references and we want to
        load it all at once for faster access.
        """
        if not self._in_files:
            return  # Output data is already in memory

        # Read each file into memory and save it as a new column in the table.
        new_output_column = f"{self._output_column}_loaded"
        loaded_data = []
        for idx in range(len(self.data)):
            filename = self.data[self._output_column][idx]
            loaded_data.append(np.load(filename))
        self.data[new_output_column] = loaded_data

        # Update the output column name and mark it as loaded.
        self._output_column = new_output_column
        self._in_files = False

    def export_output_to_files(self, file_prefix):
        """Writes all the output data to separate files and updates the dataset
        to reference those files.

        :param file_prefix: A string specifying the prefix for the output files. The files
            will be named as "{file_prefix}_{idx}.npy" where idx is the index of the data point.
        """
        if self._in_files:
            return  # Output data is already in separate files

        filenames = []
        for idx in range(len(self.data)):
            output_data = self.data[self._output_column][idx]
            filename = f"{file_prefix}_{idx}.npy"
            print(f"Saving output data for index {idx} to file: {filename}")
            np.save(filename, output_data)
            filenames.append(filename)
        self.data[self._output_column] = filenames
        self._in_files = True

    def save(self, filename, overwrite=False, separate_files=False):
        """Saves the dataset to a file.

        :param filename: A string or Path specifying the path to save the dataset file.
        :param overwrite: A boolean indicating whether to overwrite the file if it already exists.
        :param separate_files: A boolean indicating whether to save the output grids as separate
            files (e.g., NumPy files) instead of directly in the table. If True, the output
            grids will be saved as separate files and the table will contain references to those
            files.
        """
        filename = Path(filename)
        if filename.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filename}")

        # Handle converting to/from separate files.
        if not separate_files:
            self.load_output_to_memory()
        else:
            self.export_output_to_files(file_prefix=filename.parent / filename.stem)

        self.data.meta["output_column"] = self._output_column
        if self.times is not None:
            self.data.meta["times"] = self.times.tolist()
        if self.wavelengths is not None:
            self.data.meta["wavelengths"] = self.wavelengths.tolist()
        self.data.write(filename, overwrite=True)

    def split(self, test_size=0.2, random_state=None):
        """Splits the dataset into training and testing sets.

        :param test_size: A float between 0 and 1 representing the proportion of the dataset
            to include in the test split.
        :param random_state: An optional integer seed for reproducibility.
        :return: A tuple containing the training and testing datasets as LearnedSurrogateDataset
            instances.
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size must be a float between 0 and 1.")

        np.random.seed(random_state)
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        split_idx = int(len(self.data) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_data = LearnedSurrogateDataset(
            self.data[train_indices],
            output_column=self._output_column,
            times=self.times,
            wavelengths=self.wavelengths,
        )
        test_data = LearnedSurrogateDataset(
            self.data[test_indices],
            output_column=self._output_column,
            times=self.times,
            wavelengths=self.wavelengths,
        )

        return (train_data, test_data)


def create_data_set_from_function(
    func,
    parameters,
    times=None,
    wavelengths=None,
    filename=None,
    overwrite=False,
    separate_files=False,
    **kwargs,
):
    """Creates a LearnedSurrogateDataset from a redback-style function.

    :param func: A callable that takes a set of parameters and returns a grid of values.
    :param parameters: An astropy table with columns corresponding to the parameters required by the
        function. Each row should represent a unique set of parameters.
    :param times: An array-like object representing the time points associated with the
        grid data.
    :param wavelengths: An array-like object representing the wavelength points associated
        with the grid data.
    :param filename: An optional string or Path specifying the path to save the dataset file.
        If provided the dataset will be saved to this file.
    :param overwrite: A boolean indicating whether to overwrite the file if it already exists.
    :param separate_files: A boolean indicating whether to save the output grids as separate
        files (e.g., NumPy files) instead of directly in the table. If True, the output
        grids will be saved as separate files and the table will contain references to those
        files.
    :param kwargs: Additional keyword arguments to pass to the function.

    :return: An instance of LearnedSurrogateDataset containing the generated data.
    """
    if separate_files and filename is None:
        raise ValueError("filename must be provided when separate_files is True.")
    if filename is not None:
        filename = Path(filename)
        if filename.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filename}")

    # Generate the data for each set of parameters, either saving it as a file or storing
    # it in memory.
    data_column = []
    for idx in range(len(parameters)):
        output_grid = func(**parameters[idx], **kwargs)

        if separate_files:
            output_filename = f"{filename.parent / filename.stem}_{idx}.npy"
            np.save(output_filename, output_grid)
            data_column.append(output_filename)
        else:
            data_column.append(output_grid)

    # Create a new table with the parameters and the output data.
    expanded_table = parameters.copy()
    expanded_table["output"] = data_column
    data_set = LearnedSurrogateDataset(
        expanded_table,
        output_column="output",
        times=times,
        wavelengths=wavelengths,
    )

    # Save the dataset to a file if a filename was provided. We've already handled the
    # separate_files case above, so we just need to save the table.
    if filename is not None:
        data_set.save(filename, overwrite=overwrite, separate_files=separate_files)

    return data_set
