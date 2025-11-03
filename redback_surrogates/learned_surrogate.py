"""A class for general surrogate models learned from data."""

import json
import onnx
import onnxruntime as rt

from pathlib import Path


class LearnedSurrogateModel:
    """A general surrogate model class."""

    def __init__(
            self,
            model,
            *,
            times=None,
            wavelengths=None,
            parameter_names=None,
            metadata=None,
        ):
        """Initialize the surrogate model.

        :param model: The underlying learned model
        :param times: List of time points
        :param wavelengths: List of wavelength points
        :param parameter_names: List of parameter names
        :param metadata: Additional metadata dictionary.
        """
        self._model = model
        if model is None:
            raise ValueError("Model must be provided.")

        # We store all the metadata in a single dictionary so that we can keep it in one
        # place as we convert back and forth to ONNX files.
        self._metadata = metadata if metadata is not None else {}
        if times is not None:
            self._metadata["times"] = list(times)
        elif "times" not in self._metadata:
            raise ValueError("Times must be provided either in metadata or as argument.")

        if wavelengths is not None:
            self._metadata["wavelengths"] = list(wavelengths)
        elif "wavelengths" not in self._metadata:
            raise ValueError("Wavelengths must be provided either in metadata or as argument.")

        if parameter_names is not None:
            self._metadata["parameter_names"] = list(parameter_names)
        elif "parameter_names" not in self._metadata:
            raise ValueError("Parameter names must be provided either in metadata or as argument.")

        # Determine the output shape if we do not already have it.
        if "output_shape" not in self._metadata:
            self._metadata["output_shape"] = (len(self.times), len(self.wavelengths))

        # Create the ONNX runtime session for inference.
        self._ort_session = rt.InferenceSession(
            self._model.SerializeToString(),
            providers=rt.get_available_providers(),
        )

    @property
    def times(self):
        """List of time points."""
        return self._metadata.get("times", None)

    @property
    def wavelengths(self):
        """List of wavelength points."""
        return self._metadata.get("wavelengths", None)

    @property
    def parameter_names(self):
        """List of parameter names."""
        return self._metadata.get("parameter_names", None)

    @property
    def output_shape(self):
        """Shape of the output spectra (times, wavelengths)."""
        return self._metadata.get("output_shape", None)

    def __call__(self, parameters):
        """Compute the spectral energy distribution for given parameters.

        :param parameters: DataFrame or of physical parameters
        """
        self.predict_spectra(parameters)

    def tensor_from_params(self, **parameters):
        """Convert parameters to input tensor for the model.

        :param parameters: DataFrame or of physical parameters

        :return: Input tensor for the model
        """
        torch.tensor

    @staticmethod
    def _onnx_metadata_to_dict(model):
        """Convert ONNX model metadata to a dictionary.

        :param model: The ONNX model

        :return: Dictionary of metadata
        """
        metadata = {}
        for prop in model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)
        return metadata

    @classmethod
    def from_onnx_file(cls, filepath):
        """Load saved model from an Onnx file.

        :param filepath: Path to the model file.

        :return: An instance of LearnedSurrogateModel
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file {filepath} does not exist.")

        # Load the ONNX model and parse the meta data. This includes all the expected
        # data, such as times, wavelengths, and parameter names.
        model = onnx.load(filepath)
        metadata = LearnedSurrogateModel._onnx_metadata_to_dict(model)
        return cls(model, metadata=metadata)

    def to_onnx_file(self, filepath, overwrite=False):
        """Save the model to an Onnx file.

        :param filepath: Path to which to save the model file
        :param overwrite: Whether to overwrite the file if it exists
        """
        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"Model file {filepath} already exists.")

        # Start with the existing model's metadata and update it with the
        # this objects metadata dictionary.
        new_metadata = LearnedSurrogateModel._onnx_metadata_to_dict(self._model)
        for key, value in self._metadata.items():
            new_metadata[key] = value

        # Clear existing metadata and set new metadata.
        self._model.metadata_props.clear()
        for key, value in new_metadata.items():
            meta_prop = self._model.metadata_props.add()
            meta_prop.key = key
            meta_prop.value = json.dumps(value)

        onnx.save(self._model, filepath)

    @classmethod
    def fit_from_data(cls, parameters, spectra, training_config):
        """Fit the surrogate model from training data.

        :param parameters: DataFrame or of physical parameters
        :param spectra: Array of spectral data
        :param training_config: Configuration dictionary for training
        """
        pass

    def predict_spectra(self, parameters):
        """Compute the spectral energy distribution for given parameters.

        :param parameters: DataFrame or of physical parameters
        """

        pass
