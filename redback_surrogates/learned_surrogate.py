"""A class for general surrogate models learned from data."""

import inspect
import json
import numpy as np
import onnx
import onnxruntime as rt
import re

from pathlib import Path


def assert_safe_param_names(param_names):
    """Check that a list of parameter names are safe to use in dynamic method creation.

    :param param_names: The original parameter names as a list of strings.
    """
    identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for name in param_names:
        if not isinstance(name, str):
            raise ValueError(
                f"Parameter name '{name}' is not a string. Parameter names must be strings."
            )
        if not identifier_re.match(name):
            raise ValueError(
                f"Parameter name '{name}' is invalid. Parameter names can "
                "only contain alphanumeric characters and underscores."
            )


class LearnedSurrogateModel:
    """A general surrogate model class that produces spectral energy distributions
    (f_lambda in units of erg/s/Hz) on a time and wavelength grid given a set of
    input parameters. It wraps an underlying learned model in ONNX format to support
    inference from multiple learning frameworks, including tensorflow, pytorch, and
    sklearn.
    """

    def __init__(
        self,
        model,
        *,
        times=None,
        wavelengths=None,
        param_names=None,
        metadata=None,
    ):
        """Initialize the surrogate model.

        :param model: The underlying learned model
        :param times: List of time points
        :param wavelengths: List of wavelength points
        :param param_names: List of parameter names (optional)
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
            raise ValueError(
                "Times must be provided either in metadata or as argument."
            )

        if wavelengths is not None:
            self._metadata["wavelengths"] = list(wavelengths)
        elif "wavelengths" not in self._metadata:
            raise ValueError(
                "Wavelengths must be provided either in metadata or as argument."
            )

        # If given the parameter names, use those. Otherwise try to pull them from
        # the metadata or model itself.
        if param_names is not None:
            self.param_names = list(param_names)
        elif "param_names" in self._metadata:
            self.param_names = self._metadata["param_names"]
        else:
            self.param_names = [p.name for p in model.graph.input]
        assert_safe_param_names(self.param_names)
        self._metadata["param_names"] = self.param_names

        # Determine if we need to collapse the parameters into a single input array
        # such as used for sklearn models.
        self._collapse_parameters = len(self.param_names) > 1 and len(model.graph.input) == 1

        # Create the ONNX runtime session for inference.
        self._ort_session = rt.InferenceSession(
            self._model.SerializeToString(),
            providers=rt.get_available_providers(),
        )

        # Determine the information about the output. We only support one output value
        # the grid itself. Check that we can reshape the output to match the expected grid.
        num_wave = len(self.wavelengths)
        num_time = len(self.times)
        self.output_shape = (num_time, num_wave)

        output0 = self._ort_session.get_outputs()[0]
        self.output_name = output0.name
        if (
            not (len(output0.shape) == 2 and output0.shape[1] == num_wave * num_time) and
            not (len(output0.shape) == 2 and (output0.shape[0] == num_time and output0.shape[1] == num_wave)) and
            not (len(output0.shape) == 3 and (output0.shape[1] == num_time and output0.shape[2] == num_wave))
        ):
            raise ValueError(
                f"Shape of output {output0.shape} does not match the times ({len(self.times)}) "
                f" and wavelengths ({len(self.wavelengths)})."
            )

    def __repr__(self):
        """Create a human-readable description of the model and its parameters as pulled from the metadata."""
        description = (
            f"LearnedSurrogateModel with {len(self.param_names)} parameters:\n"
            f"    Times Dimension: {len(self.times)} steps [{self.times[0]}, {self.times[-1]}]\n"
            f"    Wavelengths Dimension: {len(self.wavelengths)} steps "
            f"[{self.wavelengths[0]}, {self.wavelengths[-1]}]\n\n"
            f"Parameters:\n"
        )

        param_info = self._metadata.get("parameter_info", {})
        for name in self.param_names:
            info = param_info.get(name, "No description available")
            description += (f" - {name}: {info}\n")
        return description

    @property
    def times(self):
        """List of time points."""
        return self._metadata.get("times", None)

    @property
    def wavelengths(self):
        """List of wavelength points."""
        return self._metadata.get("wavelengths", None)

    def add_parameter_info(self, param_name, info):
        """Add information about a parameter to the metadata.

        :param param_name: The name of the parameter
        :param info: The information string about the parameter
        """
        if param_name not in self.param_names:
            raise ValueError(f"'{param_name}' is not in the model parameters.")
        if "parameter_info" not in self._metadata:
            self._metadata["parameter_info"] = {}
        self._metadata["parameter_info"][param_name] = info

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
    def from_pytorch_model(cls, pytorch_model, times, wavelengths, param_info=None):
        """Create a LearnedSurrogateModel from a PyTorch model.

        :param pytorch_model: The PyTorch model to convert
        :param times: List of time points
        :param wavelengths: List of wavelength points
        :param param_info: Optional dictionary mapping parameter names to descriptions.

        :return: An instance of LearnedSurrogateModel
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to use from_pytorch_model.")

        if param_info is None:
            # If no parameter info is provided, try to get the parameter names
            # from the model's forward method.
            args = inspect.getfullargspec(pytorch_model.forward).args
            param_info = {name: "No description available" for name in args if name != "self"}

        # We create example input that has one float for each parameter.
        example_input = tuple(torch.tensor(1.1) for _ in param_info)

        # Compile the PyTorch model to ONNX format using torch.onnx.export, build
        # the surrogate model, and add parameter info.
        onnx_program = torch.onnx.export(pytorch_model, example_input, dynamo=True)
        surrogate_model = LearnedSurrogateModel(
            onnx_program.model_proto,
            times=times,
            wavelengths=wavelengths,
        )
        for name, info in param_info.items():
            surrogate_model.add_parameter_info(name, info)

        return surrogate_model

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

    def predict_spectra_grid(self, **params):
        """Compute the rest frame spectral energy distribution for the given parameters
        in units of erg/s/Hz.

        :param params: dict mapping parameter name to its value

        :return: The predicted spectral energy distribution grid in f_lambda
            and units of erg/s/Hz.
        """
        if self._collapse_parameters:
            # Collapse all parameters into a single input array
            input_array = np.column_stack([np.array(params[name]) for name in self.param_names])
            inputs = {self._ort_session.get_inputs()[0].name: input_array}
        else:
            inputs = {key: np.array(params[key]) for key in self.param_names}

        # Unflatten the output if needed.
        output = self._ort_session.run([self.output_name], inputs)[0]
        output = output.reshape(self.output_shape)
        return output
