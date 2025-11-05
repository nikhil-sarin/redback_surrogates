"""A class for general surrogate models learned from data."""

import json
import numpy as np
import onnx
import onnxruntime as rt
import re
import types

from pathlib import Path


class LearnedSurrogateModel:
    """A general surrogate model class.

    The class will have a dynamic method `predict` created at initialization time
    that acts like a redback model with parameters matching the model inputs.
    """

    def __init__(
        self,
        model,
        *,
        times=None,
        wavelengths=None,
        metadata=None,
    ):
        """Initialize the surrogate model.

        :param model: The underlying learned model
        :param times: List of time points
        :param wavelengths: List of wavelength points
        :param metadata: Additional metadata dictionary.
        """
        self._model = model
        if model is None:
            raise ValueError("Model must be provided.")

        # Load the parameter names from the model inputs.
        self.param_names = [p.name for p in model.graph.input]

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

        # Create the ONNX runtime session for inference.
        self._ort_session = rt.InferenceSession(
            self._model.SerializeToString(),
            providers=rt.get_available_providers(),
        )

        # Determine the information about the output. We only support one output value
        # the grid itself.
        output0 = self._ort_session.get_outputs()[0]
        self.output_name = output0.name
        if output0.shape[1] != len(self.times) or output0.shape[2] != len(
            self.wavelengths
        ):
            raise ValueError(
                f"Shape of output {output0.shape} does not match the times ({len(self.times)}) "
                f" and wavelengths ({len(self.wavelengths)})."
            )

        # Create a dynamic method that takes parameters matching param_names
        self._create_dynamic_method()

    def _create_dynamic_method(self):
        """Create a dynamic method with parameters matching param_names."""
        # Check that the parameter names are safe to use in an exec statement.
        # We do this by restricting to valid Python identifiers.
        identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for name in self.param_names:
            if not identifier_re.match(name):
                raise ValueError(
                    f"Parameter name '{name}' is invalid. Parameter names can "
                    "only contain alphanumeric characters and underscores."
                )

        # Use the parameter list to create the function signature and
        # internal dictionary.
        param_str = ", ".join(["self"] + self.param_names)
        param_dict_str = (
            "{" + ", ".join([f"'{name}': {name}" for name in self.param_names]) + "}"
        )
        doc_str = (
            "Dynamically generated method to predict spectra.\n\n"
            + "\n".join([f":param {name}: float" for name in self.param_names])
            + "\n:return: Predicted spectra array of shape (1, len(times), len(wavelengths))\n"
        )

        # Build the complete function string. We have already checked that
        # the parameter names are safe.
        function_code = (
            f"def _dynamic_predict({param_str}):\n"
            f"    '''{doc_str}    '''\n"
            f"    param_dict = {param_dict_str}\n"
            f"    return self.predict_spectra(param_dict)\n"
        )

        # Execute the function definition and bind it to this instance.
        # Note that we can only do exec safely here because we checked
        # the parameter names earlier to ensure they are safe.
        local_namespace = {"self": self}
        exec(function_code, globals(), local_namespace)

        # Bind the function as a method to this instance
        self.predict = types.MethodType(local_namespace["_dynamic_predict"], self)

    @property
    def times(self):
        """List of time points."""
        return self._metadata.get("times", None)

    @property
    def wavelengths(self):
        """List of wavelength points."""
        return self._metadata.get("wavelengths", None)

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

    def predict_spectra(self, params):
        """Compute the spectral energy distribution for given parameters.

        :param params: dict mapping parameter name to its value
        """
        inputs = {
            key: np.array(params[key], dtype=np.float32) for key in self.param_names
        }
        output = self._ort_session.run([self.output_name], inputs)
        return output
