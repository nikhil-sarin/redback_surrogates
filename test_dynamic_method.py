"""Test script for the dynamic method creation."""

import numpy as np
from unittest.mock import Mock
import sys
import os

# Add the redback_surrogates directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'redback_surrogates'))

from learned_surrogate import LearnedSurrogateModel


def test_dynamic_method():
    """Test that the dynamic method creation works correctly."""
    # Create a mock ONNX model with input parameters
    mock_model = Mock()
    mock_model.graph.input = [
        Mock(name='mass'),
        Mock(name='velocity'),
        Mock(name='temperature')
    ]
    mock_model.SerializeToString.return_value = b'mock_model_data'
    
    # Mock the ONNX runtime session
    mock_session = Mock()
    mock_output = Mock()
    mock_output.name = 'output'
    mock_output.shape = [1, 10, 5]  # batch_size=1, times=10, wavelengths=5
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.random.rand(1, 10, 5)]
    
    # Mock the rt.InferenceSession to return our mock session
    import onnxruntime as rt
    original_inference_session = rt.InferenceSession
    rt.InferenceSession = Mock(return_value=mock_session)
    
    try:
        # Create the surrogate model
        model = LearnedSurrogateModel(
            mock_model,
            times=list(range(10)),
            wavelengths=list(range(5))
        )
        
        # Test that the dynamic method was created with the correct signature
        assert hasattr(model, 'predict_grid')
        
        # Test calling the dynamic method
        result = model.predict_grid(mass=1.4, velocity=0.1, temperature=5000)
        
        # Verify that the mock session was called
        mock_session.run.assert_called_once()
        
        print("✓ Dynamic method creation test passed!")
        print(f"✓ Method signature includes parameters: {model.param_names}")
        print(f"✓ Method call successful, returned shape: {np.array(result).shape}")
        
    finally:
        # Restore the original InferenceSession
        rt.InferenceSession = original_inference_session


if __name__ == "__main__":
    test_dynamic_method()