import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd

class TestTypeIICaching(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures and clear caches before each test."""
        # Import the functions and clear caches
        from redback_surrogates.supernovamodels import (
            typeII_lbol, typeII_photosphere, typeII_spectra,
            _load_lbol_models, _load_photosphere_models, _load_spectra_model,
            clear_typeII_model_cache
        )

        self.typeII_lbol = typeII_lbol
        self.typeII_photosphere = typeII_photosphere
        self.typeII_spectra = typeII_spectra
        self._load_lbol_models = _load_lbol_models
        self._load_photosphere_models = _load_photosphere_models
        self._load_spectra_model = _load_spectra_model
        self.clear_typeII_model_cache = clear_typeII_model_cache

        # Clear all caches before each test
        self.clear_typeII_model_cache()

    def tearDown(self):
        """Clear caches after each test."""
        self.clear_typeII_model_cache()

    @patch('redback_surrogates.supernovamodels.load')
    @patch('redback_surrogates.supernovamodels.keras.saving.load_model')
    def test_lbol_models_cached(self, mock_keras_load, mock_joblib_load):
        """Test that lbol models are loaded only once and cached."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_lbol_model = MagicMock()
        mock_x_scaler = MagicMock()

        mock_joblib_load.side_effect = [mock_lbol_scaler, mock_x_scaler]
        mock_keras_load.return_value = mock_lbol_model

        # Setup mock behaviors
        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_lbol_model.return_value = np.random.rand(1, 200)
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)

        # Call the function multiple times
        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        result1 = self.typeII_lbol(*params)
        result2 = self.typeII_lbol(*params)
        result3 = self.typeII_lbol(*params)

        # Verify models were loaded only once
        self.assertEqual(mock_joblib_load.call_count, 2)  # lbolscaler + xscaler
        self.assertEqual(mock_keras_load.call_count, 1)  # lbol_model

        # Verify results are consistent
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[0], result3[0])

    @patch('redback_surrogates.supernovamodels.load')
    @patch('redback_surrogates.supernovamodels.keras.saving.load_model')
    def test_photosphere_models_cached(self, mock_keras_load, mock_joblib_load):
        """Test that photosphere models are loaded only once and cached."""
        # Setup mocks
        mock_x_scaler = MagicMock()
        mock_temp_scaler = MagicMock()
        mock_rad_scaler = MagicMock()
        mock_temp_model = MagicMock()
        mock_rad_model = MagicMock()

        mock_joblib_load.side_effect = [mock_x_scaler, mock_temp_scaler, mock_rad_scaler]
        mock_keras_load.side_effect = [mock_temp_model, mock_rad_model]

        # Setup mock behaviors
        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_temp_model.return_value = np.random.rand(1, 200)
        mock_rad_model.return_value = np.random.rand(1, 200)
        mock_temp_scaler.inverse_transform.return_value = np.random.rand(200)
        mock_rad_scaler.inverse_transform.return_value = np.random.rand(200)

        # Call the function multiple times
        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        result1 = self.typeII_photosphere(*params)
        result2 = self.typeII_photosphere(*params)
        result3 = self.typeII_photosphere(*params)

        # Verify models were loaded only once
        self.assertEqual(mock_joblib_load.call_count, 3)  # xscaler + tempscaler + radscaler
        self.assertEqual(mock_keras_load.call_count, 2)  # temp_model + rad_model

        # Verify results are consistent
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[0], result3[0])

    @patch('redback_surrogates.supernovamodels.EnhancedSpectralModel.load_model')
    def test_spectra_model_cached(self, mock_load_model):
        """Test that spectra model is loaded only once and cached."""
        # Setup mock
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Setup mock behavior
        mock_spectrum = np.random.rand(50, 50)
        mock_model.predict_spectrum.return_value = mock_spectrum

        # Call the function multiple times
        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        result1 = self.typeII_spectra(*params)
        result2 = self.typeII_spectra(*params)
        result3 = self.typeII_spectra(*params)

        # Verify model was loaded only once
        mock_load_model.assert_called_once()

        # Verify predict_spectrum was called each time
        self.assertEqual(mock_model.predict_spectrum.call_count, 3)

        # Verify results have the same structure
        self.assertTrue(hasattr(result1, 'spectrum'))
        self.assertTrue(hasattr(result2, 'spectrum'))
        self.assertTrue(hasattr(result3, 'spectrum'))

    @patch('redback_surrogates.supernovamodels.load')
    @patch('redback_surrogates.supernovamodels.keras.saving.load_model')
    def test_cache_independence(self, mock_keras_load, mock_joblib_load):
        """Test that different function caches are independent."""
        # Setup mocks for different models
        mock_lbol_scaler = MagicMock()
        mock_lbol_model = MagicMock()
        mock_x_scaler_lbol = MagicMock()
        mock_x_scaler_photo = MagicMock()
        mock_temp_scaler = MagicMock()
        mock_rad_scaler = MagicMock()
        mock_temp_model = MagicMock()
        mock_rad_model = MagicMock()

        # Setup side effects for different calls
        def joblib_side_effect(path):
            if 'lbolscaler' in path:
                return mock_lbol_scaler
            elif 'xscaler' in path:
                # Return different instances for different functions to test independence
                if mock_joblib_load.call_count <= 2:  # First two calls for lbol
                    return mock_x_scaler_lbol
                else:  # Subsequent calls for photosphere
                    return mock_x_scaler_photo
            elif 'temperature' in path:
                return mock_temp_scaler
            elif 'radius' in path:
                return mock_rad_scaler

        def keras_side_effect(path):
            if 'lbol_model' in path:
                return mock_lbol_model
            elif 'temp_model' in path:
                return mock_temp_model
            elif 'radius_model' in path:
                return mock_rad_model

        mock_joblib_load.side_effect = joblib_side_effect
        mock_keras_load.side_effect = keras_side_effect

        # Setup mock behaviors
        mock_x_scaler_lbol.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_x_scaler_photo.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_lbol_model.return_value = np.random.rand(1, 200)
        mock_temp_model.return_value = np.random.rand(1, 200)
        mock_rad_model.return_value = np.random.rand(1, 200)
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)
        mock_temp_scaler.inverse_transform.return_value = np.random.rand(200)
        mock_rad_scaler.inverse_transform.return_value = np.random.rand(200)

        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        # Call lbol function twice
        self.typeII_lbol(*params)
        self.typeII_lbol(*params)

        # Call photosphere function twice
        self.typeII_photosphere(*params)
        self.typeII_photosphere(*params)

        # Verify each cache loaded its models only once
        # lbol: 2 calls (lbolscaler + xscaler), photosphere: 3 calls (xscaler + tempscaler + radscaler)
        self.assertEqual(mock_joblib_load.call_count, 5)
        # lbol: 1 call (lbol_model), photosphere: 2 calls (temp_model + rad_model)
        self.assertEqual(mock_keras_load.call_count, 3)

    def test_cache_info_functionality(self):
        """Test that cache info is properly accessible."""
        # Check that cache functions have cache_info method
        self.assertTrue(hasattr(self._load_lbol_models, 'cache_info'))
        self.assertTrue(hasattr(self._load_photosphere_models, 'cache_info'))
        self.assertTrue(hasattr(self._load_spectra_model, 'cache_info'))

        # Initially, caches should be empty
        lbol_info = self._load_lbol_models.cache_info()
        photo_info = self._load_photosphere_models.cache_info()
        spectra_info = self._load_spectra_model.cache_info()

        self.assertEqual(lbol_info.currsize, 0)
        self.assertEqual(photo_info.currsize, 0)
        self.assertEqual(spectra_info.currsize, 0)

    @patch('redback_surrogates.supernovamodels.load')
    @patch('redback_surrogates.supernovamodels.keras.saving.load_model')
    def test_cache_clear_functionality(self, mock_keras_load, mock_joblib_load):
        """Test that cache clearing works properly."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_lbol_model = MagicMock()
        mock_x_scaler = MagicMock()

        mock_joblib_load.side_effect = [mock_lbol_scaler, mock_x_scaler,
                                        mock_lbol_scaler, mock_x_scaler]  # For after clear
        mock_keras_load.return_value = mock_lbol_model

        # Setup mock behaviors
        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_lbol_model.return_value = np.random.rand(1, 200)
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)

        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        # Call function to populate cache
        self.typeII_lbol(*params)

        # Check cache is populated
        cache_info = self._load_lbol_models.cache_info()
        self.assertEqual(cache_info.currsize, 1)
        self.assertEqual(mock_joblib_load.call_count, 2)
        self.assertEqual(mock_keras_load.call_count, 1)

        # Clear cache
        self.clear_typeII_model_cache()

        # Check cache is cleared
        cache_info = self._load_lbol_models.cache_info()
        self.assertEqual(cache_info.currsize, 0)

        # Call function again - should reload models
        self.typeII_lbol(*params)

        # Check models were loaded again
        self.assertEqual(mock_joblib_load.call_count, 4)  # 2 initial + 2 after clear
        self.assertEqual(mock_keras_load.call_count, 2)  # 1 initial + 1 after clear

    @patch('redback_surrogates.supernovamodels.load')
    @patch('redback_surrogates.supernovamodels.keras.saving.load_model')
    def test_cache_with_different_parameters(self, mock_keras_load, mock_joblib_load):
        """Test that caching works correctly with different input parameters."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_lbol_model = MagicMock()
        mock_x_scaler = MagicMock()

        mock_joblib_load.side_effect = [mock_lbol_scaler, mock_x_scaler]
        mock_keras_load.return_value = mock_lbol_model

        # Setup mock behaviors
        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_lbol_model.return_value = np.random.rand(1, 200)
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)

        # Call with different parameters
        params1 = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)
        params2 = (20.0, 0.1, -2.5, 1.5, 10.0, 2.0)
        params3 = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)  # Same as params1

        self.typeII_lbol(*params1)
        self.typeII_lbol(*params2)
        self.typeII_lbol(*params3)

        # Models should still be loaded only once despite different parameters
        self.assertEqual(mock_joblib_load.call_count, 2)
        self.assertEqual(mock_keras_load.call_count, 1)

        # But the model should be called each time
        self.assertEqual(mock_lbol_model.call_count, 3)

    @patch('redback_surrogates.supernovamodels.EnhancedSpectralModel.load_model')
    def test_spectra_model_exception_handling(self, mock_load_model):
        """Test that exceptions during model loading are handled properly."""
        # Make the first call raise an exception
        mock_load_model.side_effect = [Exception("Model loading failed"), MagicMock()]

        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        # First call should raise exception
        with self.assertRaises(Exception):
            self.typeII_spectra(*params)

        # Cache should not be populated due to exception
        cache_info = self._load_spectra_model.cache_info()
        self.assertEqual(cache_info.currsize, 0)

        # Second call should work and reload the model
        mock_model = MagicMock()
        mock_model.predict_spectrum.return_value = np.random.rand(50, 50)
        mock_load_model.side_effect = None
        mock_load_model.return_value = mock_model

        result = self.typeII_spectra(*params)

        # Should have attempted to load twice
        self.assertEqual(mock_load_model.call_count, 2)
        # Cache should now be populated
        cache_info = self._load_spectra_model.cache_info()
        self.assertEqual(cache_info.currsize, 1)


class TestCachePerformance(unittest.TestCase):
    """Performance tests for caching functionality."""

    def setUp(self):
        from redback_surrogates.supernovamodels import clear_typeII_model_cache
        self.clear_typeII_model_cache = clear_typeII_model_cache
        self.clear_typeII_model_cache()

    def tearDown(self):
        self.clear_typeII_model_cache()

    @patch('redback_surrogates.supernovamodels.load')
    @patch('redback_surrogates.supernovamodels.keras.saving.load_model')
    def test_cache_performance_benefit(self, mock_keras_load, mock_joblib_load):
        """Test that caching provides performance benefits."""
        import time
        from redback_surrogates.supernovamodels import typeII_lbol

        # Setup mocks with artificial delay to simulate loading time
        def slow_load(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            mock = MagicMock()
            if 'lbol' in str(args[0]):
                mock.return_value = np.random.rand(1, 200)
            else:
                mock.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
                mock.inverse_transform.return_value = np.random.rand(200)
            return mock

        mock_joblib_load.side_effect = slow_load
        mock_keras_load.side_effect = slow_load

        params = (15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        # First call (should load models)
        start_time = time.time()
        typeII_lbol(*params)
        first_call_time = time.time() - start_time

        # Second call (should use cached models)
        start_time = time.time()
        typeII_lbol(*params)
        second_call_time = time.time() - start_time

        # Second call should be significantly faster
        self.assertLess(second_call_time, first_call_time)

        # Models should be loaded only once
        self.assertEqual(mock_joblib_load.call_count, 2)
        self.assertEqual(mock_keras_load.call_count, 1)

class TestEnhancedSpectralModel(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Import the classes and functions under test
        from redback_surrogates.supernovamodels import EnhancedSpectralModel
        self.EnhancedSpectralModel = EnhancedSpectralModel

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = self.EnhancedSpectralModel()

        self.assertEqual(model.latent_dim, 64)
        self.assertEqual(model.learning_rate, 1e-3)
        self.assertTrue(model.use_pca)
        self.assertEqual(model.pca_components, 32)
        self.assertIsNone(model.encoder)
        self.assertIsNone(model.decoder)
        self.assertIsNone(model.regressor)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = self.EnhancedSpectralModel(
            latent_dim=128,
            learning_rate=1e-4,
            use_pca=False,
            pca_components=16
        )

        self.assertEqual(model.latent_dim, 128)
        self.assertEqual(model.learning_rate, 1e-4)
        self.assertFalse(model.use_pca)
        self.assertEqual(model.pca_components, 16)

    def test_predict_spectrum_dataframe_input(self):
        """Test predict_spectrum with DataFrame input."""
        model = self.EnhancedSpectralModel()

        # Create mock objects
        mock_param_scaler = MagicMock()
        mock_regressor = MagicMock()
        mock_decoder = MagicMock()
        mock_flux_scaler = MagicMock()

        # Set up model attributes
        model.param_scaler = mock_param_scaler
        model.regressor = mock_regressor
        model.decoder = mock_decoder
        model.flux_scaler = mock_flux_scaler
        model.use_pca = False
        model.latent_scaler = MagicMock()

        # Create test data
        test_params = pd.DataFrame({'param1': [1.0], 'param2': [2.0]})

        # Set up mock returns with more realistic shapes
        mock_param_scaler.transform.return_value = np.array([[0.5, 0.6]])
        mock_regressor.predict.return_value = np.array([[0.1, 0.2, 0.3]])
        model.latent_scaler.inverse_transform.return_value = np.array([[0.2, 0.3, 0.4]])
        mock_decoder.predict.return_value = np.array([[[0.5, 0.6], [0.7, 0.8]]])
        mock_flux_scaler.inverse_transform.return_value = np.array([[1.0, 1.1, 1.2, 1.3]])

        # Call the method
        result = model.predict_spectrum(test_params)

        # Verify calls
        mock_param_scaler.transform.assert_called_once()
        mock_regressor.predict.assert_called_once()
        mock_decoder.predict.assert_called_once()

    def test_predict_spectrum_array_input(self):
        """Test predict_spectrum with array input."""
        model = self.EnhancedSpectralModel()

        # Create mock objects
        mock_param_scaler = MagicMock()
        mock_regressor = MagicMock()
        mock_decoder = MagicMock()
        mock_flux_scaler = MagicMock()

        # Set up model attributes
        model.param_scaler = mock_param_scaler
        model.regressor = mock_regressor
        model.decoder = mock_decoder
        model.flux_scaler = mock_flux_scaler
        model.use_pca = False
        model.latent_scaler = MagicMock()

        # Create test data
        test_params = np.array([1.0, 2.0])

        # Set up mock returns
        mock_param_scaler.transform.return_value = np.array([[0.5, 0.6]])
        mock_regressor.predict.return_value = np.array([[0.1, 0.2, 0.3]])
        model.latent_scaler.inverse_transform.return_value = np.array([[0.2, 0.3, 0.4]])
        mock_decoder.predict.return_value = np.array([[[0.5, 0.6], [0.7, 0.8]]])
        mock_flux_scaler.inverse_transform.return_value = np.array([[1.0, 1.1, 1.2, 1.3]])

        # Call the method
        result = model.predict_spectrum(test_params)

        # Verify the input was converted to 2D array
        mock_param_scaler.transform.assert_called_once()
        call_args = mock_param_scaler.transform.call_args[0][0]
        self.assertEqual(call_args.shape, (1, 2))

    def test_predict_spectrum_with_pca(self):
        """Test predict_spectrum when PCA is enabled."""
        model = self.EnhancedSpectralModel()

        # Create mock objects
        mock_param_scaler = MagicMock()
        mock_regressor = MagicMock()
        mock_decoder = MagicMock()
        mock_flux_scaler = MagicMock()
        mock_pca = MagicMock()
        mock_latent_scaler = MagicMock()

        # Set up model attributes
        model.param_scaler = mock_param_scaler
        model.regressor = mock_regressor
        model.decoder = mock_decoder
        model.flux_scaler = mock_flux_scaler
        model.use_pca = True
        model.pca = mock_pca
        model.latent_scaler = mock_latent_scaler

        # Create test data
        test_params = np.array([[1.0, 2.0]])

        # Set up mock returns
        mock_param_scaler.transform.return_value = np.array([[0.5, 0.6]])
        mock_regressor.predict.return_value = np.array([[0.1, 0.2]])
        mock_latent_scaler.inverse_transform.return_value = np.array([[0.2, 0.3]])
        mock_pca.inverse_transform.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_decoder.predict.return_value = np.array([[[0.5, 0.6], [0.7, 0.8]]])
        mock_flux_scaler.inverse_transform.return_value = np.array([[1.0, 1.1, 1.2, 1.3]])

        # Call the method
        result = model.predict_spectrum(test_params)

        # Verify PCA was called
        mock_pca.inverse_transform.assert_called_once()

    def test_inverse_preprocess_flux_no_scaler(self):
        """Test inverse_preprocess_flux when scaler is None."""
        model = self.EnhancedSpectralModel()
        model.flux_scaler = None

        test_flux = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        with patch('builtins.print') as mock_print:
            result = model.inverse_preprocess_flux(test_flux)
            mock_print.assert_called_once_with("Warning: flux_scaler not found, returning unscaled data")
            np.testing.assert_array_equal(result, test_flux)

    def test_inverse_preprocess_flux_with_scaler(self):
        """Test inverse_preprocess_flux with scaler."""
        model = self.EnhancedSpectralModel()
        mock_flux_scaler = MagicMock()
        model.flux_scaler = mock_flux_scaler

        test_flux = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        mock_flux_scaler.inverse_transform.return_value = np.array([[2.0, 3.0, 4.0, 5.0]])

        result = model.inverse_preprocess_flux(test_flux)

        mock_flux_scaler.inverse_transform.assert_called_once()
        # Check that reshape operations work correctly
        call_args = mock_flux_scaler.inverse_transform.call_args[0][0]
        self.assertEqual(call_args.shape, (1, 4))

    def test_inverse_scale_latent_no_scaler(self):
        """Test inverse_scale_latent when scaler is None."""
        model = self.EnhancedSpectralModel()
        model.latent_scaler = None

        test_latent = np.array([[1.0, 2.0, 3.0]])

        with patch('builtins.print') as mock_print:
            result = model.inverse_scale_latent(test_latent)
            mock_print.assert_called_once_with("Warning: latent_scaler not found, returning unscaled data")
            np.testing.assert_array_equal(result, test_latent)

    def test_inverse_scale_latent_with_scaler(self):
        """Test inverse_scale_latent with scaler."""
        model = self.EnhancedSpectralModel()
        mock_latent_scaler = MagicMock()
        model.latent_scaler = mock_latent_scaler

        test_latent = np.array([[1.0, 2.0, 3.0]])
        expected_result = np.array([[2.0, 3.0, 4.0]])
        mock_latent_scaler.inverse_transform.return_value = expected_result

        result = model.inverse_scale_latent(test_latent)

        mock_latent_scaler.inverse_transform.assert_called_once_with(test_latent)
        np.testing.assert_array_equal(result, expected_result)

    @patch('joblib.load')
    @patch('tensorflow.keras.models.load_model')
    @patch('numpy.load')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_model_complete(self, mock_json_load, mock_file_open, mock_exists,
                                 mock_np_load, mock_tf_load, mock_joblib_load):
        """Test load_model with all components present."""
        # Setup mock config
        mock_config = {
            'latent_dim': 128,
            'use_pca': True,
            'pca_components': 64
        }
        mock_json_load.return_value = mock_config

        # Mock file existence checks
        mock_exists.side_effect = lambda path: True  # All files exist

        # Mock numpy load
        mock_grids = {
            'times': np.array([1, 2, 3]),
            'freqs': np.array([100, 200, 300])
        }
        mock_np_load.return_value = mock_grids

        # Mock keras models
        mock_encoder = MagicMock()
        mock_decoder = MagicMock()
        mock_regressor = MagicMock()
        mock_tf_load.side_effect = [mock_encoder, mock_decoder, mock_regressor]

        # Mock scalers and PCA
        mock_param_scaler = MagicMock()
        mock_flux_scaler = MagicMock()
        mock_latent_scaler = MagicMock()
        mock_pca = MagicMock()
        mock_joblib_load.side_effect = [mock_param_scaler, mock_flux_scaler,
                                        mock_latent_scaler, mock_pca]

        # Call load_model
        model = self.EnhancedSpectralModel.load_model('test_directory')

        # Verify initialization parameters
        self.assertEqual(model.latent_dim, 128)
        self.assertTrue(model.use_pca)
        self.assertEqual(model.pca_components, 64)

        # Verify models are loaded
        self.assertEqual(model.encoder, mock_encoder)
        self.assertEqual(model.decoder, mock_decoder)
        self.assertEqual(model.regressor, mock_regressor)

        # Verify scalers are loaded
        self.assertEqual(model.param_scaler, mock_param_scaler)
        self.assertEqual(model.flux_scaler, mock_flux_scaler)
        self.assertEqual(model.latent_scaler, mock_latent_scaler)
        self.assertEqual(model.pca, mock_pca)

        # Verify grids are loaded
        np.testing.assert_array_equal(model.standard_times, mock_grids['times'])
        np.testing.assert_array_equal(model.standard_freqs, mock_grids['freqs'])


class TestTypeIIFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Import functions under test
        from redback_surrogates.supernovamodels import typeII_lbol, typeII_photosphere, typeII_spectra, \
            clear_typeII_model_cache
        self.typeII_lbol = typeII_lbol
        self.typeII_photosphere = typeII_photosphere
        self.typeII_spectra = typeII_spectra
        self.clear_typeII_model_cache = clear_typeII_model_cache

        # Clear caches before each test
        self.clear_typeII_model_cache()

    def tearDown(self):
        """Clear caches after each test."""
        self.clear_typeII_model_cache()

    @patch('redback_surrogates.supernovamodels._load_lbol_models')
    def test_typeII_lbol_single_params(self, mock_load_lbol_models):
        """Test typeII_lbol with single parameter values."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_x_scaler = MagicMock()
        mock_model = MagicMock()

        mock_load_lbol_models.return_value = (mock_lbol_scaler, mock_model, mock_x_scaler)

        # Setup mock returns - need to match the expected 200 time points
        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_model.return_value = np.random.rand(1, 200)  # 200 time points
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)  # Flattened for single param

        # Test parameters
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = -3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Call function
        tts, lbols = self.typeII_lbol(progenitor, ni_mass, log10_mdot, beta, rcsm, esn)

        # Verify results
        self.assertEqual(len(tts), 200)
        self.assertTrue(np.all(tts >= 0.1))
        self.assertTrue(np.all(tts <= 400))
        self.assertEqual(len(lbols), 200)

        # Verify mock calls
        mock_load_lbol_models.assert_called_once()
        mock_x_scaler.transform.assert_called_once()
        mock_model.assert_called_once()
        mock_lbol_scaler.inverse_transform.assert_called_once()

    @patch('redback_surrogates.supernovamodels._load_lbol_models')
    def test_typeII_lbol_array_params(self, mock_load_lbol_models):
        """Test typeII_lbol with array parameter values."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_x_scaler = MagicMock()
        mock_model = MagicMock()

        mock_load_lbol_models.return_value = (mock_lbol_scaler, mock_model, mock_x_scaler)

        # Setup mock returns for multiple parameters
        mock_x_scaler.transform.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ])
        mock_model.return_value = np.random.rand(2, 200)  # 2 parameter sets, 200 time points
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(2, 200)

        # Test with arrays
        progenitor = np.array([15.0, 20.0])
        ni_mass = np.array([0.05, 0.1])
        log10_mdot = np.array([-3.0, -2.5])
        beta = np.array([1.0, 1.5])
        rcsm = np.array([5.0, 10.0])
        esn = np.array([1.0, 2.0])

        # Call function
        tts, lbols = self.typeII_lbol(progenitor, ni_mass, log10_mdot, beta, rcsm, esn)

        # Verify results
        self.assertEqual(len(tts), 200)
        self.assertEqual(lbols.shape, (2, 200))

    @patch('redback_surrogates.supernovamodels._load_photosphere_models')
    def test_typeII_photosphere_single_params(self, mock_load_photosphere_models):
        """Test typeII_photosphere with single parameter values."""
        # Setup mocks
        mock_x_scaler = MagicMock()
        mock_temp_scaler = MagicMock()
        mock_rad_scaler = MagicMock()
        mock_temp_model = MagicMock()
        mock_rad_model = MagicMock()

        mock_load_photosphere_models.return_value = (
        mock_x_scaler, mock_temp_scaler, mock_rad_scaler, mock_temp_model, mock_rad_model)

        # Setup mock returns
        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_temp_model.return_value = np.random.rand(1, 200)
        mock_rad_model.return_value = np.random.rand(1, 200)
        mock_temp_scaler.inverse_transform.return_value = np.random.rand(200)
        mock_rad_scaler.inverse_transform.return_value = np.random.rand(200)

        # Test parameters
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = -3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Call function
        tts, temp, rad = self.typeII_photosphere(progenitor, ni_mass, log10_mdot, beta, rcsm, esn)

        # Verify results
        self.assertEqual(len(tts), 200)
        self.assertEqual(len(temp), 200)
        self.assertEqual(len(rad), 200)

        # Verify mock calls
        mock_load_photosphere_models.assert_called_once()

    @patch('redback_surrogates.supernovamodels._load_spectra_model')
    @patch('redback_surrogates.supernovamodels.pd.DataFrame')
    def test_typeII_spectra(self, mock_dataframe, mock_load_spectra_model):
        """Test typeII_spectra function."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_spectra_model.return_value = mock_model

        # Mock DataFrame creation
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df

        # Mock spectrum prediction
        mock_spectrum_log = np.random.rand(50, 50)  # 50x50 spectrum
        mock_model.predict_spectrum.return_value = mock_spectrum_log

        # Test parameters
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = 3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Call function
        result = self.typeII_spectra(progenitor, ni_mass, log10_mdot, beta, rcsm, esn)

        # Verify result structure
        self.assertTrue(hasattr(result, 'spectrum'))
        self.assertTrue(hasattr(result, 'frequency'))
        self.assertTrue(hasattr(result, 'time'))

        # Verify units
        self.assertTrue(hasattr(result.spectrum, 'unit'))
        self.assertTrue(hasattr(result.frequency, 'unit'))
        self.assertTrue(hasattr(result.time, 'unit'))

        # Verify DataFrame was created with correct parameters
        mock_dataframe.assert_called_once()
        call_args = mock_dataframe.call_args[0][0][0]  # First argument, first element
        self.assertEqual(call_args['progenitor'], progenitor)
        self.assertEqual(call_args['ni_mass'], ni_mass)
        self.assertEqual(call_args['mass_loss'], log10_mdot)
        self.assertEqual(call_args['beta'], beta)
        self.assertEqual(call_args['csm_radius'], rcsm * 1e14)  # Should be scaled
        self.assertEqual(call_args['explosion_energy'], esn)

    @patch('redback_surrogates.supernovamodels._load_lbol_models')
    def test_rcsm_scaling(self, mock_load_lbol_models):
        """Test that rcsm parameter is properly scaled in all functions."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_model = MagicMock()
        mock_x_scaler = MagicMock()

        mock_load_lbol_models.return_value = (mock_lbol_scaler, mock_model, mock_x_scaler)

        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_model.return_value = np.random.rand(1, 200)
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)

        # Call with rcsm = 5.0, should become 5e14 internally
        self.typeII_lbol(15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        # Check that the scaled value was used
        call_args = mock_x_scaler.transform.call_args[0][0]
        self.assertEqual(call_args[0][4], 5e14)  # rcsm should be scaled

    @patch('redback_surrogates.supernovamodels._load_lbol_models')
    def test_log10_mdot_abs_value(self, mock_load_lbol_models):
        """Test that log10_mdot is converted to absolute value."""
        # Setup mocks
        mock_lbol_scaler = MagicMock()
        mock_model = MagicMock()
        mock_x_scaler = MagicMock()

        mock_load_lbol_models.return_value = (mock_lbol_scaler, mock_model, mock_x_scaler)

        mock_x_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_model.return_value = np.random.rand(1, 200)
        mock_lbol_scaler.inverse_transform.return_value = np.random.rand(200)

        # Call with negative log10_mdot
        self.typeII_lbol(15.0, 0.05, -3.0, 1.0, 5.0, 1.0)

        # Check that absolute value was used
        call_args = mock_x_scaler.transform.call_args[0][0]
        self.assertEqual(call_args[0][2], 3.0)  # Should be abs(-3.0) = 3.0



