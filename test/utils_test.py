import unittest
from unittest.mock import patch, MagicMock
import logging
from bilby.core.prior import PriorDict


class TestGetPriors(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the function to test
        from redback_surrogates.utils import get_priors
        self.get_priors = get_priors

    @patch('os.path.dirname')  # Patch os directly
    @patch('os.path.join')     # Patch os directly
    @patch.object(PriorDict, 'from_file')
    def test_get_priors_success(self, mock_from_file, mock_join, mock_dirname):
        """Test successful loading of prior file."""
        # Setup mocks
        mock_dirname.return_value = '/fake/path'
        mock_join.return_value = '/fake/path/priors/test_model.prior'
        mock_from_file.return_value = None  # from_file modifies the object in place

        # Call function
        result = self.get_priors('test_model')

        # Assertions
        self.assertIsInstance(result, PriorDict)
        mock_dirname.assert_called_once()
        mock_join.assert_called_once_with('/fake/path', 'priors', 'test_model.prior')
        mock_from_file.assert_called_once_with('/fake/path/priors/test_model.prior')

    @patch('os.path.dirname')
    @patch('os.path.join')
    @patch.object(PriorDict, 'from_file')
    @patch('logging.getLogger')  # Patch logging directly
    def test_get_priors_file_not_found(self, mock_get_logger, mock_from_file,
                                       mock_join, mock_dirname):
        """Test behavior when prior file is not found."""
        # Setup mocks
        mock_dirname.return_value = '/fake/path'
        mock_join.return_value = '/fake/path/priors/nonexistent_model.prior'
        mock_from_file.side_effect = FileNotFoundError("File not found")

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Call function
        result = self.get_priors('nonexistent_model')

        # Assertions
        self.assertIsInstance(result, PriorDict)
        mock_from_file.assert_called_once_with('/fake/path/priors/nonexistent_model.prior')
        mock_get_logger.assert_called_once_with('redback_surrogates.utils')

        # Check that warning and info messages were logged
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_called_once_with('Returning Empty PriorDict.')

        # Check warning message content
        warning_call_args = mock_logger.warning.call_args[0][0]
        self.assertIn('No prior file found for model nonexistent_model', warning_call_args)
        self.assertIn('Perhaps you also want to set up the prior for the base model?', warning_call_args)

    @patch('logging.basicConfig')  # Patch logging directly
    @patch('logging.getLogger')    # Patch logging directly
    def test_logging_configuration(self, mock_get_logger, mock_basic_config):
        """Test that logging is configured correctly."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        with patch.object(PriorDict, 'from_file', side_effect=FileNotFoundError):
            self.get_priors('test_model')

        # Check logging configuration
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        mock_get_logger.assert_called_once_with('redback_surrogates.utils')

    @patch('os.path.dirname')
    @patch('os.path.join')
    def test_file_path_construction(self, mock_join, mock_dirname):
        """Test that file path is constructed correctly."""
        mock_dirname.return_value = '/module/path'
        mock_join.return_value = '/module/path/priors/custom_model.prior'

        with patch.object(PriorDict, 'from_file'):
            self.get_priors('custom_model')

        # Verify path construction - need to check what __file__ resolves to
        mock_dirname.assert_called_once()
        mock_join.assert_called_once_with('/module/path', 'priors', 'custom_model.prior')

    def test_return_type(self):
        """Test that function always returns a PriorDict instance."""
        with patch.object(PriorDict, 'from_file'):
            result = self.get_priors('test_model')
            self.assertIsInstance(result, PriorDict)

        with patch.object(PriorDict, 'from_file', side_effect=FileNotFoundError):
            result = self.get_priors('test_model')
            self.assertIsInstance(result, PriorDict)

    @patch('os.path.dirname')
    @patch('os.path.join')
    @patch.object(PriorDict, 'from_file')
    def test_different_model_names(self, mock_from_file, mock_join, mock_dirname):
        """Test function with different model names."""
        mock_dirname.return_value = '/fake/path'

        test_models = ['model1', 'model_with_underscores', 'model-with-dashes', '123numeric']

        for model in test_models:
            expected_filename = f'{model}.prior'
            mock_join.return_value = f'/fake/path/priors/{expected_filename}'

            result = self.get_priors(model)

            self.assertIsInstance(result, PriorDict)
            # Use assert_any_call since we're calling it multiple times in the loop
            mock_join.assert_any_call('/fake/path', 'priors', expected_filename)

    @patch('os.path.dirname')
    @patch('os.path.join')
    @patch.object(PriorDict, 'from_file')
    @patch('logging.getLogger')
    def test_other_exceptions_not_caught(self, mock_get_logger, mock_from_file,
                                         mock_join, mock_dirname):
        """Test that exceptions other than FileNotFoundError are not caught."""
        mock_dirname.return_value = '/fake/path'
        mock_join.return_value = '/fake/path/priors/test_model.prior'
        mock_from_file.side_effect = PermissionError("Permission denied")

        # Should raise PermissionError, not be caught
        with self.assertRaises(PermissionError):
            self.get_priors('test_model')

    @patch('os.path.dirname')
    @patch('os.path.join')
    @patch.object(PriorDict, 'from_file')
    @patch('logging.getLogger')
    def test_empty_model_name(self, mock_get_logger, mock_from_file,
                              mock_join, mock_dirname):
        """Test function with empty model name."""
        mock_dirname.return_value = '/fake/path'
        mock_join.return_value = '/fake/path/priors/.prior'
        mock_from_file.side_effect = FileNotFoundError()

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        result = self.get_priors('')

        self.assertIsInstance(result, PriorDict)
        mock_join.assert_called_once_with('/fake/path', 'priors', '.prior')