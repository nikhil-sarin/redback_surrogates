import os
import tempfile
import zipfile
import hashlib
import shutil
import unittest
from unittest import mock
from pathlib import Path
from io import BytesIO

# Import the module to test
from redback_surrogates.data_management import (
    get_surrogate_data_dir,
    calculate_md5,
    download_file,
    extract_zip,
    download_surrogate_data,
    get_surrogate_file_path,
    list_surrogate_files,
    get_md5_hash,
    SURROGATE_FILES
)


class MockResponse:
    """Mock class for requests.get responses."""

    def __init__(self, content=b'', status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {'content-length': str(len(content))}
        self._content_consumed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")

    def iter_content(self, chunk_size=1):
        self._content_consumed = True
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class TestDownloadUtility(unittest.TestCase):
    """Test cases for the surrogate data download utility."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Create a test ZIP file
        self.test_zip_content = {
            'test_file1.txt': b'This is test file 1',
            'test_file2.txt': b'This is test file 2',
            'subfolder/test_file3.txt': b'This is test file 3 in a subfolder'
        }

        self.test_zip_path = Path(self.test_dir) / 'test.zip'
        self._create_test_zip(self.test_zip_path, self.test_zip_content)

        # Create a file that simulates the surrogate data file
        for filename in SURROGATE_FILES.keys():
            file_path = Path(self.test_dir) / filename
            with open(file_path, 'wb') as f:
                f.write(b'Test surrogate data')

        # Mock the get_surrogate_data_dir function to return our test directory
        self.data_dir_patcher = mock.patch(
            'redback_surrogates.data_management.get_surrogate_data_dir',
            return_value=Path(self.test_dir)
        )
        self.mock_data_dir = self.data_dir_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        # Stop all patches
        self.data_dir_patcher.stop()

        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def _create_test_zip(self, zip_path, content_dict):
        """Create a test ZIP file with the given content."""
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_path, content in content_dict.items():
                zip_file.writestr(file_path, content)

    def test_get_surrogate_data_dir(self):
        """Test get_surrogate_data_dir creates directory if needed."""
        # Restore the original function for this test
        self.data_dir_patcher.stop()

        try:
            # Mock __file__ to point to our temp directory
            with mock.patch('redback_surrogates.data_management.__file__',
                            new=os.path.join(self.test_dir, 'data_management.py')):
                # Remove the directory if it exists
                data_dir = Path(self.test_dir) / 'surrogate_data'
                if data_dir.exists():
                    shutil.rmtree(data_dir)

                # Call the function
                result = get_surrogate_data_dir()

                # Check results
                self.assertTrue(data_dir.exists())
                self.assertEqual(result, data_dir)
        finally:
            # Restart the patcher
            self.mock_data_dir = self.data_dir_patcher.start()

    def test_calculate_md5(self):
        """Test MD5 calculation."""
        # Create a test file with known content
        test_file = Path(self.test_dir) / 'test_md5.txt'
        test_content = b'This is a test file for MD5 calculation'
        with open(test_file, 'wb') as f:
            f.write(test_content)

        # Calculate expected MD5
        expected_md5 = hashlib.md5(test_content).hexdigest()

        # Call the function
        result = calculate_md5(test_file)

        # Check results
        self.assertEqual(result, expected_md5)

    @mock.patch('redback_surrogates.data_management.requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        # Setup mock
        test_content = b'This is test content for download'
        mock_get.return_value = MockResponse(test_content)

        # Set up target path
        target_path = Path(self.test_dir) / 'downloaded_file.txt'

        # Call the function
        result = download_file('http://test.url', target_path)

        # Check results
        self.assertTrue(result)
        self.assertTrue(target_path.exists())
        with open(target_path, 'rb') as f:
            self.assertEqual(f.read(), test_content)

    @mock.patch('redback_surrogates.data_management.requests.get')
    def test_download_file_failure(self, mock_get):
        """Test file download failure."""
        # Setup mock to raise an exception
        mock_get.return_value = MockResponse(status_code=404)

        # Set up target path
        target_path = Path(self.test_dir) / 'should_not_exist.txt'

        # Call the function
        result = download_file('http://test.url', target_path)

        # Check results
        self.assertFalse(result)
        self.assertFalse(target_path.exists())

    def test_extract_zip(self):
        """Test ZIP extraction."""
        # Set up extraction directory
        extract_dir = Path(self.test_dir) / 'extracted'

        # Call the function
        result = extract_zip(self.test_zip_path, extract_dir)

        # Check results
        self.assertTrue(result)
        self.assertTrue(extract_dir.exists())

        # Check all files were extracted
        for file_path, content in self.test_zip_content.items():
            extracted_file = extract_dir / file_path
            self.assertTrue(extracted_file.exists())
            with open(extracted_file, 'rb') as f:
                self.assertEqual(f.read(), content)

    def test_extract_zip_failure(self):
        """Test ZIP extraction failure."""
        # Create an invalid ZIP file
        invalid_zip = Path(self.test_dir) / 'invalid.zip'
        with open(invalid_zip, 'wb') as f:
            f.write(b'This is not a valid ZIP file')

        # Set up extraction directory
        extract_dir = Path(self.test_dir) / 'should_not_exist'

        # Call the function
        result = extract_zip(invalid_zip, extract_dir)

        # Check results
        self.assertFalse(result)

    @mock.patch('redback_surrogates.data_management.download_file')
    @mock.patch('redback_surrogates.data_management.extract_zip')
    @mock.patch('redback_surrogates.data_management.calculate_md5')
    def test_download_surrogate_data_success(self, mock_md5, mock_extract, mock_download):
        """Test successful surrogate data download and extraction."""
        # Setup mocks
        mock_download.return_value = True
        mock_extract.return_value = True
        mock_md5.return_value = SURROGATE_FILES['TypeII_surrogate_Sarin+2025.zip']['md5']

        # Call the function
        result = download_surrogate_data(force_download=True)

        # Check results
        self.assertIsNotNone(result)
        self.assertEqual(result, Path(self.test_dir))
        mock_download.assert_called_once()
        mock_extract.assert_called_once()

    @mock.patch('redback_surrogates.data_management.download_file')
    @mock.patch('redback_surrogates.data_management.calculate_md5')
    def test_download_surrogate_data_download_failure(self, mock_md5, mock_download):
        """Test surrogate data download failure."""
        # Setup mock
        mock_download.return_value = False
        mock_md5.return_value = SURROGATE_FILES['TypeII_surrogate_Sarin+2025.zip']['md5']

        # Call the function
        result = download_surrogate_data(force_download=True)

        # Check results
        self.assertIsNone(result)
        mock_download.assert_called_once()

    @mock.patch('redback_surrogates.data_management.download_file')
    @mock.patch('redback_surrogates.data_management.extract_zip')
    @mock.patch('redback_surrogates.data_management.calculate_md5')
    def test_download_surrogate_data_extract_failure(self, mock_md5, mock_extract, mock_download):
        """Test surrogate data extraction failure."""
        # Setup mocks
        mock_download.return_value = True
        mock_extract.return_value = False
        mock_md5.return_value = SURROGATE_FILES['TypeII_surrogate_Sarin+2025.zip']['md5']

        # Call the function
        result = download_surrogate_data(force_download=True)

        # Check results
        self.assertIsNone(result)
        mock_download.assert_called_once()
        mock_extract.assert_called_once()

    def test_get_surrogate_file_path_exists(self):
        """Test getting path to existing surrogate file."""
        # Create a test file
        test_file = Path(self.test_dir) / 'test_surrogate_file.txt'
        with open(test_file, 'w') as f:
            f.write('Test content')

        # Call the function
        result = get_surrogate_file_path('test_surrogate_file.txt')

        # Check results
        self.assertEqual(result, test_file)

    @mock.patch('redback_surrogates.data_management.download_surrogate_data')
    def test_get_surrogate_file_path_download(self, mock_download):
        """Test getting path triggers download if file doesn't exist."""

        # Setup mock to create the file when called
        def side_effect(*args, **kwargs):
            test_file = Path(self.test_dir) / 'test_new_file.txt'
            with open(test_file, 'w') as f:
                f.write('Downloaded content')
            return Path(self.test_dir)

        mock_download.side_effect = side_effect

        # Call the function
        result = get_surrogate_file_path('test_new_file.txt')

        # Check results
        self.assertIsNotNone(result)
        self.assertEqual(result, Path(self.test_dir) / 'test_new_file.txt')
        mock_download.assert_called_once()

    @mock.patch('redback_surrogates.data_management.download_surrogate_data')
    def test_get_surrogate_file_path_not_found(self, mock_download):
        """Test getting path returns None if file not found after download."""
        # Setup mock to not create the file
        mock_download.return_value = Path(self.test_dir)

        # Call the function
        result = get_surrogate_file_path('nonexistent_file.txt')

        # Check results
        self.assertIsNone(result)
        mock_download.assert_called_once()

    def test_list_surrogate_files(self):
        """Test listing surrogate files."""
        # Create some test files
        files = ['file1.txt', 'file2.txt', 'subdir/file3.txt']
        for file_path in files:
            full_path = Path(self.test_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write('Test content')

        # Call the function
        result = list_surrogate_files()

        # Check results - should find at least 3 files (plus the ones we created in setUp)
        self.assertTrue(len(result) >= 3)
        self.assertTrue(all(p.is_file() for p in result))

        # Test with subdir
        result_subdir = list_surrogate_files('subdir')
        self.assertEqual(len(result_subdir), 1)
        self.assertEqual(result_subdir[0].name, 'file3.txt')

    def test_get_md5_hash(self):
        """Test getting MD5 hash of a file."""
        # Create a test file
        test_file = Path(self.test_dir) / 'test_hash.txt'
        test_content = b'Test content for hash calculation'
        with open(test_file, 'wb') as f:
            f.write(test_content)

        # Calculate expected MD5
        expected_md5 = hashlib.md5(test_content).hexdigest()

        # Call the function
        with mock.patch('builtins.print') as mock_print:
            result = get_md5_hash(str(test_file))

        # Check results
        self.assertEqual(result, expected_md5)
        mock_print.assert_called_once()

    def test_get_md5_hash_nonexistent_file(self):
        """Test getting MD5 hash of a nonexistent file."""
        # Call the function
        with mock.patch('builtins.print') as mock_print:
            result = get_md5_hash('nonexistent_file.txt')

        # Check results
        self.assertEqual(result, "")
        mock_print.assert_called_once()