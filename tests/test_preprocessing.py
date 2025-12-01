###
## cluster_maker - test file for preprocessing module
## Student: [Your Name]
## Date: [Current Date]
###

import unittest
import pandas as pd
import numpy as np
from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    """
    Unit tests for the preprocessing module.
    """
    
    def setUp(self):
        """Create a sample DataFrame for testing."""
        self.sample_data = pd.DataFrame({
            'numeric_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric_2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'numeric_3': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
    
    def test_select_features_valid_numeric_columns(self):
        """
        Test that select_features correctly extracts and validates numeric columns.
        
        This test verifies the core functionality: selecting valid numeric columns
        while ensuring non-numeric columns are rejected. It catches issues where
        the function might incorrectly accept non-numeric data or fail to return
        the correct subset of columns.
        """
        # Test with valid numeric columns
        result = select_features(self.sample_data, ['numeric_1', 'numeric_2', 'numeric_3'])
        
        # Verify shape and columns
        self.assertEqual(result.shape, (5, 3))
        self.assertListEqual(list(result.columns), ['numeric_1', 'numeric_2', 'numeric_3'])
        
        # Verify data integrity
        pd.testing.assert_frame_equal(
            result,
            self.sample_data[['numeric_1', 'numeric_2', 'numeric_3']]
        )
        
        # Verify all columns are numeric
        for col in result.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))
    
    def test_select_features_raises_error_for_missing_columns(self):
        """
        Test that select_features raises KeyError for non-existent columns.
        
        This test ensures the function properly validates input and provides
        clear error messages when requested columns don't exist. Without this
        check, users might get cryptic errors later in the pipeline when
        attempting to access missing data.
        """
        # Test with one missing column
        with self.assertRaises(KeyError) as context:
            select_features(self.sample_data, ['numeric_1', 'non_existent_column'])
        
        # Verify error message is informative
        self.assertIn('non_existent_column', str(context.exception))
        
        # Test with multiple missing columns
        with self.assertRaises(KeyError) as context:
            select_features(self.sample_data, ['numeric_1', 'missing_1', 'missing_2'])
        
        # Verify all missing columns are reported
        error_msg = str(context.exception)
        self.assertIn('missing_1', error_msg)
        self.assertIn('missing_2', error_msg)
    
    def test_select_features_raises_error_for_non_numeric_columns(self):
        """
        Test that select_features raises TypeError for non-numeric columns.
        
        This test ensures the function enforces data type requirements early
        in the pipeline. Clustering algorithms require numeric input, so
        accepting non-numeric data would lead to runtime errors or incorrect
        results later. The test catches cases where type checking might be
        incomplete or incorrectly implemented.
        """
        # Test with string column
        with self.assertRaises(TypeError) as context:
            select_features(self.sample_data, ['numeric_1', 'string_col'])
        
        # Verify error message identifies the problematic column
        error_msg = str(context.exception)
        self.assertIn('string_col', error_msg)
        
        # Test with mixed valid and invalid columns
        with self.assertRaises(TypeError) as context:
            select_features(self.sample_data, ['numeric_1', 'numeric_2', 'string_col', 'numeric_3'])
        
        # Verify the error message lists all non-numeric columns
        error_msg = str(context.exception)
        self.assertIn('string_col', error_msg)
        
        # Test edge case: all columns are non-numeric
        string_only_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })
        
        with self.assertRaises(TypeError) as context:
            select_features(string_only_df, ['col1', 'col2'])
    


if __name__ == '__main__':
    unittest.main()