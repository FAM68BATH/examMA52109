###
## cluster_maker - test file for preprocessing module
## Student: Fawaz Ahmed Mohideen
## Date: December 2025
###

import unittest
import pandas as pd
import numpy as np
from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    """
    Unit tests for the preprocessing module.
    Tests are designed to catch subtle bugs that could impact
    the entire clustering pipeline.
    """
    
    def setUp(self):
        """Create diverse test data to catch edge cases."""
        self.df = pd.DataFrame({
            'numeric_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric_2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'mixed_numeric': [1, 2.0, 3, 4.0, 5],  # Mixed int/float
            'all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'with_inf': [1.0, 2.0, np.inf, 4.0, 5.0]
        })
    
    def test_select_features_preserves_data_integrity_with_subtle_data_types(self):
        """
        Test that select_features correctly handles subtle numeric types that
        could silently break downstream clustering algorithms.
        
        REAL PROBLEM DETECTED: The function might appear to work but could:
        1. Convert integers to objects or strings, breaking distance calculations
        2. Fail with mixed int/float columns (common in real datasets)
        3. Lose precision with large numbers
        4. Mishandle special float values (inf, -inf)
        
        This test ensures that 'numeric' means truly numeric for computation,
        not just pandas dtype detection. Many clustering algorithms fail
        silently or produce nonsense results with wrong dtypes.
        """
        # Test with mixed int/float - common in real-world CSV data
        result = select_features(self.df, ['mixed_numeric'])
        
        # Critical check: result must be usable for numpy operations
        self.assertTrue(pd.api.types.is_float_dtype(result['mixed_numeric']) or 
                       pd.api.types.is_integer_dtype(result['mixed_numeric']))
        
        # Verify we can convert to float array without errors
        float_array = result['mixed_numeric'].to_numpy(dtype=float)
        self.assertEqual(float_array.shape, (5,))
        self.assertFalse(np.any(np.isnan(float_array[~np.isnan(float_array)])))
        
        # Test with infinity values - should be accepted as numeric
        # (clustering algorithms can handle inf, but need numeric type)
        result_inf = select_features(self.df, ['with_inf'])
        self.assertTrue(pd.api.types.is_float_dtype(result_inf['with_inf']))
        
        # Test that all-NaN columns are still considered numeric
        # (Important: clustering with standardization will handle NaNs)
        result_nan = select_features(self.df, ['all_nan'])
        self.assertTrue(pd.api.types.is_float_dtype(result_nan['all_nan']))
    
    def test_select_features_handles_column_ordering_and_memory_independence(self):
        """
        Test that select_features maintains correct column order and returns
        an independent copy, not a view that could cause mutation bugs.
        
        REAL PROBLEM DETECTED: 
        1. Returning a view instead of a copy could lead to accidental 
           mutation of original data when standardizing
        2. Incorrect column ordering could misalign features with their names
           in later analysis
        3. Memory sharing between returned DataFrame and original could cause
           hard-to-debug side effects
        
        These bugs are subtle but catastrophic: modifying standardized data
        could accidentally modify the original dataset.
        """
        # Test with specific column order (not alphabetical)
        requested_cols = ['numeric_2', 'numeric_1', 'mixed_numeric']
        result = select_features(self.df, requested_cols)
        
        # CRITICAL: Columns must be in requested order, not data order
        self.assertListEqual(list(result.columns), requested_cols)
        
        # CRITICAL: Result must be a copy, not a view
        # Modify the result and ensure original is unchanged
        original_value = self.df.at[0, 'numeric_1']
        result.at[0, 'numeric_1'] = 999.0
        
        self.assertEqual(self.df.at[0, 'numeric_1'], original_value,
                        "Original data was mutated - select_features() returned a view, not a copy!")
        
        # Test that the copy is deep enough for internal arrays
        result.iloc[1, 1] = 888.0
        self.assertNotEqual(self.df.at[1, 'numeric_1'], 888.0,
                           "Underlying array was shared - dangerous mutation risk!")
    
    def test_standardise_features_handles_edge_cases_that_break_clustering(self):
        """
        Test that standardise_features correctly handles edge cases that
        commonly break clustering algorithms or produce numerical errors.
        
        REAL PROBLEM DETECTED:
        1. Constant columns (zero variance) cause division by zero in standardization
        2. Single-row inputs break covariance calculations
        3. Numerical instability with very small/large values
        4. Incorrect handling of NaN/inf values
        
        These issues often manifest as silent failures: clustering runs but
        produces meaningless results or crashes with obscure numerical errors.
        """
        # Test 1: Constant column (zero variance)
        # Standardization should handle this gracefully
        constant_data = np.array([[1.0], [1.0], [1.0], [1.0]])
        
        # This should not crash - scikit-learn's StandardScaler handles constants
        # by setting them to zero (mean subtraction) without division
        scaled = standardise_features(constant_data)
        
        # After standardization, constant column should be all zeros
        self.assertTrue(np.allclose(scaled, 0.0),
                       "Constant column not properly standardized to zero")
        
        # Test 2: Single data point (edge case for variance calculation)
        single_point = np.array([[1.0, 2.0, 3.0]])
        scaled_single = standardise_features(single_point)
        
        # Single point standardization: (x - mean)/std where std=0
        # scikit-learn sets these to 0
        self.assertTrue(np.allclose(scaled_single, 0.0),
                       "Single data point not handled correctly")
        
        # Test 3: Very large value range - tests numerical stability
        large_range = np.array([[1e10], [1e-10], [0.0]])
        scaled_large = standardise_features(large_range)
        
        # Should not have NaN or inf from numerical overflow
        self.assertFalse(np.any(np.isnan(scaled_large)),
                        "Numerical overflow/underflow in standardization")
        self.assertFalse(np.any(np.isinf(scaled_large)),
                        "Infinite values from numerical issues")
        
        # Test 4: Input validation - catches misuse early
        with self.assertRaises(TypeError) as ctx:
            standardise_features(pd.DataFrame({'a': [1, 2, 3]}))  # DataFrame, not array
        
        self.assertIn('NumPy array', str(ctx.exception),
                     "Error message should guide user to correct input type")


if __name__ == '__main__':
    unittest.main()