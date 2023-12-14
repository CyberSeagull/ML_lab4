import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processing import get_length, remove_symbols, remove_stopwords, convert_lower_case, stemming, preprocess_data
from random_forest import predict_and_save_probabilities
import mergee
import os
from sklearn.ensemble import RandomForestClassifier



class TestTextProcessing(unittest.TestCase):

    def test_get_length(self):
        mock_data = pd.DataFrame({'text': ['sample1', 'sample2', 'sample3']})
        self.assertEqual(get_length(mock_data), 3)

    def test_remove_symbols(self):
        self.assertEqual(remove_symbols("Hello! This, is a test."), "Hello This is a test")

    def test_remove_stopwords(self):
        self.assertEqual(remove_stopwords("This is a sample sentence"), "This sample sentence")

    def test_convert_lower_case(self):
        self.assertEqual(convert_lower_case("SAMPLE TEXT"), "sample text")

    def test_stemming(self):
        self.assertEqual(stemming("running"), "run")

    def test_preprocess_data(self):
        self.assertEqual(preprocess_data("Running, FAST!"), "run fast")


class TestPredictAndSaveProbabilities(unittest.TestCase):
    class TestPredictAndSaveProbabilities(unittest.TestCase):
        def setUp(self):
            num_samples = 10  # Example number of samples
            self.mock_train_set = pd.DataFrame({
                'feature1': np.random.rand(num_samples),
                'feature2': np.random.rand(num_samples),
                'label': np.random.randint(0, 2, num_samples)
            })
            self.mock_test_set = pd.DataFrame({
                'feature1': np.random.rand(num_samples),
                'feature2': np.random.rand(num_samples)
            })
            # ...

        def test_probability_column_addition(self):
            # Adjust the test to ensure consistency
            predict_and_save_probabilities('label', self.mock_train_set, self.mock_test_set, self.train_features,
                                           self.test_features, self.output_file)
            result_df = pd.read_csv(self.output_file)
            self.assertIn('label_probability', result_df.columns)
            self.assertEqual(len(self.mock_test_set), len(result_df))
            os.remove(self.output_file)


# Run the tests
if __name__ == '__main__':
    unittest.main()

