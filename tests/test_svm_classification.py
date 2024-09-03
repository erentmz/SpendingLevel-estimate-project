import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from svm_classification import load_and_preprocess_data, create_pipeline, train_and_evaluate_model


class TestSVMClassification(unittest.TestCase):

    def setUp(self):
        """Set up common data for the tests."""
        self.data_path = '../data/customer_table1k.csv'
        self.sample_data = pd.DataFrame({
            'age': [25, 35, 45],
            'salary': [30000, 60000, 90000],
            'job': ['Cashier', 'Teacher', 'Engineer'],
            'spending_score': [75, 50, 25]
        })

    def test_load_and_preprocess_data_success(self):
        """Test that data loading and preprocessing completes successfully."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(self.data_path)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_load_and_preprocess_data_file_not_found(self):
        """Test that file not found error is handled correctly."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data("non_existent_file.csv")
        self.assertIsNone(X_train)
        self.assertIsNone(X_test)
        self.assertIsNone(y_train)
        self.assertIsNone(y_test)

    def test_create_pipeline(self):
        """Test that the pipeline is created correctly."""
        pipeline = create_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.steps), 2)

    def test_train_and_evaluate_model(self):
        """Test that the model is trained and evaluated."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.sample_data[['age', 'salary', 'job']],
            self.sample_data['spending_score'],
            test_size=0.2,
            random_state=42
        )
        pipeline = create_pipeline()
        trained_pipeline = train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        self.assertIsNotNone(trained_pipeline)
        self.assertIsInstance(trained_pipeline.named_steps['classifier'], SVC)


if __name__ == '__main__':
    unittest.main()
