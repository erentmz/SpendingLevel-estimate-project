import unittest
import pandas as pd
from create_data import create_customers, low_income_jobs, income_ranges, medium_income_jobs, high_income_jobs


class TestCreateCustomers(unittest.TestCase):

    def test_creates_correct_number_of_customers(self):
        """Test that the function creates the correct number of customer records."""
        num_customers = 500
        df = create_customers(num_customers)
        self.assertEqual(len(df), num_customers)

    def test_dataframe_has_correct_columns(self):
        """Test that the DataFrame has the expected columns."""
        df = create_customers(10)
        expected_columns = ['customerId', 'name_surname', 'age', 'gender', 'job', 'salary', 'spending_score']
        self.assertListEqual(list(df.columns), expected_columns)

    def test_age_is_within_range(self):
        """Test that the generated age values are within the expected range."""
        df = create_customers(100)
        self.assertTrue(df['age'].between(18, 70).all())

    def test_spending_score_is_within_range(self):
        """Test that the spending score is always between 0 and 100."""
        df = create_customers(100)
        self.assertTrue(df['spending_score'].between(0, 100).all())

    def test_income_level_and_job_match(self):
        """Test that the generated job matches the income level."""
        df = create_customers(100)
        for index, row in df.iterrows():
            job_level = row['job']
            salary = row['salary']
            if job_level in low_income_jobs:
                self.assertTrue(income_ranges['low'][0] <= salary <= income_ranges['low'][1])
            elif job_level in medium_income_jobs:
                self.assertTrue(income_ranges['medium'][0] <= salary <= income_ranges['medium'][1])
            elif job_level in high_income_jobs:
                self.assertTrue(income_ranges['high'][0] <= salary <= income_ranges['high'][1])


if __name__ == '__main__':
    unittest.main()