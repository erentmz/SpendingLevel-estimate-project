import unittest
import json
from app import app


class TestApp(unittest.TestCase):
    """Test suite for the Flask application."""

    def setUp(self):
        """Set up for test methods."""
        self.app = app.test_client()
        self.app.testing = True  # Enable testing mode

    def test_index_route_get(self):
        """Test that the index route responds with a 200 status code and HTML content."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)

    def test_index_route_post_valid_data(self):
        """Test the index route with a POST request and valid data."""
        data = {
            'age': 30,
            'salary': 50000,
            'job': 'Engineer'
        }
        response = self.app.post('/', data=data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction:', response.data)  # Check if prediction result is in HTML

    def test_index_route_post_invalid_data(self):
        """Test the index route with a POST request and invalid data."""
        data = {
            'age': 'invalid',  # Invalid data
            'salary': 50000,
            'job': 'Engineer'
        }
        response = self.app.post('/', data=data, follow_redirects=True)
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Prediction Error:', response.data)  # Check for error message in HTML

    def test_predict_route_valid_json(self):
        """Test the /predict route with valid JSON data."""
        data = {
            'age': 30,
            'salary': 50000,
            'job': 'Engineer'
        }
        response = self.app.post('/predict', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/json')
        json_data = json.loads(response.data)
        self.assertIn('prediction', json_data)  # Check that 'prediction' key is in JSON response

    def test_predict_route_invalid_json(self):
        """Test the /predict route with invalid JSON data."""
        data = {'invalid': 'data'}  # Invalid JSON
        response = self.app.post('/predict', json=data)
        self.assertEqual(response.status_code, 500)  # Expect an error
        self.assertEqual(response.content_type, 'application/json')
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)  # Check for error message in JSON response


if __name__ == '__main__':
    unittest.main()
