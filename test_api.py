import unittest
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_invalid_input(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
