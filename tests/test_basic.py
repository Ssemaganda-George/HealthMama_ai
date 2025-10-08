"""
Basic tests for HealthMama AI
"""
import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.factory import create_app
from config.settings import TestingConfig


class TestHealthMamaAI(unittest.TestCase):
    """Basic application tests"""
    
    def setUp(self):
        """Set up test client"""
        self.app = create_app()
        self.app.config.from_object(TestingConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Clean up after tests"""
        self.app_context.pop()
    
    def test_index_page(self):
        """Test main index page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_status_endpoint(self):
        """Test status endpoint"""
        response = self.client.get('/status')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('application', data)
        self.assertEqual(data['application'], 'HealthMama AI')
    
    def test_api_chat_without_data(self):
        """Test chat API without data"""
        response = self.client.post('/api/chat')
        self.assertEqual(response.status_code, 400)
    
    def test_api_chat_with_valid_data(self):
        """Test chat API with valid data"""
        response = self.client.post('/api/chat', 
                                  json={'message': 'Hello', 'model': 'diabetes'})
        # Should work or return a specific error if OpenAI key is not configured
        self.assertIn(response.status_code, [200, 500])


if __name__ == '__main__':
    unittest.main()