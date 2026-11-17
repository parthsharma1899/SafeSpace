#!/usr/bin/env python3
"""
Test script to verify the SafeSpace Fusion web application
"""

import requests
import json

def test_analyze_endpoint():
    """Test the /analyze endpoint with sample data"""
    
    # Sample form data
    data = {
        'input_text': 'I am feeling stressed and overwhelmed today.',
        'wax': '-32.5',
        'way': '-55.0', 
        'waz': '4.5',
        'bvp': '-0.8',
        'eda': '0.4',
        'temp': '32.0',
        'sleep_duration': '7',
        'quality_of_sleep': '6',
        'physical_activity_level': '45',
        'daily_steps': '7000',
        'bmi_category': '1'
    }
    
    try:
        # Make POST request to analyze endpoint
        response = requests.post('http://localhost:8080/analyze', data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("✅ Test PASSED - JSON serialization working!")
                print(f"WMSS Score: {result['wmss']}")
                print(f"Stress Level: {result['stress_level']}")
                print(f"Dominant Modality: {result['dominant_modality']}")
                print("\nIndividual Scores:")
                for modality, score in result['individual_scores'].items():
                    print(f"  {modality}: {score}")
                print("\nModality Weights:")
                for modality, weight in result['normalized_weights'].items():
                    print(f"  {modality}: {weight:.4f}")
                return True
            else:
                print(f"❌ Test FAILED - Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Test FAILED - HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Test FAILED - Could not connect to server. Make sure the Flask app is running on port 8080.")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Test FAILED - JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test FAILED - Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing SafeSpace Fusion Web Application...")
    print("=" * 50)
    
    success = test_analyze_endpoint()
    
    if success:
        print("\n🎉 All tests passed! The web application is working correctly.")
        print("\nYou can now:")
        print("1. Open your browser")
        print("2. Go to http://localhost:8080")
        print("3. Use the web interface to analyze stress levels")
    else:
        print("\n❌ Tests failed. Please check the Flask application logs.") 