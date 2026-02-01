#!/usr/bin/env python3
"""
Test script to verify JSON serialization fix
"""

import sys
import json
sys.path.append('src')

from realtime_predictor import RealTimeAQIPredictor
import os

def test_json_serialization():
    """Test if the prediction results can be serialized to JSON"""
    print("🧪 Testing JSON Serialization Fix")
    print("=" * 40)
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No models found. Run: python run_pipeline.py")
        return False
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Test prediction
    test_data = {
        "PM2.5": 65,
        "PM10": 95,
        "Temperature": 30,
        "Humidity": 70,
        "Wind_Speed": 2.5,
        "City": "Lahore"
    }
    
    print("\n🔄 Making prediction...")
    result = predictor.predict_from_input(test_data)
    
    if 'error' in result:
        print(f"❌ Prediction error: {result['error']}")
        return False
    
    print("✅ Prediction successful")
    
    # Test JSON serialization
    print("\n🔄 Testing JSON serialization...")
    try:
        json_string = json.dumps(result, indent=2)
        print("✅ JSON serialization successful!")
        
        # Test deserialization
        parsed_result = json.loads(json_string)
        print("✅ JSON deserialization successful!")
        
        print(f"\n📊 Result Preview:")
        print(f"   Category: {parsed_result['predicted_category']}")
        print(f"   Confidence: {parsed_result['confidence']:.1%}")
        print(f"   AQI: {parsed_result['aqi_estimate']}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

def test_web_api():
    """Test if web API can handle the results"""
    print("\n🌐 Testing Web API Compatibility")
    print("-" * 30)
    
    try:
        from flask import jsonify
        
        # Simulate web API response
        test_data = {
            "PM2.5": 45,
            "PM10": 75,
            "Temperature": 28,
            "Humidity": 65,
            "Wind_Speed": 3
        }
        
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        model_path = f'models/{model_files[0]}'
        predictor = RealTimeAQIPredictor(model_path)
        
        result = predictor.predict_from_input(test_data)
        
        # Test Flask jsonify
        response = jsonify(result)
        print("✅ Flask jsonify successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Web API test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_json_serialization()
    success2 = test_web_api()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 All tests passed! JSON serialization issue is fixed.")
        print("\n🚀 You can now run:")
        print("   • python realtime_cli.py")
        print("   • python realtime_web.py")
    else:
        print("❌ Some tests failed. Check the error messages above.")