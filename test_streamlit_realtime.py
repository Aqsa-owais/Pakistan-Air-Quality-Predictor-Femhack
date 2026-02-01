"""
Test script for Streamlit real-time prediction functionality
"""

import sys
sys.path.append('src')

from realtime_predictor import RealTimeAQIPredictor
import os

def test_realtime_prediction():
    """Test the real-time prediction functionality"""
    print("Testing real-time prediction...")
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No trained models found")
        return False
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Test prediction
    test_data = {
        "PM2.5": 35.0,
        "PM10": 65.0,
        "Temperature": 28.0,
        "Humidity": 65.0,
        "Wind_Speed": 5.0,
        "City": "Lahore"
    }
    
    print(f"Testing with data: {test_data}")
    
    result = predictor.predict_from_input(test_data)
    
    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
        return False
    
    print("✅ Prediction successful!")
    print(f"   Category: {result['predicted_category']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   AQI Estimate: {result.get('aqi_estimate', 'N/A')}")
    
    # Test probability chart data
    probs = result['all_probabilities']
    print(f"   Probabilities: {len(probs)} categories")
    
    total_prob = sum(probs.values())
    print(f"   Total probability: {total_prob:.3f} (should be ~1.0)")
    
    if abs(total_prob - 1.0) > 0.01:
        print("⚠️  Warning: Probabilities don't sum to 1.0")
    
    return True

if __name__ == "__main__":
    success = test_realtime_prediction()
    if success:
        print("\n🎉 All tests passed! Streamlit app should work correctly.")
    else:
        print("\n❌ Tests failed. Please check the issues above.")