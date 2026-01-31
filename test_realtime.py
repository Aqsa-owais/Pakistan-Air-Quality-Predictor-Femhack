#!/usr/bin/env python3
"""
Quick test of real-time prediction system
"""

import sys
sys.path.append('src')

from realtime_predictor import RealTimeAQIPredictor
import os

def main():
    """Quick test of the real-time predictor"""
    print("🧪 Testing Real-time Air Quality Predictor")
    print("=" * 50)
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No models found. Run: python run_pipeline.py")
        return
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # Test with very clean air conditions
    print("\n🧪 Test 1: Very Clean Air")
    clean_air = {
        "PM2.5": 5,
        "PM10": 10,
        "Temperature": 20,
        "Humidity": 40,
        "Wind_Speed": 15,
        "City": "Mountain Area"
    }
    
    result = predictor.predict_from_input(clean_air)
    print(f"   Input: PM2.5={clean_air['PM2.5']}, PM10={clean_air['PM10']}")
    print(f"   Result: {result['predicted_category']} ({result['confidence']:.1%})")
    
    # Test with extremely polluted conditions
    print("\n🧪 Test 2: Extremely Polluted")
    polluted_air = {
        "PM2.5": 300,
        "PM10": 500,
        "Temperature": 35,
        "Humidity": 90,
        "Wind_Speed": 0.1,
        "City": "Industrial Zone"
    }
    
    result = predictor.predict_from_input(polluted_air)
    print(f"   Input: PM2.5={polluted_air['PM2.5']}, PM10={polluted_air['PM10']}")
    print(f"   Result: {result['predicted_category']} ({result['confidence']:.1%})")
    
    # Show model info
    print("\n🤖 Model Info:")
    info = predictor.get_model_info()
    print(f"   Categories: {info.get('categories', [])}")
    print(f"   Features: {info.get('feature_count', 0)}")
    
    print("\n✅ Test completed!")
    print("\n🚀 To use the real-time predictor:")
    print("   • CLI: python realtime_cli.py")
    print("   • Web: python realtime_web.py")

if __name__ == "__main__":
    main()