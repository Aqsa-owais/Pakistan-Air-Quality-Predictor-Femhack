#!/usr/bin/env python3
"""
Simple test to verify the real-time predictor works
"""

import sys
import json
sys.path.append('src')

from realtime_predictor import RealTimeAQIPredictor
import os

def main():
    print("🧪 Simple Real-time Prediction Test")
    print("=" * 40)
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No models found. Please run: python run_pipeline.py")
        return
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Test with sample data
    print("\n📊 Testing with sample air quality data:")
    
    sample_data = {
        "PM2.5": 75,
        "PM10": 120,
        "Temperature": 32,
        "Humidity": 75,
        "Wind_Speed": 2,
        "City": "Lahore"
    }
    
    print("Input conditions:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    print("\n🔄 Making prediction...")
    result = predictor.predict_from_input(sample_data)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("✅ Prediction successful!")
    
    # Display results
    print(f"\n🎯 Results:")
    print(f"   Category: {result['predicted_category']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   AQI Estimate: {result['aqi_estimate']}")
    print(f"   Timestamp: {result['timestamp']}")
    
    print(f"\n💡 Top Recommendations:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    # Test JSON serialization
    print(f"\n🔄 Testing JSON conversion...")
    try:
        json_str = json.dumps(result, indent=2)
        print("✅ JSON serialization works!")
        
        # Show first few lines of JSON
        lines = json_str.split('\n')[:10]
        print("JSON preview:")
        for line in lines:
            print(f"   {line}")
        if len(json_str.split('\n')) > 10:
            print("   ...")
            
    except Exception as e:
        print(f"❌ JSON error: {e}")
        return
    
    print(f"\n🎉 All tests passed!")
    print(f"\n🚀 Ready to use:")
    print(f"   • CLI: python realtime_cli.py")
    print(f"   • Web: python realtime_web.py")

if __name__ == "__main__":
    main()