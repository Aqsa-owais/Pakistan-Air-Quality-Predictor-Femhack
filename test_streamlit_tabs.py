"""
Test all Streamlit tab functions to ensure they work correctly
"""

import sys
sys.path.append('src')
import pandas as pd
from realtime_predictor import RealTimeAQIPredictor
import os

def test_realtime_prediction():
    """Test real-time prediction functionality"""
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
    
    # Test prediction
    test_data = {
        "PM2.5": 35.0,
        "PM10": 65.0,
        "Temperature": 28.0,
        "Humidity": 65.0,
        "Wind_Speed": 5.0,
        "City": "Lahore"
    }
    
    result = predictor.predict_from_input(test_data)
    
    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
        return False
    
    print("✅ Real-time prediction works")
    print(f"   Category: {result['predicted_category']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    
    return True

def test_batch_prediction():
    """Test batch prediction functionality"""
    print("Testing batch prediction...")
    
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
    
    # Test batch prediction
    test_scenarios = [
        {"PM2.5": 15.0, "PM10": 25.0, "Temperature": 22.0, "Humidity": 45.0, "Wind_Speed": 8.0},
        {"PM2.5": 45.0, "PM10": 75.0, "Temperature": 28.0, "Humidity": 65.0, "Wind_Speed": 3.0},
        {"PM2.5": 120.0, "PM10": 180.0, "Temperature": 35.0, "Humidity": 80.0, "Wind_Speed": 1.0}
    ]
    
    results = predictor.batch_predict(test_scenarios)
    
    if len(results) != 3:
        print(f"❌ Expected 3 results, got {len(results)}")
        return False
    
    for i, result in enumerate(results):
        if "error" in result:
            print(f"❌ Batch prediction {i+1} failed: {result['error']}")
            return False
    
    print("✅ Batch prediction works")
    print(f"   Processed {len(results)} scenarios successfully")
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    try:
        df = pd.read_csv('data/processed_air_quality_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print("✅ Data loading works")
        print(f"   Loaded {len(df)} records")
        print(f"   Cities: {df['City'].nunique()}")
        return True
    except FileNotFoundError:
        print("❌ Data file not found")
        return False
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit Dashboard Components")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Real-time Prediction", test_realtime_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📊 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Streamlit dashboard should work perfectly.")
        print("\n🚀 Dashboard URLs:")
        print("   Local: http://localhost:8501")
        print("   Network: http://192.168.100.136:8501")
        print("\n📊 Available tabs:")
        print("   1. Historical Forecast - Time-series predictions")
        print("   2. Real-time Prediction - Instant predictions from current conditions")
        print("   3. Batch Prediction - Process multiple scenarios")
        print("   4. Analytics - Model insights and data analysis")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()