"""
Test different prediction scenarios to check if we get varied results
"""

import sys
sys.path.append('src')
from realtime_predictor import RealTimeAQIPredictor
import os

def test_varied_predictions():
    """Test if different inputs give different predictions"""
    print("Testing varied predictions...")
    
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
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'Very Clean Air (Mountain Area)',
            'data': {
                'PM2.5': 5, 'PM10': 10, 'Temperature': 18, 
                'Humidity': 35, 'Wind_Speed': 15, 'City': 'Islamabad'
            }
        },
        {
            'name': 'Clean Air (Good Day)',
            'data': {
                'PM2.5': 15, 'PM10': 25, 'Temperature': 22, 
                'Humidity': 45, 'Wind_Speed': 8, 'City': 'Islamabad'
            }
        },
        {
            'name': 'Moderate Pollution (Urban)',
            'data': {
                'PM2.5': 45, 'PM10': 75, 'Temperature': 28, 
                'Humidity': 65, 'Wind_Speed': 3, 'City': 'Lahore'
            }
        },
        {
            'name': 'High Pollution (Industrial)',
            'data': {
                'PM2.5': 85, 'PM10': 120, 'Temperature': 32, 
                'Humidity': 75, 'Wind_Speed': 2, 'City': 'Faisalabad'
            }
        },
        {
            'name': 'Very High Pollution (Smog)',
            'data': {
                'PM2.5': 150, 'PM10': 200, 'Temperature': 35, 
                'Humidity': 85, 'Wind_Speed': 1, 'City': 'Karachi'
            }
        },
        {
            'name': 'Hazardous Level (Emergency)',
            'data': {
                'PM2.5': 300, 'PM10': 400, 'Temperature': 40, 
                'Humidity': 90, 'Wind_Speed': 0.5, 'City': 'Lahore'
            }
        }
    ]
    
    print("\n🔍 Testing different scenarios:")
    print("=" * 70)
    
    results = []
    unique_categories = set()
    
    for i, scenario in enumerate(scenarios, 1):
        result = predictor.predict_from_input(scenario['data'])
        
        if "error" in result:
            print(f"❌ Scenario {i} failed: {result['error']}")
            continue
        
        category = result['predicted_category']
        confidence = result['confidence']
        aqi_est = result.get('aqi_estimate', 'N/A')
        
        results.append({
            'scenario': scenario['name'],
            'category': category,
            'confidence': confidence,
            'aqi': aqi_est,
            'pm25': scenario['data']['PM2.5']
        })
        
        unique_categories.add(category)
        
        print(f"{i}. {scenario['name']}")
        print(f"   PM2.5: {scenario['data']['PM2.5']} | Category: {category}")
        print(f"   Confidence: {confidence:.1%} | AQI: {aqi_est}")
        print()
    
    print("=" * 70)
    print(f"📊 Results Summary:")
    print(f"   Total scenarios tested: {len(results)}")
    print(f"   Unique categories predicted: {len(unique_categories)}")
    print(f"   Categories found: {', '.join(sorted(unique_categories))}")
    
    # Check if we have variety
    if len(unique_categories) <= 2:
        print("❌ PROBLEM: Too few unique predictions - model may be biased")
        print("   Expected: Different categories for different PM2.5 levels")
        return False
    else:
        print("✅ GOOD: Multiple different predictions found")
        return True

def analyze_prediction_logic():
    """Analyze the prediction logic to understand why same results occur"""
    print("\n🔬 Analyzing prediction logic...")
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    # Test PM2.5-based prediction directly
    test_pm25_values = [5, 15, 35, 55, 120, 200, 350]
    
    print("\n📊 PM2.5-based predictions (should vary):")
    for pm25 in test_pm25_values:
        test_data = {
            'PM2.5': pm25, 'PM10': pm25 * 1.5, 'Temperature': 25,
            'Humidity': 60, 'Wind_Speed': 5
        }
        
        # Test the PM2.5-based prediction method directly
        category, confidence = predictor._predict_from_pm25(test_data)
        print(f"   PM2.5: {pm25:3d} → {category} ({confidence:.1%})")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Prediction Variety")
    print("=" * 50)
    
    success1 = test_varied_predictions()
    success2 = analyze_prediction_logic()
    
    if success1 and success2:
        print("\n🎉 Prediction variety test completed!")
    else:
        print("\n❌ Issues found with prediction variety.")
        print("💡 Recommendations:")
        print("   1. Check if model is overfitted")
        print("   2. Verify feature engineering")
        print("   3. Review PM2.5-based prediction logic")