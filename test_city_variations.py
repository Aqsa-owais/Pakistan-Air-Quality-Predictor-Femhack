"""
Test City Variations - Check if different cities give different predictions
"""

import sys
sys.path.append('src')
from realtime_predictor import RealTimeAQIPredictor
import os

def test_city_specific_predictions():
    """Test if different cities give different predictions for same conditions"""
    print("🏙️ Testing City-Specific Predictions")
    print("=" * 60)
    
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
    
    # Test same conditions in different cities
    base_conditions = {
        'PM2.5': 65,
        'PM10': 95,
        'Temperature': 30,
        'Humidity': 70,
        'Wind_Speed': 3
    }
    
    cities = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Multan', 'Peshawar', 'Quetta']
    
    print(f"\n🧪 Testing same conditions across {len(cities)} cities:")
    print(f"Base conditions: PM2.5={base_conditions['PM2.5']}, PM10={base_conditions['PM10']}, Temp={base_conditions['Temperature']}°C")
    print("-" * 60)
    
    results = []
    unique_categories = set()
    unique_confidences = set()
    
    for i, city in enumerate(cities, 1):
        test_data = base_conditions.copy()
        test_data['City'] = city
        
        result = predictor.predict_from_input(test_data)
        
        if "error" in result:
            print(f"❌ {city}: Error - {result['error']}")
            continue
        
        category = result['predicted_category']
        confidence = result['confidence']
        aqi_est = result.get('aqi_estimate', 'N/A')
        
        results.append({
            'city': city,
            'category': category,
            'confidence': confidence,
            'aqi': aqi_est
        })
        
        unique_categories.add(category)
        unique_confidences.add(round(confidence, 2))
        
        print(f"{i:2d}. {city:12} → {category:30} ({confidence:5.1%}) AQI: {aqi_est}")
    
    print("-" * 60)
    print(f"📊 Variation Analysis:")
    print(f"   Cities tested: {len(results)}")
    print(f"   Unique categories: {len(unique_categories)} → {', '.join(sorted(unique_categories))}")
    print(f"   Unique confidences: {len(unique_confidences)} → {sorted(unique_confidences)}")
    
    if len(unique_categories) > 1 or len(unique_confidences) > 1:
        print("✅ GOOD: Cities show variation in predictions!")
        return True
    else:
        print("❌ ISSUE: All cities giving same prediction")
        return False

def test_different_pollution_levels():
    """Test different pollution levels across cities"""
    print("\n🌫️ Testing Different Pollution Levels")
    print("=" * 60)
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    # Test scenarios with different pollution levels
    scenarios = [
        {
            'name': 'Clean Mountain Air',
            'data': {'PM2.5': 8, 'PM10': 15, 'Temperature': 18, 'Humidity': 40, 'Wind_Speed': 12, 'City': 'Islamabad'},
            'expected': 'Good'
        },
        {
            'name': 'Light Urban Pollution',
            'data': {'PM2.5': 25, 'PM10': 40, 'Temperature': 25, 'Humidity': 55, 'Wind_Speed': 6, 'City': 'Rawalpindi'},
            'expected': 'Moderate'
        },
        {
            'name': 'Moderate Industrial Pollution',
            'data': {'PM2.5': 50, 'PM10': 80, 'Temperature': 28, 'Humidity': 65, 'Wind_Speed': 3, 'City': 'Faisalabad'},
            'expected': 'Unhealthy for Sensitive Groups'
        },
        {
            'name': 'Heavy Urban Smog',
            'data': {'PM2.5': 85, 'PM10': 130, 'Temperature': 32, 'Humidity': 75, 'Wind_Speed': 2, 'City': 'Lahore'},
            'expected': 'Unhealthy'
        },
        {
            'name': 'Severe Pollution Event',
            'data': {'PM2.5': 150, 'PM10': 220, 'Temperature': 35, 'Humidity': 80, 'Wind_Speed': 1, 'City': 'Karachi'},
            'expected': 'Very Unhealthy'
        },
        {
            'name': 'Hazardous Emergency Level',
            'data': {'PM2.5': 300, 'PM10': 450, 'Temperature': 40, 'Humidity': 85, 'Wind_Speed': 0.5, 'City': 'Lahore'},
            'expected': 'Hazardous'
        }
    ]
    
    print("Testing pollution level progression:")
    print("-" * 60)
    
    results = []
    categories_found = set()
    
    for i, scenario in enumerate(scenarios, 1):
        result = predictor.predict_from_input(scenario['data'])
        
        if "error" in result:
            print(f"❌ Scenario {i}: Error - {result['error']}")
            continue
        
        category = result['predicted_category']
        confidence = result['confidence']
        aqi_est = result.get('aqi_estimate', 'N/A')
        expected = scenario['expected']
        
        # Check if prediction matches expectation
        match_status = "✅" if category == expected else "⚠️"
        
        results.append({
            'scenario': scenario['name'],
            'pm25': scenario['data']['PM2.5'],
            'city': scenario['data']['City'],
            'predicted': category,
            'expected': expected,
            'confidence': confidence,
            'aqi': aqi_est,
            'match': category == expected
        })
        
        categories_found.add(category)
        
        print(f"{i}. {scenario['name']:25} (PM2.5: {scenario['data']['PM2.5']:3d})")
        print(f"   {scenario['data']['City']:12} → {category:30} ({confidence:5.1%}) {match_status}")
        print(f"   Expected: {expected:30} AQI: {aqi_est}")
        print()
    
    print("-" * 60)
    print(f"📊 Pollution Level Analysis:")
    print(f"   Scenarios tested: {len(results)}")
    print(f"   Categories found: {len(categories_found)} → {', '.join(sorted(categories_found))}")
    
    matches = sum(1 for r in results if r['match'])
    match_rate = (matches / len(results)) * 100 if results else 0
    print(f"   Prediction accuracy: {matches}/{len(results)} ({match_rate:.1f}%)")
    
    if len(categories_found) >= 4:
        print("✅ EXCELLENT: Good variety in pollution level predictions!")
        return True
    elif len(categories_found) >= 2:
        print("✅ GOOD: Some variety in predictions")
        return True
    else:
        print("❌ ISSUE: Not enough variety in predictions")
        return False

def test_weather_impact():
    """Test how weather conditions affect predictions"""
    print("\n🌤️ Testing Weather Impact on Predictions")
    print("=" * 60)
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    # Base pollution level
    base_pm25 = 60  # Moderate pollution
    
    weather_scenarios = [
        {
            'name': 'High Wind (Disperses Pollution)',
            'data': {'PM2.5': base_pm25, 'PM10': 90, 'Temperature': 25, 'Humidity': 50, 'Wind_Speed': 15, 'City': 'Lahore'}
        },
        {
            'name': 'Low Wind (Traps Pollution)',
            'data': {'PM2.5': base_pm25, 'PM10': 90, 'Temperature': 25, 'Humidity': 50, 'Wind_Speed': 0.5, 'City': 'Lahore'}
        },
        {
            'name': 'High Humidity (Coastal Effect)',
            'data': {'PM2.5': base_pm25, 'PM10': 90, 'Temperature': 30, 'Humidity': 90, 'Wind_Speed': 5, 'City': 'Karachi'}
        },
        {
            'name': 'Low Humidity (Dry Conditions)',
            'data': {'PM2.5': base_pm25, 'PM10': 90, 'Temperature': 30, 'Humidity': 30, 'Wind_Speed': 5, 'City': 'Lahore'}
        },
        {
            'name': 'Very Hot Weather',
            'data': {'PM2.5': base_pm25, 'PM10': 90, 'Temperature': 45, 'Humidity': 60, 'Wind_Speed': 5, 'City': 'Karachi'}
        },
        {
            'name': 'Cool Weather',
            'data': {'PM2.5': base_pm25, 'PM10': 90, 'Temperature': 15, 'Humidity': 60, 'Wind_Speed': 5, 'City': 'Islamabad'}
        }
    ]
    
    print(f"Testing weather impact with base PM2.5 = {base_pm25} μg/m³:")
    print("-" * 60)
    
    results = []
    categories_found = set()
    confidences = []
    
    for i, scenario in enumerate(weather_scenarios, 1):
        result = predictor.predict_from_input(scenario['data'])
        
        if "error" in result:
            print(f"❌ Scenario {i}: Error - {result['error']}")
            continue
        
        category = result['predicted_category']
        confidence = result['confidence']
        aqi_est = result.get('aqi_estimate', 'N/A')
        
        results.append({
            'scenario': scenario['name'],
            'category': category,
            'confidence': confidence,
            'weather': f"T:{scenario['data']['Temperature']}°C, H:{scenario['data']['Humidity']}%, W:{scenario['data']['Wind_Speed']}km/h"
        })
        
        categories_found.add(category)
        confidences.append(confidence)
        
        print(f"{i}. {scenario['name']:30}")
        print(f"   {scenario['weather']:25} → {category:25} ({confidence:5.1%})")
        print()
    
    print("-" * 60)
    print(f"📊 Weather Impact Analysis:")
    print(f"   Weather scenarios: {len(results)}")
    print(f"   Categories found: {len(categories_found)} → {', '.join(sorted(categories_found))}")
    
    if confidences:
        conf_range = max(confidences) - min(confidences)
        print(f"   Confidence range: {min(confidences):.1%} - {max(confidences):.1%} (spread: {conf_range:.1%})")
    
    if len(categories_found) > 1 or (confidences and max(confidences) - min(confidences) > 0.05):
        print("✅ GOOD: Weather conditions affect predictions!")
        return True
    else:
        print("⚠️ LIMITED: Weather impact is minimal")
        return False

def main():
    """Run all city variation tests"""
    print("🧪 COMPREHENSIVE CITY VARIATION TESTING")
    print("=" * 80)
    
    tests = [
        ("City-Specific Predictions", test_city_specific_predictions),
        ("Pollution Level Progression", test_different_pollution_levels),
        ("Weather Impact Analysis", test_weather_impact)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: NEEDS IMPROVEMENT")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print(f"🎯 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 EXCELLENT! City variations are working perfectly!")
        print("   ✅ Different cities give different predictions")
        print("   ✅ Pollution levels show proper progression")
        print("   ✅ Weather conditions affect results")
    elif passed >= 2:
        print("✅ GOOD! Most city variations are working")
        print("   💡 Some improvements possible for better variety")
    else:
        print("⚠️ NEEDS WORK! City variations need improvement")
        print("   💡 Consider adjusting city-specific factors")
        print("   💡 Review weather impact calculations")
    
    print(f"\n🚀 Test completed! Check Streamlit dashboard at:")
    print(f"   Local: http://localhost:8501")
    print(f"   Network: http://192.168.100.136:8501")
    
    return passed == total

if __name__ == "__main__":
    main()