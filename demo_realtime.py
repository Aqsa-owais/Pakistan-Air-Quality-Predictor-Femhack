#!/usr/bin/env python3
"""
Real-time Air Quality Prediction Demo
Comprehensive demonstration of real-time prediction capabilities
"""

import sys
import os
sys.path.append('src')

from realtime_predictor import RealTimeAQIPredictor
import json
import time

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🌫️  {title.upper()}")
    print("="*60)

def print_section(title):
    """Print section header"""
    print(f"\n📊 {title}")
    print("-" * 40)

def display_prediction_summary(result, scenario_name=""):
    """Display a concise prediction summary"""
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    category = result['predicted_category']
    confidence = result['confidence']
    aqi = result.get('aqi_estimate', 'N/A')
    
    # Category emoji mapping
    category_emojis = {
        'Good': '🟢',
        'Moderate': '🟡',
        'Unhealthy for Sensitive Groups': '🟠',
        'Unhealthy': '🔴',
        'Very Unhealthy': '🟣',
        'Hazardous': '⚫'
    }
    
    emoji = category_emojis.get(category, '⚪')
    
    if scenario_name:
        print(f"\n🎯 {scenario_name}:")
    
    print(f"   {emoji} Category: {category}")
    print(f"   📊 Confidence: {confidence:.1%}")
    print(f"   🔢 AQI Estimate: {aqi}")
    print(f"   💡 Top Recommendation: {result['recommendations'][0]}")

def demo_basic_prediction(predictor):
    """Demo basic real-time prediction"""
    print_section("Basic Real-time Prediction")
    
    # Sample input data
    input_data = {
        "PM2.5": 65,
        "PM10": 95,
        "Temperature": 30,
        "Humidity": 70,
        "Wind_Speed": 2.5,
        "City": "Lahore"
    }
    
    print("Input conditions:")
    for key, value in input_data.items():
        print(f"  {key}: {value}")
    
    print("\n🔄 Making prediction...")
    result = predictor.predict_from_input(input_data)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Category: {result['predicted_category']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   AQI Estimate: {result['aqi_estimate']}")
    print(f"   Timestamp: {result['timestamp']}")
    
    print(f"\n💡 Health Recommendations:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📈 Category Probabilities:")
    sorted_probs = sorted(result['all_probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    for category, prob in sorted_probs:
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"   {category:30} {bar} {prob:.1%}")

def demo_multiple_scenarios(predictor):
    """Demo predictions for multiple scenarios"""
    print_section("Multiple Scenario Predictions")
    
    scenarios = [
        {
            "name": "Clean Mountain Air",
            "data": {
                "PM2.5": 8,
                "PM10": 15,
                "Temperature": 18,
                "Humidity": 40,
                "Wind_Speed": 12,
                "City": "Murree"
            }
        },
        {
            "name": "Urban Morning",
            "data": {
                "PM2.5": 35,
                "PM10": 60,
                "Temperature": 25,
                "Humidity": 55,
                "Wind_Speed": 6,
                "City": "Islamabad"
            }
        },
        {
            "name": "Industrial Area",
            "data": {
                "PM2.5": 85,
                "PM10": 140,
                "Temperature": 32,
                "Humidity": 75,
                "Wind_Speed": 2,
                "City": "Faisalabad"
            }
        },
        {
            "name": "Smog Alert",
            "data": {
                "PM2.5": 180,
                "PM10": 280,
                "Temperature": 28,
                "Humidity": 85,
                "Wind_Speed": 0.5,
                "City": "Lahore"
            }
        }
    ]
    
    for scenario in scenarios:
        result = predictor.predict_from_input(scenario['data'])
        display_prediction_summary(result, scenario['name'])

def demo_batch_prediction(predictor):
    """Demo batch prediction functionality"""
    print_section("Batch Prediction Demo")
    
    # Create batch input data
    batch_data = [
        {
            "PM2.5": 25,
            "PM10": 45,
            "Temperature": 20,
            "Humidity": 50,
            "Wind_Speed": 8,
            "City": "Islamabad"
        },
        {
            "PM2.5": 75,
            "PM10": 120,
            "Temperature": 35,
            "Humidity": 80,
            "Wind_Speed": 1,
            "City": "Karachi"
        },
        {
            "PM2.5": 150,
            "PM10": 220,
            "Temperature": 30,
            "Humidity": 70,
            "Wind_Speed": 2,
            "City": "Lahore"
        }
    ]
    
    print(f"Processing {len(batch_data)} scenarios...")
    
    results = predictor.batch_predict(batch_data)
    
    print("\n📊 Batch Results Summary:")
    for i, result in enumerate(results):
        city = batch_data[i].get('City', f'Location {i+1}')
        display_prediction_summary(result, f"Scenario {i+1} ({city})")

def demo_sensitivity_analysis(predictor):
    """Demo sensitivity analysis - how predictions change with different inputs"""
    print_section("Sensitivity Analysis")
    
    base_conditions = {
        "PM2.5": 50,
        "PM10": 80,
        "Temperature": 25,
        "Humidity": 60,
        "Wind_Speed": 5,
        "City": "Lahore"
    }
    
    print("Base conditions:")
    for key, value in base_conditions.items():
        print(f"  {key}: {value}")
    
    base_result = predictor.predict_from_input(base_conditions)
    print(f"\nBase prediction: {base_result['predicted_category']} ({base_result['confidence']:.1%})")
    
    # Test sensitivity to different parameters
    sensitivity_tests = [
        ("High PM2.5", {"PM2.5": 120}),
        ("Low Wind", {"Wind_Speed": 0.5}),
        ("High Humidity", {"Humidity": 90}),
        ("High Temperature", {"Temperature": 40}),
        ("Combined Worst", {"PM2.5": 150, "Wind_Speed": 0.5, "Humidity": 90})
    ]
    
    print("\n🔬 Sensitivity Analysis:")
    for test_name, changes in sensitivity_tests:
        test_conditions = base_conditions.copy()
        test_conditions.update(changes)
        
        result = predictor.predict_from_input(test_conditions)
        
        if 'error' not in result:
            category = result['predicted_category']
            confidence = result['confidence']
            change_desc = ", ".join([f"{k}={v}" for k, v in changes.items()])
            
            print(f"   {test_name} ({change_desc}):")
            print(f"     → {category} ({confidence:.1%})")

def demo_real_world_examples(predictor):
    """Demo with real-world example scenarios"""
    print_section("Real-world Example Scenarios")
    
    real_scenarios = [
        {
            "name": "Lahore Winter Smog (Typical)",
            "description": "Heavy smog conditions during winter months",
            "data": {
                "PM2.5": 200,
                "PM10": 350,
                "Temperature": 15,
                "Humidity": 85,
                "Wind_Speed": 1,
                "City": "Lahore"
            }
        },
        {
            "name": "Karachi Sea Breeze Effect",
            "description": "Coastal winds helping disperse pollution",
            "data": {
                "PM2.5": 40,
                "PM10": 70,
                "Temperature": 28,
                "Humidity": 75,
                "Wind_Speed": 15,
                "City": "Karachi"
            }
        },
        {
            "name": "Islamabad Post-Rain",
            "description": "Clean air after rainfall",
            "data": {
                "PM2.5": 12,
                "PM10": 20,
                "Temperature": 22,
                "Humidity": 65,
                "Wind_Speed": 8,
                "City": "Islamabad"
            }
        },
        {
            "name": "Industrial Zone Peak Hours",
            "description": "High pollution during industrial activity",
            "data": {
                "PM2.5": 110,
                "PM10": 180,
                "Temperature": 32,
                "Humidity": 60,
                "Wind_Speed": 2,
                "City": "Faisalabad"
            }
        }
    ]
    
    for scenario in real_scenarios:
        print(f"\n🌍 {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        result = predictor.predict_from_input(scenario['data'])
        
        if 'error' not in result:
            category = result['predicted_category']
            confidence = result['confidence']
            aqi = result['aqi_estimate']
            
            print(f"   🎯 Prediction: {category}")
            print(f"   📊 Confidence: {confidence:.1%}")
            print(f"   🔢 AQI: {aqi}")
            print(f"   💡 Key Advice: {result['recommendations'][0]}")

def demo_model_info(predictor):
    """Display model information"""
    print_section("Model Information")
    
    model_info = predictor.get_model_info()
    
    if 'error' in model_info:
        print(f"❌ Error getting model info: {model_info['error']}")
        return
    
    print("🤖 Loaded Model Details:")
    print(f"   Name: {model_info.get('model_name', 'Unknown')}")
    print(f"   Training Date: {model_info.get('training_date', 'Unknown')}")
    print(f"   Feature Count: {model_info.get('feature_count', 0)}")
    print(f"   Categories: {', '.join(model_info.get('categories', []))}")
    
    if 'performance' in model_info and model_info['performance']:
        print(f"\n📈 Model Performance:")
        for metric, value in model_info['performance'].items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.3f}")
            else:
                print(f"   {metric}: {value}")

def main():
    """Main demo function"""
    print_header("Real-time Air Quality Prediction Demo")
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No trained models found.")
        print("   Please run: python run_pipeline.py")
        return
    
    model_path = f'models/{model_files[0]}'
    print(f"📂 Loading model: {model_path}")
    
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    print("✅ Model loaded successfully!")
    
    # Run all demos
    try:
        demo_model_info(predictor)
        
        demo_basic_prediction(predictor)
        
        demo_multiple_scenarios(predictor)
        
        demo_batch_prediction(predictor)
        
        demo_sensitivity_analysis(predictor)
        
        demo_real_world_examples(predictor)
        
        # Final summary
        print_header("Demo Complete")
        print("🎉 All real-time prediction demos completed successfully!")
        print("\n🚀 Next Steps:")
        print("   • Run CLI interface: python realtime_cli.py")
        print("   • Start web interface: python realtime_web.py")
        print("   • Use in your own code: from src.realtime_predictor import RealTimeAQIPredictor")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()