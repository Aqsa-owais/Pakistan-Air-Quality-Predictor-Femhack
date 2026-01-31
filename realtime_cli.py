#!/usr/bin/env python3
"""
Real-time Air Quality Prediction CLI
Interactive command-line interface for immediate AQI predictions
"""

import sys
import os
sys.path.append('src')

from realtime_predictor import RealTimeAQIPredictor
import json

def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("🌫️  REAL-TIME AIR QUALITY PREDICTOR  🌫️")
    print("="*60)
    print("Enter current air quality conditions for instant predictions!")
    print("Type 'help' for commands or 'quit' to exit\n")

def print_help():
    """Print help information"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("-" * 40)
    print("predict    - Start interactive prediction")
    print("quick      - Quick prediction with sample data")
    print("batch      - Batch prediction from file")
    print("model      - Show model information")
    print("examples   - Show input examples")
    print("help       - Show this help")
    print("quit/exit  - Exit the application")
    print()

def get_user_input():
    """Get air quality parameters from user"""
    print("📊 Enter current air quality conditions:")
    print("(Press Enter to skip optional parameters)")
    print()
    
    input_data = {}
    
    # Required parameters
    required_params = {
        'PM2.5': 'PM2.5 concentration (μg/m³)',
        'PM10': 'PM10 concentration (μg/m³)',
        'Temperature': 'Temperature (°C)',
        'Humidity': 'Humidity (%)',
        'Wind_Speed': 'Wind Speed (km/h)'
    }
    
    for param, description in required_params.items():
        while True:
            try:
                value = input(f"  {description}: ").strip()
                if value:
                    input_data[param] = float(value)
                    break
                else:
                    print(f"    ⚠️  {param} is required. Please enter a value.")
            except ValueError:
                print("    ❌ Please enter a valid number.")
    
    # Optional parameters
    optional_params = {
        'Pressure': 'Atmospheric Pressure (hPa)',
        'City': 'City name'
    }
    
    print("\n📍 Optional parameters:")
    for param, description in optional_params.items():
        value = input(f"  {description} (optional): ").strip()
        if value:
            if param == 'City':
                input_data[param] = value
            else:
                try:
                    input_data[param] = float(value)
                except ValueError:
                    print(f"    ⚠️  Invalid number for {param}, skipping...")
    
    return input_data

def display_prediction(result):
    """Display prediction results in a formatted way"""
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    print("\n" + "="*50)
    print("🎯 PREDICTION RESULTS")
    print("="*50)
    
    # Main prediction
    category = result['predicted_category']
    confidence = result['confidence']
    
    # Color coding for categories
    category_colors = {
        'Good': '🟢',
        'Moderate': '🟡',
        'Unhealthy for Sensitive Groups': '🟠',
        'Unhealthy': '🔴',
        'Very Unhealthy': '🟣',
        'Hazardous': '⚫'
    }
    
    color = category_colors.get(category, '⚪')
    
    print(f"\n{color} PREDICTED CATEGORY: {category}")
    print(f"📊 CONFIDENCE: {confidence:.1%}")
    
    if result.get('aqi_estimate'):
        print(f"🔢 ESTIMATED AQI: {result['aqi_estimate']}")
    
    print(f"📍 INPUT CONDITIONS: {result['input_summary']}")
    print(f"⏰ PREDICTION TIME: {result['timestamp']}")
    
    # Recommendations
    print(f"\n💡 HEALTH RECOMMENDATIONS:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Probability breakdown
    print(f"\n📈 CATEGORY PROBABILITIES:")
    sorted_probs = sorted(result['all_probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    
    for category, prob in sorted_probs:
        bar_length = int(prob * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {category:30} {bar} {prob:.1%}")
    
    print("="*50)

def quick_prediction(predictor):
    """Run quick prediction with sample data"""
    print("\n🚀 Quick Prediction with Sample Data")
    print("-" * 40)
    
    sample_data = {
        "PM2.5": 65,
        "PM10": 95,
        "Temperature": 30,
        "Humidity": 70,
        "Wind_Speed": 2.5,
        "City": "Lahore"
    }
    
    print("Using sample conditions:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict_from_input(sample_data)
    display_prediction(result)

def batch_prediction(predictor):
    """Run batch prediction from JSON file"""
    print("\n📁 Batch Prediction from File")
    print("-" * 40)
    
    filename = input("Enter JSON file path (or press Enter for sample): ").strip()
    
    if not filename:
        # Create sample batch file
        sample_batch = [
            {
                "scenario": "Morning in Islamabad",
                "PM2.5": 25,
                "PM10": 40,
                "Temperature": 18,
                "Humidity": 55,
                "Wind_Speed": 6,
                "City": "Islamabad"
            },
            {
                "scenario": "Evening in Karachi",
                "PM2.5": 85,
                "PM10": 120,
                "Temperature": 32,
                "Humidity": 75,
                "Wind_Speed": 3,
                "City": "Karachi"
            }
        ]
        
        print("Using sample batch data...")
        input_list = sample_batch
    else:
        try:
            with open(filename, 'r') as f:
                input_list = json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON format in: {filename}")
            return
    
    print(f"\n🔄 Processing {len(input_list)} scenarios...")
    
    results = predictor.batch_predict(input_list)
    
    for i, result in enumerate(results):
        scenario_name = input_list[i].get('scenario', f'Scenario {i+1}')
        print(f"\n📊 {scenario_name}:")
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   🎯 Category: {result['predicted_category']}")
            print(f"   📊 Confidence: {result['confidence']:.1%}")
            if result.get('aqi_estimate'):
                print(f"   🔢 AQI: {result['aqi_estimate']}")

def show_examples():
    """Show input examples"""
    print("\n📝 INPUT EXAMPLES:")
    print("-" * 40)
    
    examples = [
        {
            "name": "Clean Air Day",
            "data": {
                "PM2.5": 12,
                "PM10": 20,
                "Temperature": 25,
                "Humidity": 50,
                "Wind_Speed": 10,
                "City": "Murree"
            }
        },
        {
            "name": "Moderate Pollution",
            "data": {
                "PM2.5": 40,
                "PM10": 70,
                "Temperature": 28,
                "Humidity": 60,
                "Wind_Speed": 5,
                "City": "Islamabad"
            }
        },
        {
            "name": "Heavy Pollution",
            "data": {
                "PM2.5": 150,
                "PM10": 220,
                "Temperature": 35,
                "Humidity": 80,
                "Wind_Speed": 1,
                "City": "Lahore"
            }
        }
    ]
    
    for example in examples:
        print(f"\n🌟 {example['name']}:")
        for key, value in example['data'].items():
            print(f"   {key}: {value}")

def show_model_info(predictor):
    """Display model information"""
    print("\n🤖 MODEL INFORMATION:")
    print("-" * 40)
    
    model_info = predictor.get_model_info()
    
    if 'error' in model_info:
        print(f"❌ {model_info['error']}")
        return
    
    for key, value in model_info.items():
        if key == 'categories':
            print(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
        elif key == 'performance':
            print(f"{key.replace('_', ' ').title()}:")
            for metric, score in value.items():
                print(f"   {metric}: {score}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

def main():
    """Main CLI application"""
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No trained models found. Please run model training first.")
        print("   Run: python run_pipeline.py")
        return
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model. Please check model file.")
        return
    
    print_banner()
    
    while True:
        try:
            command = input("🔮 Enter command: ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Air Quality Predictor!")
                break
            
            elif command == 'help' or command == 'h':
                print_help()
            
            elif command == 'predict' or command == 'p':
                try:
                    input_data = get_user_input()
                    print("\n🔄 Making prediction...")
                    result = predictor.predict_from_input(input_data)
                    display_prediction(result)
                except KeyboardInterrupt:
                    print("\n⚠️  Prediction cancelled.")
            
            elif command == 'quick' or command == 'q':
                quick_prediction(predictor)
            
            elif command == 'batch' or command == 'b':
                batch_prediction(predictor)
            
            elif command == 'model' or command == 'm':
                show_model_info(predictor)
            
            elif command == 'examples' or command == 'e':
                show_examples()
            
            elif command == '':
                continue
            
            else:
                print(f"❌ Unknown command: '{command}'. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Type 'help' for available commands.")

if __name__ == "__main__":
    main()