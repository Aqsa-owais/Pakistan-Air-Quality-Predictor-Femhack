"""
Quick City Variation Test
"""

import sys
sys.path.append('src')
from realtime_predictor import RealTimeAQIPredictor
import os

def main():
    print('🧪 Quick City Variation Test')
    print('=' * 40)
    
    # Load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    print('✅ Model loaded successfully')
    
    # Test same conditions in different cities
    base_conditions = {
        'PM2.5': 65, 
        'PM10': 95, 
        'Temperature': 30, 
        'Humidity': 70, 
        'Wind_Speed': 3
    }
    
    cities = ['Karachi', 'Lahore', 'Islamabad', 'Faisalabad']
    
    print('\nTesting same conditions across cities:')
    print('Base: PM2.5=65, PM10=95, Temp=30°C, Humidity=70%, Wind=3km/h')
    print('-' * 60)
    
    results = []
    for city in cities:
        test_data = base_conditions.copy()
        test_data['City'] = city
        result = predictor.predict_from_input(test_data)
        
        category = result['predicted_category']
        confidence = result['confidence']
        
        results.append((city, category, confidence))
        print(f'{city:12} → {category:25} ({confidence:.1%})')
    
    # Check for variation
    categories = set(r[1] for r in results)
    confidences = set(round(r[2], 2) for r in results)
    
    print('-' * 60)
    print(f'Unique categories: {len(categories)} → {", ".join(categories)}')
    print(f'Unique confidences: {len(confidences)}')
    
    if len(categories) > 1 or len(confidences) > 1:
        print('✅ SUCCESS: Cities show variation!')
    else:
        print('⚠️  WARNING: All cities giving same result')
    
    print('\n🚀 Test completed!')

if __name__ == "__main__":
    main()