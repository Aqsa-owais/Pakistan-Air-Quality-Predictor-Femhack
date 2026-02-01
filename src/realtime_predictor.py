"""
Real-time Air Quality Prediction Module
Handles real-time input and immediate AQI predictions
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealTimeAQIPredictor:
    def __init__(self, model_path=None):
        self.model_data = None
        self.model = None
        self.label_encoder = None
        self.feature_engineer = None
        self.feature_columns = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model and preprocessing objects"""
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.label_encoder = self.model_data['label_encoder']
            self.feature_engineer = self.model_data['feature_engineer']
            self.feature_columns = self.model_data['feature_columns']
            
            print(f"✅ Model loaded: {self.model_data['model_name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_from_input(self, input_data):
        """
        Predict AQI category from real-time input data
        
        Args:
            input_data (dict): Dictionary containing current conditions
                Required keys: PM2.5, PM10, Temperature, Humidity, Wind_Speed
                Optional keys: Pressure, City
        
        Returns:
            dict: Prediction results with category, confidence, and recommendations
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Use a hybrid approach: PM2.5-based prediction with model adjustment
            pm25_category, pm25_confidence = self._predict_from_pm25(input_data)
            
            # Create feature vector from input
            features = self._create_features_from_input(input_data)
            
            # Make model prediction
            prediction_encoded = self.model.predict(features)
            prediction_proba = self.model.predict_proba(features)
            
            # Decode model prediction
            model_category = self.label_encoder.inverse_transform(prediction_encoded)[0]
            model_confidence = float(max(prediction_proba[0]))
            
            # Combine PM2.5-based and model predictions
            # If PM2.5-based prediction has high confidence, use it
            # Otherwise, blend with model prediction
            if pm25_confidence > 0.8:
                predicted_category = pm25_category
                confidence = pm25_confidence
            else:
                # Use model prediction but adjust confidence
                predicted_category = model_category
                confidence = min(model_confidence, 0.85)  # Cap confidence to avoid overconfidence
            
            # Get confidence scores for all categories (more realistic distribution)
            confidence_scores = self._create_realistic_probabilities(predicted_category, confidence)
            
            # Generate recommendations
            recommendations = self._get_recommendations(predicted_category, input_data)
            
            # Calculate AQI estimate
            aqi_estimate = self._estimate_aqi(input_data)
            
            return {
                "predicted_category": str(predicted_category),
                "confidence": float(confidence),
                "aqi_estimate": int(aqi_estimate) if aqi_estimate else None,
                "all_probabilities": confidence_scores,
                "recommendations": recommendations,
                "input_summary": self._summarize_input(input_data),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _predict_from_pm25(self, input_data):
        """Predict AQI category based on PM2.5 values with realistic confidence"""
        if 'PM2.5' not in input_data:
            return 'Moderate', 0.5
        
        pm25 = float(input_data['PM2.5'])
        
        # PM2.5-based AQI category mapping (US EPA standards)
        if pm25 <= 12:
            category = 'Good'
            confidence = 0.9 if pm25 <= 8 else 0.8
        elif pm25 <= 35.4:
            category = 'Moderate'
            confidence = 0.85 if 15 <= pm25 <= 30 else 0.75
        elif pm25 <= 55.4:
            category = 'Unhealthy for Sensitive Groups'
            confidence = 0.8 if 40 <= pm25 <= 50 else 0.7
        elif pm25 <= 150.4:
            category = 'Unhealthy'
            confidence = 0.85 if 60 <= pm25 <= 120 else 0.75
        elif pm25 <= 250.4:
            category = 'Very Unhealthy'
            confidence = 0.8
        else:
            category = 'Hazardous'
            confidence = 0.9
        
        # Adjust confidence based on other factors
        if 'Wind_Speed' in input_data:
            wind_speed = float(input_data['Wind_Speed'])
            if wind_speed > 10:  # High wind can disperse pollution
                confidence *= 0.9
            elif wind_speed < 2:  # Low wind can trap pollution
                confidence *= 0.95
        
        return category, min(confidence, 0.95)
    
    def _create_realistic_probabilities(self, predicted_category, confidence):
        """Create realistic probability distribution across all categories"""
        categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                     'Unhealthy', 'Very Unhealthy', 'Hazardous']
        
        # Initialize probabilities
        probs = {cat: 0.0 for cat in categories}
        
        # Set main prediction probability
        probs[predicted_category] = confidence
        
        # Distribute remaining probability among adjacent categories
        remaining_prob = 1.0 - confidence
        
        # Find adjacent categories (pollution levels are ordered)
        pred_index = categories.index(predicted_category)
        
        # Distribute to adjacent categories
        if pred_index > 0:  # Can go to better category
            probs[categories[pred_index - 1]] = remaining_prob * 0.4
        if pred_index < len(categories) - 1:  # Can go to worse category
            probs[categories[pred_index + 1]] = remaining_prob * 0.4
        
        # Distribute remaining to other categories
        remaining_after_adjacent = remaining_prob * 0.2
        other_categories = [cat for i, cat in enumerate(categories) 
                          if abs(i - pred_index) > 1]
        
        if other_categories:
            prob_per_other = remaining_after_adjacent / len(other_categories)
            for cat in other_categories:
                probs[cat] = prob_per_other
        
        # Ensure probabilities sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {cat: prob/total for cat, prob in probs.items()}
        
        return probs
    
    def _create_features_from_input(self, input_data):
        """Create feature vector from input data"""
        # Initialize feature vector with zeros
        feature_vector = np.zeros(len(self.feature_columns))
        
        # Map input data to features
        feature_mapping = {
            'PM2.5': ['PM2.5'],
            'PM10': ['PM10'], 
            'Temperature': ['Temperature'],
            'Humidity': ['Humidity'],
            'Wind_Speed': ['Wind_Speed'],
            'Pressure': ['Pressure']
        }
        
        # Fill in available features
        for input_key, feature_names in feature_mapping.items():
            if input_key in input_data:
                value = float(input_data[input_key])
                
                for feature_name in feature_names:
                    if feature_name in self.feature_columns:
                        idx = self.feature_columns.index(feature_name)
                        feature_vector[idx] = value
        
        # Create synthetic lag and rolling features based on current values
        # This is a simplification - in a real system you'd have historical data
        self._create_synthetic_features(feature_vector, input_data)
        
        # Add seasonal features
        self._add_seasonal_features(feature_vector, input_data)
        
        return feature_vector.reshape(1, -1)
    
    def _create_synthetic_features(self, feature_vector, input_data):
        """Create synthetic lag and rolling features from current input"""
        # For real-time prediction without historical data, we create more realistic
        # synthetic features based on the input values and their relationships
        
        base_values = {
            'PM2.5': input_data.get('PM2.5', 50),
            'PM10': input_data.get('PM10', 80),
            'Temperature': input_data.get('Temperature', 25),
            'Humidity': input_data.get('Humidity', 60),
            'Wind_Speed': input_data.get('Wind_Speed', 5)
        }
        
        # Set random seed based on input values for consistent but varied results
        seed = int(sum(base_values.values()) * 1000) % 2147483647
        np.random.seed(seed)
        
        # Create lag features with more realistic variations
        for base_key, base_value in base_values.items():
            for lag in [1, 2, 3, 7]:
                lag_feature = f'{base_key}_lag_{lag}'
                if lag_feature in self.feature_columns:
                    idx = self.feature_columns.index(lag_feature)
                    
                    # Create more realistic lag values based on pollution patterns
                    if base_key in ['PM2.5', 'PM10']:
                        # Pollution tends to accumulate, so lag values might be lower
                        lag_factor = 0.7 + (lag * 0.05)  # Older values tend to be different
                        variation = np.random.normal(0, 0.15 + lag * 0.05) * base_value
                        feature_vector[idx] = max(0, base_value * lag_factor + variation)
                    elif base_key == 'Wind_Speed':
                        # Wind speed varies more randomly
                        variation = np.random.normal(0, 0.3) * base_value
                        feature_vector[idx] = max(0.1, base_value + variation)
                    else:
                        # Temperature and humidity have seasonal patterns
                        variation = np.random.normal(0, 0.1 + lag * 0.02) * base_value
                        feature_vector[idx] = base_value + variation
        
        # Create rolling features with more realistic statistics
        for base_key, base_value in base_values.items():
            for window in [3, 7, 14]:
                # Rolling mean (slightly different from current)
                roll_mean_feature = f'{base_key}_roll_mean_{window}'
                if roll_mean_feature in self.feature_columns:
                    idx = self.feature_columns.index(roll_mean_feature)
                    mean_variation = np.random.normal(0, 0.05) * base_value
                    feature_vector[idx] = base_value + mean_variation
                
                # Rolling std (realistic standard deviation)
                roll_std_feature = f'{base_key}_roll_std_{window}'
                if roll_std_feature in self.feature_columns:
                    idx = self.feature_columns.index(roll_std_feature)
                    if base_key in ['PM2.5', 'PM10']:
                        # Pollution has higher variability
                        feature_vector[idx] = base_value * (0.15 + np.random.uniform(0, 0.1))
                    else:
                        # Weather parameters have lower variability
                        feature_vector[idx] = base_value * (0.05 + np.random.uniform(0, 0.05))
                
                # Rolling max (higher than current, varies by parameter)
                roll_max_feature = f'{base_key}_roll_max_{window}'
                if roll_max_feature in self.feature_columns:
                    idx = self.feature_columns.index(roll_max_feature)
                    if base_key in ['PM2.5', 'PM10']:
                        # Pollution peaks can be much higher
                        max_factor = 1.2 + np.random.uniform(0, 0.5)
                    else:
                        # Weather parameters have smaller peaks
                        max_factor = 1.05 + np.random.uniform(0, 0.1)
                    feature_vector[idx] = base_value * max_factor
    
    def _add_seasonal_features(self, feature_vector, input_data):
        """Add seasonal and time-based features"""
        now = datetime.now()
        
        # Seasonal features
        month = now.month
        day_of_week = now.weekday()
        day_of_year = now.timetuple().tm_yday
        
        # Cyclical encoding
        seasonal_features = {
            'Month_sin': np.sin(2 * np.pi * month / 12),
            'Month_cos': np.cos(2 * np.pi * month / 12),
            'DayOfWeek_sin': np.sin(2 * np.pi * day_of_week / 7),
            'DayOfWeek_cos': np.cos(2 * np.pi * day_of_week / 7),
            'DayOfYear_sin': np.sin(2 * np.pi * day_of_year / 365),
            'DayOfYear_cos': np.cos(2 * np.pi * day_of_year / 365),
            'IsWeekend': 1 if day_of_week >= 5 else 0
        }
        
        for feature_name, value in seasonal_features.items():
            if feature_name in self.feature_columns:
                idx = self.feature_columns.index(feature_name)
                feature_vector[idx] = value
        
        # Interaction features - these make predictions more responsive to input combinations
        if 'Temperature' in input_data and 'Humidity' in input_data:
            temp_humidity = input_data['Temperature'] * input_data['Humidity'] / 100.0
            if 'Temp_Humidity' in self.feature_columns:
                idx = self.feature_columns.index('Temp_Humidity')
                feature_vector[idx] = temp_humidity
        
        # PM ratio - important for distinguishing pollution types
        if 'PM2.5' in input_data and 'PM10' in input_data:
            pm_ratio = input_data['PM2.5'] / (input_data['PM10'] + 1e-6)
            if 'PM_Ratio' in self.feature_columns:
                idx = self.feature_columns.index('PM_Ratio')
                feature_vector[idx] = pm_ratio
        
        # Wind-pollution interaction - wind affects pollution dispersion
        if 'Wind_Speed' in input_data and 'PM2.5' in input_data:
            wind_pm_interaction = input_data['PM2.5'] / (input_data['Wind_Speed'] + 1e-6)
            if 'Wind_PM_Interaction' in self.feature_columns:
                idx = self.feature_columns.index('Wind_PM_Interaction')
                feature_vector[idx] = wind_pm_interaction
        
        # Temperature-pollution interaction
        if 'Temperature' in input_data and 'PM2.5' in input_data:
            temp_pm_interaction = input_data['Temperature'] * input_data['PM2.5'] / 100.0
            if 'Temp_PM_Interaction' in self.feature_columns:
                idx = self.feature_columns.index('Temp_PM_Interaction')
                feature_vector[idx] = temp_pm_interaction
    
    def _estimate_aqi(self, input_data):
        """Rough AQI estimation from PM2.5 values"""
        if 'PM2.5' not in input_data:
            return None
        
        pm25 = float(input_data['PM2.5'])
        
        # Simplified AQI calculation based on PM2.5
        if pm25 <= 12:
            aqi = pm25 * 50 / 12
        elif pm25 <= 35.4:
            aqi = 50 + (pm25 - 12) * 50 / (35.4 - 12)
        elif pm25 <= 55.4:
            aqi = 100 + (pm25 - 35.4) * 50 / (55.4 - 35.4)
        elif pm25 <= 150.4:
            aqi = 150 + (pm25 - 55.4) * 50 / (150.4 - 55.4)
        elif pm25 <= 250.4:
            aqi = 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4)
        else:
            aqi = 300 + (pm25 - 250.4) * 100 / (350.4 - 250.4)
        
        return min(int(aqi), 500)
    
    def _get_recommendations(self, category, input_data):
        """Generate health recommendations based on predicted category"""
        recommendations = {
            'Good': [
                "✅ Air quality is good - enjoy outdoor activities!",
                "🚶‍♂️ Perfect time for jogging, cycling, or outdoor sports",
                "🪟 Keep windows open for fresh air circulation"
            ],
            'Moderate': [
                "⚠️ Air quality is acceptable for most people",
                "😷 Sensitive individuals should consider limiting prolonged outdoor exertion",
                "🏃‍♂️ Reduce intensity of outdoor activities if you experience symptoms"
            ],
            'Unhealthy for Sensitive Groups': [
                "🚨 Sensitive groups should limit outdoor activities",
                "😷 Wear a mask if you must go outside",
                "🪟 Keep windows closed and use air purifiers indoors",
                "👶 Children and elderly should stay indoors"
            ],
            'Unhealthy': [
                "🚨 Everyone should limit outdoor activities",
                "😷 Wear N95 masks when going outside",
                "🏠 Stay indoors as much as possible",
                "💨 Avoid outdoor exercise and strenuous activities"
            ],
            'Very Unhealthy': [
                "🚨 AVOID outdoor activities completely",
                "😷 Wear high-quality masks (N95/N99) if you must go out",
                "🏠 Stay indoors with air purifiers running",
                "🚫 Cancel outdoor events and activities"
            ],
            'Hazardous': [
                "🚨 EMERGENCY: Stay indoors at all times",
                "😷 Wear N99 masks even for brief outdoor exposure",
                "🏠 Seal windows and doors, use multiple air purifiers",
                "🚑 Seek medical attention if experiencing breathing difficulties"
            ]
        }
        
        base_recommendations = recommendations.get(category, ["Monitor air quality closely"])
        
        # Add specific recommendations based on input values
        additional_recs = []
        
        if 'Wind_Speed' in input_data:
            wind_speed = float(input_data['Wind_Speed'])
            if wind_speed < 2:
                additional_recs.append("🌬️ Low wind speed may worsen air quality - extra caution advised")
        
        if 'Humidity' in input_data:
            humidity = float(input_data['Humidity'])
            if humidity > 80:
                additional_recs.append("💧 High humidity may increase particle concentration")
        
        return base_recommendations + additional_recs
    
    def _summarize_input(self, input_data):
        """Create a summary of input conditions"""
        summary = []
        
        if 'PM2.5' in input_data:
            summary.append(f"PM2.5: {input_data['PM2.5']} μg/m³")
        if 'PM10' in input_data:
            summary.append(f"PM10: {input_data['PM10']} μg/m³")
        if 'Temperature' in input_data:
            summary.append(f"Temperature: {input_data['Temperature']}°C")
        if 'Humidity' in input_data:
            summary.append(f"Humidity: {input_data['Humidity']}%")
        if 'Wind_Speed' in input_data:
            summary.append(f"Wind Speed: {input_data['Wind_Speed']} km/h")
        if 'City' in input_data:
            summary.append(f"Location: {input_data['City']}")
        
        return " | ".join(summary)
    
    def batch_predict(self, input_list):
        """Predict for multiple input scenarios"""
        results = []
        
        for i, input_data in enumerate(input_list):
            result = self.predict_from_input(input_data)
            result['scenario_id'] = i + 1
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model_data is None:
            return {"error": "No model loaded"}
        
        return {
            "model_name": self.model_data.get('model_name', 'Unknown'),
            "training_date": self.model_data.get('training_date', 'Unknown'),
            "performance": self.model_data.get('performance', {}),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "categories": list(self.label_encoder.classes_) if self.label_encoder else []
        }

def main():
    """Demo function for real-time prediction"""
    import os
    
    # Find and load model
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No trained models found. Please run model training first.")
        return
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        return
    
    # Demo predictions with different scenarios
    print("\n🔮 Real-time AQI Prediction Demo")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Good Air Quality Day",
            "data": {
                "PM2.5": 15,
                "PM10": 25,
                "Temperature": 22,
                "Humidity": 45,
                "Wind_Speed": 8,
                "City": "Islamabad"
            }
        },
        {
            "name": "Moderate Pollution",
            "data": {
                "PM2.5": 45,
                "PM10": 75,
                "Temperature": 28,
                "Humidity": 65,
                "Wind_Speed": 3,
                "City": "Lahore"
            }
        },
        {
            "name": "High Pollution Alert",
            "data": {
                "PM2.5": 120,
                "PM10": 180,
                "Temperature": 35,
                "Humidity": 80,
                "Wind_Speed": 1,
                "City": "Karachi"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 30)
        
        result = predictor.predict_from_input(scenario['data'])
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"📍 Input: {result['input_summary']}")
        print(f"🎯 Predicted Category: {result['predicted_category']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        if result['aqi_estimate']:
            print(f"🔢 Estimated AQI: {result['aqi_estimate']}")
        
        print(f"⏰ Timestamp: {result['timestamp']}")
        
        print("\n💡 Recommendations:")
        for rec in result['recommendations'][:3]:  # Show first 3 recommendations
            print(f"   {rec}")
        
        print("\n📈 All Category Probabilities:")
        for category, prob in sorted(result['all_probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"   {category}: {prob:.1%}")
    
    # Model info
    print(f"\n🤖 Model Information:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        if key != 'performance':
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()