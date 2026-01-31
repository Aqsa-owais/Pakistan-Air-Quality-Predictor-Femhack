"""
Prediction Module for Air Quality Forecasting
Handles model loading and prediction generation
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AQIForecastor:
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
            
            print(f"Model loaded successfully: {self.model_data['model_name']}")
            print(f"Training date: {self.model_data['training_date']}")
            print(f"Model performance: {self.model_data['performance']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_prediction_data(self, df, city, target_date):
        """Prepare data for prediction"""
        # Filter data for the specific city up to the target date
        city_data = df[(df['City'] == city) & (df['Date'] <= target_date)].copy()
        city_data = city_data.sort_values('Date')
        
        if len(city_data) == 0:
            raise ValueError(f"No data available for {city} up to {target_date}")
        
        # Get the latest available data point
        latest_data = city_data.tail(1).copy()
        
        # Create features using the feature engineer
        # Note: In a real scenario, you'd need recent weather data
        # For demo, we'll use the latest available features
        
        # Extract features
        feature_values = []
        for col in self.feature_columns:
            if col in latest_data.columns:
                feature_values.append(latest_data[col].iloc[0])
            else:
                # Handle missing features with default values
                feature_values.append(0.0)
        
        return np.array(feature_values).reshape(1, -1)
    
    def predict_aqi_category(self, features):
        """Predict AQI category"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Scale features
        features_scaled = self.feature_engineer.scaler.transform(features)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)
        prediction_proba = self.model.predict_proba(features_scaled)
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform(prediction_encoded)
        
        # Get confidence scores
        confidence_scores = dict(zip(self.label_encoder.classes_, prediction_proba[0]))
        
        return prediction[0], confidence_scores
    
    def forecast_multiple_days(self, df, city, start_date, days=3):
        """Forecast AQI for multiple days"""
        forecasts = []
        current_date = start_date
        
        for day in range(days):
            forecast_date = current_date + timedelta(days=day)
            
            try:
                # Prepare features for this date
                features = self.prepare_prediction_data(df, city, current_date)
                
                # Make prediction
                predicted_category, confidence = self.predict_aqi_category(features)
                
                forecasts.append({
                    'City': city,
                    'Date': forecast_date,
                    'Predicted_AQI_Category': predicted_category,
                    'Confidence': max(confidence.values()),
                    'All_Probabilities': confidence
                })
                
            except Exception as e:
                print(f"Error forecasting for {city} on {forecast_date}: {e}")
                forecasts.append({
                    'City': city,
                    'Date': forecast_date,
                    'Predicted_AQI_Category': 'Unknown',
                    'Confidence': 0.0,
                    'All_Probabilities': {}
                })
        
        return forecasts
    
    def get_city_risk_ranking(self, df, cities, date):
        """Rank cities by predicted air quality risk"""
        risk_scores = []
        
        aqi_risk_weights = {
            'Good': 1,
            'Moderate': 2,
            'Unhealthy for Sensitive Groups': 3,
            'Unhealthy': 4,
            'Very Unhealthy': 5,
            'Hazardous': 6
        }
        
        for city in cities:
            try:
                features = self.prepare_prediction_data(df, city, date)
                predicted_category, confidence = self.predict_aqi_category(features)
                
                risk_score = aqi_risk_weights.get(predicted_category, 3) * confidence[predicted_category]
                
                risk_scores.append({
                    'City': city,
                    'Predicted_Category': predicted_category,
                    'Risk_Score': risk_score,
                    'Confidence': confidence[predicted_category]
                })
                
            except Exception as e:
                print(f"Error calculating risk for {city}: {e}")
                risk_scores.append({
                    'City': city,
                    'Predicted_Category': 'Unknown',
                    'Risk_Score': 0,
                    'Confidence': 0
                })
        
        # Sort by risk score (highest first)
        risk_scores.sort(key=lambda x: x['Risk_Score'], reverse=True)
        
        return risk_scores
    
    def generate_alerts(self, forecasts, alert_threshold='Unhealthy for Sensitive Groups'):
        """Generate air quality alerts"""
        alerts = []
        
        risk_levels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                      'Unhealthy', 'Very Unhealthy', 'Hazardous']
        threshold_index = risk_levels.index(alert_threshold)
        
        for forecast in forecasts:
            predicted_category = forecast['Predicted_AQI_Category']
            if predicted_category in risk_levels:
                predicted_index = risk_levels.index(predicted_category)
                
                if predicted_index >= threshold_index:
                    alert_level = 'HIGH' if predicted_index >= 4 else 'MEDIUM'
                    
                    alerts.append({
                        'City': forecast['City'],
                        'Date': forecast['Date'],
                        'Alert_Level': alert_level,
                        'Predicted_Category': predicted_category,
                        'Confidence': forecast['Confidence'],
                        'Message': self._get_alert_message(predicted_category, forecast['City'])
                    })
        
        return alerts
    
    def _get_alert_message(self, category, city):
        """Generate alert message based on AQI category"""
        messages = {
            'Unhealthy for Sensitive Groups': f"Air quality in {city} may be unhealthy for sensitive groups. Consider limiting outdoor activities if you have respiratory conditions.",
            'Unhealthy': f"Air quality in {city} is unhealthy. Everyone should limit prolonged outdoor exertion.",
            'Very Unhealthy': f"Air quality in {city} is very unhealthy. Avoid outdoor activities and keep windows closed.",
            'Hazardous': f"EMERGENCY: Air quality in {city} is hazardous. Stay indoors and avoid all outdoor activities."
        }
        
        return messages.get(category, f"Air quality alert for {city}: {category}")

def main():
    """Demo function for prediction module"""
    # Load sample data
    try:
        df = pd.read_csv('data/processed_air_quality_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print("Loaded processed data successfully")
    except FileNotFoundError:
        print("Processed data not found. Please run data processing first.")
        return
    
    # Try to load a trained model
    try:
        # Look for any trained model
        import os
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        if not model_files:
            print("No trained models found. Please run model training first.")
            return
        
        model_path = f'models/{model_files[0]}'
        forecaster = AQIForecastor(model_path)
        
        # Demo predictions
        cities = df['City'].unique()[:3]  # First 3 cities
        latest_date = df['Date'].max()
        
        print(f"\nGenerating forecasts for {latest_date.date()}...")
        
        all_forecasts = []
        for city in cities:
            forecasts = forecaster.forecast_multiple_days(df, city, latest_date, days=3)
            all_forecasts.extend(forecasts)
            
            print(f"\n{city} Forecast:")
            for forecast in forecasts:
                print(f"  {forecast['Date'].date()}: {forecast['Predicted_AQI_Category']} "
                      f"(Confidence: {forecast['Confidence']:.2f})")
        
        # Generate alerts
        alerts = forecaster.generate_alerts(all_forecasts)
        if alerts:
            print(f"\nAIR QUALITY ALERTS ({len(alerts)} alerts):")
            for alert in alerts:
                print(f"  {alert['Alert_Level']}: {alert['City']} - {alert['Date'].date()}")
                print(f"    {alert['Message']}")
        else:
            print("\nNo air quality alerts for the forecast period.")
        
        # City risk ranking
        risk_ranking = forecaster.get_city_risk_ranking(df, cities, latest_date)
        print(f"\nCITY RISK RANKING for {latest_date.date()}:")
        for i, city_risk in enumerate(risk_ranking, 1):
            print(f"  {i}. {city_risk['City']}: {city_risk['Predicted_Category']} "
                  f"(Risk Score: {city_risk['Risk_Score']:.2f})")
        
    except Exception as e:
        print(f"Error in prediction demo: {e}")

if __name__ == "__main__":
    main()