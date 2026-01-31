"""
Data Processing Module for Air Quality Prediction
Handles data loading, cleaning, and preprocessing for Pakistani cities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AirQualityDataProcessor:
    def __init__(self):
        self.aqi_categories = {
            'Good': (0, 50),
            'Moderate': (51, 100),
            'Unhealthy for Sensitive Groups': (101, 150),
            'Unhealthy': (151, 200),
            'Very Unhealthy': (201, 300),
            'Hazardous': (301, 500)
        }
        
    def create_sample_data(self):
        """Create sample air quality data for Pakistani cities"""
        cities = ['Lahore', 'Karachi', 'Islamabad', 'Faisalabad', 'Rawalpindi']
        
        # Generate 3 years of daily data
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2024, 1, 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        
        for city in cities:
            for date in date_range:
                # Simulate seasonal patterns and city-specific pollution levels
                base_aqi = self._get_base_aqi(city, date)
                
                # Add weather factors
                temp = np.random.normal(25 + 10 * np.sin(2 * np.pi * date.dayofyear / 365), 8)
                humidity = np.random.normal(60, 15)
                wind_speed = np.random.exponential(5)
                pressure = np.random.normal(1013, 10)
                
                # Weather impact on AQI
                weather_factor = 1.0
                if wind_speed < 2:  # Low wind increases pollution
                    weather_factor *= 1.3
                if humidity > 80:  # High humidity can trap pollutants
                    weather_factor *= 1.1
                if temp > 35:  # High temperature increases ozone
                    weather_factor *= 1.2
                    
                aqi = max(0, min(500, base_aqi * weather_factor + np.random.normal(0, 10)))
                
                # Individual pollutants
                pm25 = aqi * 0.4 + np.random.normal(0, 5)
                pm10 = pm25 * 1.5 + np.random.normal(0, 8)
                o3 = aqi * 0.3 + np.random.normal(0, 3)
                no2 = aqi * 0.2 + np.random.normal(0, 2)
                so2 = aqi * 0.1 + np.random.normal(0, 1)
                co = aqi * 0.05 + np.random.normal(0, 0.5)
                
                data.append({
                    'Date': date,
                    'City': city,
                    'AQI': round(aqi, 1),
                    'PM2.5': max(0, round(pm25, 1)),
                    'PM10': max(0, round(pm10, 1)),
                    'O3': max(0, round(o3, 1)),
                    'NO2': max(0, round(no2, 1)),
                    'SO2': max(0, round(so2, 1)),
                    'CO': max(0, round(co, 2)),
                    'Temperature': round(temp, 1),
                    'Humidity': max(0, min(100, round(humidity, 1))),
                    'Wind_Speed': max(0, round(wind_speed, 1)),
                    'Pressure': round(pressure, 1)
                })
        
        return pd.DataFrame(data)
    
    def _get_base_aqi(self, city, date):
        """Get base AQI for city with seasonal patterns"""
        city_base = {
            'Lahore': 120,      # High pollution
            'Karachi': 100,     # Moderate-high
            'Islamabad': 80,    # Moderate
            'Faisalabad': 110,  # High
            'Rawalpindi': 90    # Moderate
        }
        
        # Seasonal variation (winter worse in Pakistan)
        seasonal_factor = 1.0
        if date.month in [11, 12, 1, 2]:  # Winter months
            seasonal_factor = 1.4
        elif date.month in [6, 7, 8]:    # Monsoon months
            seasonal_factor = 0.8
        
        # Weekly pattern (weekends slightly better)
        weekly_factor = 0.9 if date.weekday() >= 5 else 1.0
        
        return city_base[city] * seasonal_factor * weekly_factor
    
    def categorize_aqi(self, aqi_value):
        """Convert AQI value to category"""
        for category, (min_val, max_val) in self.aqi_categories.items():
            if min_val <= aqi_value <= max_val:
                return category
        return 'Hazardous'  # For values > 500
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by city and date
        df = df.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Handle missing values
        numeric_columns = ['AQI', 'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 
                          'Temperature', 'Humidity', 'Wind_Speed', 'Pressure']
        
        for col in numeric_columns:
            if col in df.columns:
                # Forward fill then backward fill
                df[col] = df.groupby('City')[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (values beyond reasonable ranges)
        df.loc[df['AQI'] > 500, 'AQI'] = 500
        df.loc[df['AQI'] < 0, 'AQI'] = 0
        
        # Add AQI category
        df['AQI_Category'] = df['AQI'].apply(self.categorize_aqi)
        
        # Add time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        print(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def save_processed_data(self, df, filename='processed_air_quality_data.csv'):
        """Save processed data"""
        filepath = f'data/{filename}'
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

def main():
    """Main function to process air quality data"""
    processor = AirQualityDataProcessor()
    
    print("Creating sample air quality data for Pakistani cities...")
    df = processor.create_sample_data()
    
    print("Processing data...")
    df_clean = processor.clean_data(df)
    
    # Save processed data
    processor.save_processed_data(df_clean)
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    print(f"Cities: {df_clean['City'].unique()}")
    print(f"Total records: {len(df_clean)}")
    print("\nAQI Category Distribution:")
    print(df_clean['AQI_Category'].value_counts())
    
    return df_clean

if __name__ == "__main__":
    df = main()