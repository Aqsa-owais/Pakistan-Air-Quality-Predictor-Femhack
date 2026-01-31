"""
Feature Engineering Module for Air Quality Prediction
Creates lag features, rolling averages, and other time-series features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_lag_features(self, df, columns, lags=[1, 2, 3, 7]):
        """Create lag features for specified columns"""
        print("Creating lag features...")
        
        df_features = df.copy()
        
        for city in df['City'].unique():
            city_mask = df_features['City'] == city
            city_data = df_features[city_mask].copy()
            
            for col in columns:
                if col in city_data.columns:
                    for lag in lags:
                        lag_col = f'{col}_lag_{lag}'
                        city_data[lag_col] = city_data[col].shift(lag)
            
            df_features.loc[city_mask] = city_data
        
        return df_features
    
    def create_rolling_features(self, df, columns, windows=[3, 7, 14]):
        """Create rolling average features"""
        print("Creating rolling features...")
        
        df_features = df.copy()
        
        for city in df['City'].unique():
            city_mask = df_features['City'] == city
            city_data = df_features[city_mask].copy()
            
            for col in columns:
                if col in city_data.columns:
                    for window in windows:
                        # Rolling mean
                        roll_mean_col = f'{col}_roll_mean_{window}'
                        city_data[roll_mean_col] = city_data[col].rolling(window=window, min_periods=1).mean()
                        
                        # Rolling std
                        roll_std_col = f'{col}_roll_std_{window}'
                        city_data[roll_std_col] = city_data[col].rolling(window=window, min_periods=1).std()
                        
                        # Rolling max
                        roll_max_col = f'{col}_roll_max_{window}'
                        city_data[roll_max_col] = city_data[col].rolling(window=window, min_periods=1).max()
            
            df_features.loc[city_mask] = city_data
        
        return df_features
    
    def create_seasonal_features(self, df):
        """Create seasonal and cyclical features"""
        print("Creating seasonal features...")
        
        df_features = df.copy()
        
        # Cyclical encoding for time features
        df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
        df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
        
        df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
        df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
        
        df_features['DayOfYear_sin'] = np.sin(2 * np.pi * df_features['DayOfYear'] / 365)
        df_features['DayOfYear_cos'] = np.cos(2 * np.pi * df_features['DayOfYear'] / 365)
        
        # Season indicator
        df_features['Season'] = df_features['Month'].apply(self._get_season)
        
        # Weekend indicator
        df_features['IsWeekend'] = (df_features['DayOfWeek'] >= 5).astype(int)
        
        return df_features
    
    def _get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def create_interaction_features(self, df):
        """Create interaction features between weather and pollution"""
        print("Creating interaction features...")
        
        df_features = df.copy()
        
        # Weather interactions
        if 'Temperature' in df_features.columns and 'Humidity' in df_features.columns:
            df_features['Temp_Humidity'] = df_features['Temperature'] * df_features['Humidity']
        
        if 'Wind_Speed' in df_features.columns and 'Pressure' in df_features.columns:
            df_features['Wind_Pressure'] = df_features['Wind_Speed'] * df_features['Pressure']
        
        # Pollution ratios
        if 'PM2.5' in df_features.columns and 'PM10' in df_features.columns:
            df_features['PM_Ratio'] = df_features['PM2.5'] / (df_features['PM10'] + 1e-6)
        
        return df_features
    
    def create_target_features(self, df, target_col='AQI', forecast_days=3):
        """Create target variables for multi-day forecasting"""
        print(f"Creating target features for {forecast_days} days ahead...")
        
        df_features = df.copy()
        
        for city in df['City'].unique():
            city_mask = df_features['City'] == city
            city_data = df_features[city_mask].copy()
            
            for day in range(1, forecast_days + 1):
                target_col_name = f'{target_col}_target_{day}d'
                city_data[target_col_name] = city_data[target_col].shift(-day)
                
                # Also create category targets
                if f'{target_col}_Category' in city_data.columns:
                    cat_target_col = f'{target_col}_Category_target_{day}d'
                    city_data[cat_target_col] = city_data[f'{target_col}_Category'].shift(-day)
            
            df_features.loc[city_mask] = city_data
        
        return df_features
    
    def prepare_features(self, df, target_col='AQI_Category'):
        """Prepare all features for modeling"""
        print("Preparing features for modeling...")
        
        # Create all feature types
        lag_columns = ['AQI', 'PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']
        df_features = self.create_lag_features(df, lag_columns)
        
        rolling_columns = ['AQI', 'PM2.5', 'PM10', 'Temperature', 'Humidity']
        df_features = self.create_rolling_features(df_features, rolling_columns)
        
        df_features = self.create_seasonal_features(df_features)
        df_features = self.create_interaction_features(df_features)
        df_features = self.create_target_features(df_features)
        
        # Encode categorical variables
        if 'City' in df_features.columns:
            df_features['City_encoded'] = self.label_encoder.fit_transform(df_features['City'])
        
        if 'Season' in df_features.columns:
            season_encoder = LabelEncoder()
            df_features['Season_encoded'] = season_encoder.fit_transform(df_features['Season'])
        
        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ['Date', 'City', 'Season', 'AQI_Category', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
        exclude_cols.extend([col for col in df_features.columns if 'target' in col])
        
        # Only include columns that actually exist in the dataframe
        self.feature_columns = [col for col in df_features.columns 
                               if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col])]
        
        print(f"Created {len(self.feature_columns)} features")
        
        return df_features
    
    def get_model_data(self, df, target_col='AQI_Category_target_1d', test_size=0.2):
        """Prepare data for model training"""
        print("Preparing model data...")
        
        # Check if target column exists, if not use current AQI_Category
        if target_col not in df.columns:
            print(f"Target column {target_col} not found, using AQI_Category instead")
            target_col = 'AQI_Category'
        
        # Remove rows with missing targets
        df_model = df.dropna(subset=[target_col]).copy()
        
        # Features and target
        X = df_model[self.feature_columns]
        y = df_model[target_col]
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Split by time (last 20% for testing)
        split_date = df_model['Date'].quantile(1 - test_size)
        train_mask = df_model['Date'] < split_date
        
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df_model[~train_mask]

def main():
    """Main function to demonstrate feature engineering"""
    # Load processed data
    df = pd.read_csv('data/processed_air_quality_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create features
    df_features = fe.prepare_features(df)
    
    # Save featured data
    df_features.to_csv('data/featured_air_quality_data.csv', index=False)
    print("Featured data saved to data/featured_air_quality_data.csv")
    
    # Prepare model data
    X_train, X_test, y_train, y_test, test_df = fe.get_model_data(df_features)
    
    print(f"\nFeature columns ({len(fe.feature_columns)}):")
    for i, col in enumerate(fe.feature_columns[:10]):  # Show first 10
        print(f"  {i+1}. {col}")
    if len(fe.feature_columns) > 10:
        print(f"  ... and {len(fe.feature_columns) - 10} more")
    
    return df_features, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df_features, X_train, X_test, y_train, y_test = main()