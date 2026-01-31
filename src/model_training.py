"""
Model Training Module for Air Quality Prediction
Trains and evaluates multiple ML models for AQI category prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer

class AQIPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_engineer = None
        self.model_performance = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train_encoded)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels, average='weighted')
        
        self.models['random_forest'] = rf_model
        self.model_performance['random_forest'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred_labels
        }
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        return rf_model, y_pred_labels
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(X_train, y_train_encoded)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels, average='weighted')
        
        self.models['xgboost'] = xgb_model
        self.model_performance['xgboost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred_labels
        }
        
        print(f"XGBoost - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        return xgb_model, y_pred_labels
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        print(f"\n{model_name} Evaluation:")
        print("="*50)
        
        # Classification report
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'outputs/{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='weighted')
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'{model_name} - Top {top_n} Feature Importance')
            plt.bar(range(top_n), importance[indices])
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'outputs/{model_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def select_best_model(self):
        """Select the best performing model"""
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['f1_score'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"F1-Score: {self.model_performance[best_model_name]['f1_score']:.4f}")
        print(f"Accuracy: {self.model_performance[best_model_name]['accuracy']:.4f}")
        
        return best_model, best_model_name
    
    def save_model(self, model, model_name, feature_engineer):
        """Save trained model and preprocessing objects"""
        model_data = {
            'model': model,
            'label_encoder': self.label_encoder,
            'feature_engineer': feature_engineer,
            'feature_columns': feature_engineer.feature_columns,
            'model_name': model_name,
            'performance': self.model_performance[model_name],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f'models/aqi_predictor_{model_name}.joblib'
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
        
        return filename
    
    def create_predictions_output(self, test_df, predictions, model_name):
        """Create prediction output CSV"""
        output_df = test_df[['City', 'Date']].copy()
        output_df['Predicted_AQI_Category'] = predictions
        output_df['Model'] = model_name
        
        # Add forecast for next 3 days for each city
        forecast_data = []
        for city in output_df['City'].unique():
            city_data = output_df[output_df['City'] == city].tail(1)
            last_date = city_data['Date'].iloc[0]
            
            for i in range(1, 4):  # Next 3 days
                forecast_date = last_date + pd.Timedelta(days=i)
                # Use last prediction as forecast (in real scenario, use model prediction)
                forecast_category = city_data['Predicted_AQI_Category'].iloc[0]
                
                forecast_data.append({
                    'City': city,
                    'Date': forecast_date,
                    'Predicted_AQI_Category': forecast_category,
                    'Model': f"{model_name}_forecast"
                })
        
        forecast_df = pd.DataFrame(forecast_data)
        final_output = pd.concat([output_df, forecast_df], ignore_index=True)
        final_output = final_output.sort_values(['City', 'Date'])
        
        # Save predictions
        output_file = f'outputs/aqi_predictions_{model_name}.csv'
        final_output.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return final_output

def main():
    """Main function to train and evaluate models"""
    print("Starting model training pipeline...")
    
    # Load featured data
    try:
        df_features = pd.read_csv('data/featured_air_quality_data.csv')
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        print("Loaded featured data successfully")
    except FileNotFoundError:
        print("Featured data not found. Running feature engineering first...")
        # Run data processing and feature engineering
        from data_processing import main as process_data
        df = process_data()
        
        fe = FeatureEngineer()
        df_features = fe.prepare_features(df)
        df_features.to_csv('data/featured_air_quality_data.csv', index=False)
    
    # Initialize feature engineer and prepare data
    fe = FeatureEngineer()
    fe.feature_columns = [col for col in df_features.columns 
                         if col not in ['Date', 'City', 'Season', 'AQI_Category', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
                         and 'target' not in col]
    
    X_train, X_test, y_train, y_test, test_df = fe.get_model_data(df_features)
    
    # Initialize predictor
    predictor = AQIPredictor()
    predictor.feature_engineer = fe
    
    # Train models
    rf_model, rf_predictions = predictor.train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_predictions = predictor.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    predictor.evaluate_model(y_test, rf_predictions, 'Random Forest')
    predictor.evaluate_model(y_test, xgb_predictions, 'XGBoost')
    
    # Plot feature importance
    predictor.plot_feature_importance(rf_model, fe.feature_columns, 'Random Forest')
    predictor.plot_feature_importance(xgb_model, fe.feature_columns, 'XGBoost')
    
    # Select best model
    best_model, best_model_name = predictor.select_best_model()
    
    # Save best model
    predictor.save_model(best_model, best_model_name, fe)
    
    # Create prediction outputs
    best_predictions = predictor.model_performance[best_model_name]['predictions']
    predictor.create_predictions_output(test_df, best_predictions, best_model_name)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    for model_name, performance in predictor.model_performance.items():
        print(f"{model_name:15} - Accuracy: {performance['accuracy']:.4f}, F1-Score: {performance['f1_score']:.4f}")
    
    print(f"\nBest Model: {best_model_name}")
    print("Files created:")
    print("- Model: models/aqi_predictor_{}.joblib".format(best_model_name))
    print("- Predictions: outputs/aqi_predictions_{}.csv".format(best_model_name))
    print("- Confusion Matrix: outputs/{}_confusion_matrix.png".format(best_model_name.lower()))
    print("- Feature Importance: outputs/{}_feature_importance.png".format(best_model_name.lower()))
    
    return predictor, best_model, best_model_name

if __name__ == "__main__":
    predictor, model, model_name = main()