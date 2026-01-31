"""
Complete Pipeline Runner for Air Quality Prediction
Runs the entire ML pipeline from data processing to model training
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def run_complete_pipeline():
    """Run the complete air quality prediction pipeline"""
    
    print("🌫️  PAKISTAN AIR QUALITY PREDICTION PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Data Processing
        print("\n📊 Step 1: Data Processing and Cleaning")
        print("-" * 40)
        
        from data_processing import main as process_data
        df_processed = process_data()
        
        print("✅ Data processing completed successfully!")
        
        # Step 2: Feature Engineering
        print("\n🔧 Step 2: Feature Engineering")
        print("-" * 40)
        
        from feature_engineering import main as engineer_features
        df_features, X_train, X_test, y_train, y_test = engineer_features()
        
        print("✅ Feature engineering completed successfully!")
        
        # Step 3: Model Training
        print("\n🤖 Step 3: Model Training and Evaluation")
        print("-" * 40)
        
        from model_training import main as train_models
        predictor, best_model, best_model_name = train_models()
        
        print("✅ Model training completed successfully!")
        
        # Step 4: Generate Predictions
        print("\n🔮 Step 4: Generating Predictions and Forecasts")
        print("-" * 40)
        
        from prediction import main as generate_predictions
        generate_predictions()
        
        print("✅ Predictions generated successfully!")
        
        # Final Summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n📈 RESULTS SUMMARY:")
        print(f"   • Best Model: {best_model_name}")
        print(f"   • Model Accuracy: {predictor.model_performance[best_model_name]['accuracy']:.3f}")
        print(f"   • Model F1-Score: {predictor.model_performance[best_model_name]['f1_score']:.3f}")
        
        print(f"\n📁 FILES CREATED:")
        print(f"   • Data: data/processed_air_quality_data.csv")
        print(f"   • Features: data/featured_air_quality_data.csv")
        print(f"   • Model: models/aqi_predictor_{best_model_name}.joblib")
        print(f"   • Predictions: outputs/aqi_predictions_{best_model_name}.csv")
        print(f"   • Visualizations: outputs/*.png")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Install dependencies: pip install -r requirements.txt")
        print(f"   2. Launch Streamlit app: streamlit run app.py")
        print(f"   3. Open Jupyter notebook: jupyter notebook notebooks/air_quality_analysis.ipynb")
        
        print(f"\n💡 STREAMLIT APP FEATURES:")
        print(f"   • Interactive city selection")
        print(f"   • 3-day air quality forecasts")
        print(f"   • Real-time alerts and warnings")
        print(f"   • Historical trend analysis")
        print(f"   • Multi-city risk ranking")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print(f"Please check the error and try again.")
        return False

def check_dependencies():
    """Check if required directories exist"""
    required_dirs = ['data', 'models', 'outputs', 'src', 'notebooks']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Creating directory: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)

if __name__ == "__main__":
    print("Checking dependencies...")
    check_dependencies()
    
    print("\nStarting pipeline...")
    success = run_complete_pipeline()
    
    if success:
        print("\n🎊 All done! Your air quality prediction system is ready to use.")
    else:
        print("\n😞 Pipeline failed. Please check the errors above.")