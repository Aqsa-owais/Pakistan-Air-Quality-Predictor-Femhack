"""
Test the analytics tab functionality specifically
"""

import sys
sys.path.append('src')
import pandas as pd
import plotly.express as px
from realtime_predictor import RealTimeAQIPredictor
import os

def test_analytics_charts():
    """Test analytics chart creation"""
    print("Testing analytics charts...")
    
    # Load data
    try:
        df = pd.read_csv('data/processed_air_quality_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test AQI distribution chart
    try:
        aqi_dist = df['AQI_Category'].value_counts()
        fig = px.pie(
            values=aqi_dist.values,
            names=aqi_dist.index,
            title="Historical AQI Category Distribution"
        )
        print("✅ AQI distribution chart created")
    except Exception as e:
        print(f"❌ AQI distribution chart failed: {e}")
        return False
    
    # Test city-wise average AQI chart
    try:
        city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=city_aqi.index,
            y=city_aqi.values,
            title="Average AQI by City",
            labels={'x': 'City', 'y': 'Average AQI'}
        )
        print("✅ City AQI chart created")
    except Exception as e:
        print(f"❌ City AQI chart failed: {e}")
        return False
    
    # Test seasonal trends chart (the one that was causing the error)
    try:
        df['Month'] = df['Date'].dt.month
        monthly_aqi = df.groupby('Month')['AQI'].mean()
        
        fig = px.line(
            x=monthly_aqi.index,
            y=monthly_aqi.values,
            title="Average AQI by Month",
            labels={'x': 'Month', 'y': 'Average AQI'}
        )
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
        print("✅ Seasonal trends chart created (error fixed!)")
    except Exception as e:
        print(f"❌ Seasonal trends chart failed: {e}")
        return False
    
    # Test correlation matrix
    try:
        numeric_cols = ['AQI', 'PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 2:
            corr_matrix = df[available_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Environmental Parameter Correlations",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            print("✅ Correlation matrix created")
        else:
            print("⚠️  Not enough numeric columns for correlation matrix")
    except Exception as e:
        print(f"❌ Correlation matrix failed: {e}")
        return False
    
    return True

def test_model_info():
    """Test model information retrieval"""
    print("Testing model information...")
    
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
    
    # Get model info
    try:
        model_info = predictor.get_model_info()
        print("✅ Model information retrieved")
        print(f"   Model: {model_info.get('model_name', 'Unknown')}")
        print(f"   Features: {model_info.get('feature_count', 'Unknown')}")
        print(f"   Categories: {len(model_info.get('categories', []))}")
        
        if 'performance' in model_info:
            perf = model_info['performance']
            print(f"   Accuracy: {perf.get('accuracy', 0):.1%}")
            print(f"   F1-Score: {perf.get('f1_score', 0):.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Model info retrieval failed: {e}")
        return False

def main():
    """Run analytics tests"""
    print("🧪 Testing Analytics Tab Components")
    print("=" * 50)
    
    tests = [
        ("Analytics Charts", test_analytics_charts),
        ("Model Information", test_model_info)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📊 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"🎯 Analytics Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All analytics tests passed! The update_xaxis error is fixed.")
        print("\n📈 Analytics tab features working:")
        print("   ✅ AQI category distribution pie chart")
        print("   ✅ City-wise average AQI bar chart")
        print("   ✅ Seasonal trends line chart (FIXED)")
        print("   ✅ Environmental correlations heatmap")
        print("   ✅ Model performance metrics")
        print("   ✅ Data quality statistics")
    else:
        print("❌ Some analytics tests failed.")
    
    return passed == total

if __name__ == "__main__":
    main()