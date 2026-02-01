"""
Test the probability chart creation to debug the error
"""

import sys
sys.path.append('src')
import plotly.graph_objects as go
from realtime_predictor import RealTimeAQIPredictor
import os

def get_aqi_color(category):
    """Get color for AQI category"""
    colors = {
        'Good': '#4caf50',
        'Moderate': '#ff9800',
        'Unhealthy for Sensitive Groups': '#ff5722',
        'Unhealthy': '#f44336',
        'Very Unhealthy': '#9c27b0',
        'Hazardous': '#795548'
    }
    return colors.get(category, '#757575')

def create_probability_chart(probabilities):
    """Create probability distribution chart"""
    categories = list(probabilities.keys())
    probs = [probabilities[cat] * 100 for cat in categories]
    colors = [get_aqi_color(cat) for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Category Probabilities",
        xaxis_title="AQI Category",
        yaxis_title="Probability (%)",
        height=400,
        showlegend=False,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def test_chart_creation():
    """Test creating the probability chart"""
    print("Testing probability chart creation...")
    
    # Load model and make prediction
    model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No trained models found")
        return False
    
    model_path = f'models/{model_files[0]}'
    predictor = RealTimeAQIPredictor(model_path)
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return False
    
    # Test data
    test_data = {
        "PM2.5": 35.0,
        "PM10": 65.0,
        "Temperature": 28.0,
        "Humidity": 65.0,
        "Wind_Speed": 5.0
    }
    
    print("Making prediction...")
    result = predictor.predict_from_input(test_data)
    
    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
        return False
    
    print("✅ Prediction successful")
    print(f"Probabilities: {result['all_probabilities']}")
    
    # Test chart creation
    try:
        print("Creating probability chart...")
        fig = create_probability_chart(result['all_probabilities'])
        print("✅ Chart created successfully")
        
        # Test if we can get the figure data
        print(f"Chart has {len(fig.data)} traces")
        print(f"Chart layout title: {fig.layout.title.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chart creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chart_creation()
    if success:
        print("\n🎉 Chart creation test passed!")
    else:
        print("\n❌ Chart creation test failed.")