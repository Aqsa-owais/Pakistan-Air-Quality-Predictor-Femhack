"""
Streamlit App for Air Quality Prediction
Interactive dashboard for AQI forecasting in Pakistani cities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import sys

# Add src to path
sys.path.append('src')

from prediction import AQIForecastor

# Page configuration
st.set_page_config(
    page_title="Pakistan Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .good-quality {
        color: #4caf50;
        font-weight: bold;
    }
    .moderate-quality {
        color: #ff9800;
        font-weight: bold;
    }
    .unhealthy-quality {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed air quality data"""
    try:
        df = pd.read_csv('data/processed_air_quality_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data processing first.")
        return None

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        if not model_files:
            st.error("No trained models found. Please run model training first.")
            return None
        
        model_path = f'models/{model_files[0]}'
        forecaster = AQIForecastor(model_path)
        return forecaster
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

def create_forecast_chart(forecasts):
    """Create forecast visualization"""
    df_forecast = pd.DataFrame(forecasts)
    
    fig = go.Figure()
    
    for city in df_forecast['City'].unique():
        city_data = df_forecast[df_forecast['City'] == city]
        
        fig.add_trace(go.Scatter(
            x=city_data['Date'],
            y=city_data['City'],
            mode='markers+text',
            marker=dict(
                size=20,
                color=[get_aqi_color(cat) for cat in city_data['Predicted_AQI_Category']]
            ),
            text=city_data['Predicted_AQI_Category'],
            textposition="middle center",
            name=city,
            hovertemplate='<b>%{y}</b><br>Date: %{x}<br>AQI: %{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title="3-Day Air Quality Forecast",
        xaxis_title="Date",
        yaxis_title="City",
        height=400,
        showlegend=False
    )
    
    return fig

def create_historical_chart(df, selected_city):
    """Create historical AQI chart"""
    city_data = df[df['City'] == selected_city].tail(30)  # Last 30 days
    
    fig = px.line(
        city_data, 
        x='Date', 
        y='AQI',
        title=f'Historical AQI Trend - {selected_city} (Last 30 Days)',
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add AQI category zones
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy for Sensitive")
    fig.add_hline(y=200, line_dash="dash", line_color="darkred", annotation_text="Unhealthy")
    
    fig.update_layout(height=400)
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌫️ Pakistan Air Quality Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time air quality forecasting for major Pakistani cities**")
    
    # Load data and model
    df = load_data()
    forecaster = load_model()
    
    if df is None or forecaster is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # City selection
    cities = sorted(df['City'].unique())
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    # Forecast days
    forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
    
    # Date selection
    max_date = df['Date'].max()
    selected_date = st.sidebar.date_input(
        "Forecast From Date",
        value=max_date.date(),
        max_value=max_date.date()
    )
    
    # Convert to datetime
    forecast_date = pd.to_datetime(selected_date)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📊 Air Quality Forecast - {selected_city}")
        
        # Generate forecast
        try:
            forecasts = forecaster.forecast_multiple_days(
                df, selected_city, forecast_date, days=forecast_days
            )
            
            # Display forecast cards
            forecast_cols = st.columns(min(len(forecasts), 3))
            
            for i, forecast in enumerate(forecasts[:3]):  # Show max 3 days in cards
                with forecast_cols[i]:
                    category = forecast['Predicted_AQI_Category']
                    confidence = forecast['Confidence']
                    date_str = forecast['Date'].strftime('%b %d')
                    
                    color_class = "good-quality" if "Good" in category else \
                                 "moderate-quality" if "Moderate" in category else \
                                 "unhealthy-quality"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{date_str}</h4>
                        <p class="{color_class}">{category}</p>
                        <small>Confidence: {confidence:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Forecast table
            if len(forecasts) > 3:
                st.subheader("Extended Forecast")
                forecast_df = pd.DataFrame(forecasts)
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
                forecast_df['Confidence'] = forecast_df['Confidence'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    forecast_df[['Date', 'Predicted_AQI_Category', 'Confidence']],
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            forecasts = []
    
    with col2:
        st.header("⚠️ Alerts & Warnings")
        
        # Generate alerts
        if forecasts:
            alerts = forecaster.generate_alerts(forecasts)
            
            if alerts:
                for alert in alerts:
                    alert_class = "alert-high" if alert['Alert_Level'] == 'HIGH' else "alert-medium"
                    
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <strong>{alert['Alert_Level']} ALERT</strong><br>
                        {alert['Date'].strftime('%b %d')}: {alert['Predicted_Category']}<br>
                        <small>{alert['Message']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.success("✅ No air quality alerts for the forecast period")
        
        # Current conditions
        st.subheader("📈 Current Conditions")
        latest_data = df[(df['City'] == selected_city)].tail(1)
        
        if not latest_data.empty:
            current_aqi = latest_data['AQI'].iloc[0]
            current_category = latest_data['AQI_Category'].iloc[0]
            
            st.metric("Current AQI", f"{current_aqi:.0f}", delta=None)
            st.metric("Category", current_category)
            
            # Weather info
            if 'Temperature' in latest_data.columns:
                temp = latest_data['Temperature'].iloc[0]
                humidity = latest_data['Humidity'].iloc[0]
                wind = latest_data['Wind_Speed'].iloc[0]
                
                st.metric("Temperature", f"{temp:.1f}°C")
                st.metric("Humidity", f"{humidity:.0f}%")
                st.metric("Wind Speed", f"{wind:.1f} km/h")
    
    # Historical trends
    st.header("📈 Historical Trends")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Historical chart
        hist_chart = create_historical_chart(df, selected_city)
        st.plotly_chart(hist_chart, use_container_width=True)
    
    with col4:
        # City comparison
        st.subheader("🏙️ City Risk Ranking")
        
        try:
            risk_ranking = forecaster.get_city_risk_ranking(df, cities, forecast_date)
            
            ranking_data = []
            for i, city_risk in enumerate(risk_ranking, 1):
                ranking_data.append({
                    'Rank': i,
                    'City': city_risk['City'],
                    'Predicted Category': city_risk['Predicted_Category'],
                    'Risk Score': f"{city_risk['Risk_Score']:.2f}"
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error generating city ranking: {e}")
    
    # Multi-city forecast visualization
    if len(cities) > 1:
        st.header("🗺️ Multi-City Forecast")
        
        try:
            all_forecasts = []
            for city in cities[:5]:  # Limit to 5 cities for performance
                city_forecasts = forecaster.forecast_multiple_days(
                    df, city, forecast_date, days=3
                )
                all_forecasts.extend(city_forecasts)
            
            if all_forecasts:
                multi_chart = create_forecast_chart(all_forecasts)
                st.plotly_chart(multi_chart, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating multi-city forecast: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** This air quality prediction system uses machine learning to forecast AQI levels 
    for major Pakistani cities. Predictions are based on historical air quality and weather data.
    
    **Disclaimer:** This is a demonstration system. For official air quality information, 
    please consult local environmental authorities.
    """)

if __name__ == "__main__":
    main()