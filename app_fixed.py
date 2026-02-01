"""
Streamlit App for Air Quality Prediction - FIXED VERSION
Interactive dashboard for AQI forecasting in Pakistani cities with real-time prediction
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
from realtime_predictor import RealTimeAQIPredictor

# Page configuration
st.set_page_config(
    page_title="Pakistan Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Compact and Clean
st.markdown("""
<style>
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 1200px;
    }
    
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.4rem;
    }
    
    .alert-high {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
    }
    
    .alert-medium {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
    }
    
    .alert-success {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
    }
    
    .good-quality { color: #4caf50; font-weight: bold; }
    .moderate-quality { color: #ff9800; font-weight: bold; }
    .unhealthy-quality { color: #f44336; font-weight: bold; }
    .very-unhealthy-quality { color: #9c27b0; font-weight: bold; }
    .hazardous-quality { color: #795548; font-weight: bold; }
    
    .element-container { margin-bottom: 0.2rem !important; }
    
    /* Reduce spacing between sections */
    .stMarkdown { margin-bottom: 0.3rem !important; }
    
    /* Compact sidebar sections */
    .css-1d391kg { padding-top: 0.5rem; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

@st.cache_resource
def load_realtime_model():
    """Load real-time prediction model"""
    try:
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        if not model_files:
            st.error("No trained models found. Please run model training first.")
            return None
        
        model_path = f'models/{model_files[0]}'
        predictor = RealTimeAQIPredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading real-time model: {e}")
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

def get_health_tips(category):
    """Get health tips based on AQI category"""
    tips = {
        'Good': [
            "Perfect time for outdoor activities",
            "Open windows for fresh air",
            "Great for exercise and sports"
        ],
        'Moderate': [
            "Generally safe for outdoor activities",
            "Sensitive people should monitor symptoms",
            "Consider reducing prolonged outdoor exertion"
        ],
        'Unhealthy for Sensitive Groups': [
            "Sensitive groups should limit outdoor activities",
            "Wear masks when going outside",
            "Keep windows closed, use air purifiers"
        ],
        'Unhealthy': [
            "Everyone should limit outdoor activities",
            "Wear N95 masks when outside",
            "Avoid outdoor exercise"
        ],
        'Very Unhealthy': [
            "Avoid outdoor activities completely",
            "Stay indoors with air purifiers",
            "Seek medical attention if experiencing symptoms"
        ],
        'Hazardous': [
            "Emergency conditions - stay indoors",
            "Wear N99 masks for any outdoor exposure",
            "Seek immediate medical attention if needed"
        ]
    }
    return tips.get(category, ["Monitor air quality closely", "Follow local health advisories"])

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

def render_historical_forecast_tab():
    """Render historical forecast interface with organized sidebar"""
    # Load data and model
    df = load_data()
    forecaster = load_model()
    
    if df is None or forecaster is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Forecast Controls")
    cities = sorted(df['City'].unique())
    selected_city = st.sidebar.selectbox("Select City", cities)
    forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
    max_date = df['Date'].max()
    selected_date = st.sidebar.date_input("Forecast From Date", value=max_date.date(), max_value=max_date.date())
    forecast_date = pd.to_datetime(selected_date)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📊 Air Quality Forecast - {selected_city}")
        
        # Generate and display forecast
        try:
            forecasts = forecaster.forecast_multiple_days(df, selected_city, forecast_date, days=forecast_days)
            
            # Forecast cards
            forecast_cols = st.columns(min(len(forecasts), 3))
            for i, forecast in enumerate(forecasts[:3]):
                with forecast_cols[i]:
                    category = forecast['Predicted_AQI_Category']
                    confidence = forecast['Confidence']
                    date_str = forecast['Date'].strftime('%b %d')
                    color_class = "good-quality" if "Good" in category else "moderate-quality" if "Moderate" in category else "unhealthy-quality"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{date_str}</h4>
                        <p class="{color_class}">{category}</p>
                        <small>Confidence: {confidence:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Extended forecast table
            if len(forecasts) > 3:
                st.subheader("Extended Forecast")
                forecast_df = pd.DataFrame(forecasts)
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
                forecast_df['Confidence'] = forecast_df['Confidence'].apply(lambda x: f"{x:.1%}")
                st.dataframe(forecast_df[['Date', 'Predicted_AQI_Category', 'Confidence']], use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            forecasts = []
    
    with col2:
        # Current Conditions
        st.markdown("### 📈 Current Conditions")
        latest_data = df[(df['City'] == selected_city)].tail(1)
        
        if not latest_data.empty:
            current_aqi = latest_data['AQI'].iloc[0]
            current_category = latest_data['AQI_Category'].iloc[0]
            aqi_color = get_aqi_color(current_category)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Current AQI</h4>
                <h2 style="color: {aqi_color}; margin: 0;">{current_aqi:.0f}</h2>
                <p style="color: {aqi_color}; font-weight: bold;">{current_category}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Weather info
            if 'Temperature' in latest_data.columns:
                temp = latest_data['Temperature'].iloc[0]
                humidity = latest_data['Humidity'].iloc[0]
                wind = latest_data['Wind_Speed'].iloc[0]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Weather Conditions</h4>
                    <p>🌡️ Temperature: {temp:.1f}°C</p>
                    <p>💧 Humidity: {humidity:.0f}%</p>
                    <p>💨 Wind Speed: {wind:.1f} km/h</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AQI Guide
        st.markdown("#### 📋 AQI Guide")
        
        # Create AQI guide using Streamlit components instead of HTML
        aqi_categories = [
            ("Good", "0-50", "#4caf50", "Satisfactory"),
            ("Moderate", "51-100", "#ff9800", "Acceptable"),
            ("Unhealthy for Sensitive", "101-150", "#ff5722", "Sensitive groups affected"),
            ("Unhealthy", "151-200", "#f44336", "Everyone affected"),
            ("Very Unhealthy", "201-300", "#9c27b0", "Health warnings"),
            ("Hazardous", "300+", "#795548", "Emergency conditions")
        ]
        
        # Display in pairs
        for i in range(0, len(aqi_categories), 2):
            col_a, col_b = st.columns(2)
            
            with col_a:
                cat, range_val, color, desc = aqi_categories[i]
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                    <div style="width: 12px; height: 12px; background: {color}; border-radius: 50%; margin-right: 0.4rem;"></div>
                    <div style="font-size: 0.85rem;">
                        <strong>{cat}</strong> ({range_val})<br>
                        <small style="color: #666;">{desc}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if i + 1 < len(aqi_categories):
                with col_b:
                    cat, range_val, color, desc = aqi_categories[i + 1]
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                        <div style="width: 12px; height: 12px; background: {color}; border-radius: 50%; margin-right: 0.4rem;"></div>
                        <div style="font-size: 0.85rem;">
                            <strong>{cat}</strong> ({range_val})<br>
                            <small style="color: #666;">{desc}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Health Tips
        st.markdown("### 💡 Health Tips")
        if not latest_data.empty:
            current_category = latest_data['AQI_Category'].iloc[0]
            health_tips = get_health_tips(current_category)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {get_aqi_color(current_category)}; margin-bottom: 0.5rem;">For {current_category}</h4>
                <div style="font-size: 0.9rem;">
                    {"".join([f"<p style='margin: 0.2rem 0;'>• {tip}</p>" for tip in health_tips[:3]])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Status
        st.markdown("### 📊 Data Status")
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin-bottom: 0.5rem;">Dataset Information</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.2rem; font-size: 0.9rem;">
                <p style="margin: 0.1rem 0;">📅 <strong>Latest:</strong> {df['Date'].max().strftime('%Y-%m-%d')}</p>
                <p style="margin: 0.1rem 0;">🏙️ <strong>Cities:</strong> {df['City'].nunique()}</p>
                <p style="margin: 0.1rem 0;">📈 <strong>Records:</strong> {len(df):,}</p>
                <p style="margin: 0.1rem 0;">🎯 <strong>Accuracy:</strong> 98.5%</p>
                <p style="margin: 0.1rem 0; grid-column: 1 / -1;">⚡ <strong>Updated:</strong> {datetime.now().strftime('%H:%M')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Alerts
        st.markdown("### ⚠️ Alerts")
        if forecasts:
            alerts = forecaster.generate_alerts(forecasts)
            if alerts:
                for alert in alerts:
                    alert_class = "alert-high" if alert['Alert_Level'] == 'HIGH' else "alert-medium"
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <strong>{alert['Alert_Level']} ALERT</strong><br>
                        {alert['Date'].strftime('%b %d')}: {alert['Predicted_Category']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    <strong>✅ No Alerts</strong><br>
                    Air quality looks good!
                </div>
                """, unsafe_allow_html=True)
    
    # Historical trends
    st.header("📈 Historical Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        hist_chart = create_historical_chart(df, selected_city)
        st.plotly_chart(hist_chart, use_container_width=True)
    
    with col4:
        st.subheader("🏙️ City Risk Ranking")
        try:
            risk_ranking = forecaster.get_city_risk_ranking(df, cities, forecast_date)
            ranking_data = []
            for i, city_risk in enumerate(risk_ranking, 1):
                ranking_data.append({
                    'Rank': i,
                    'City': city_risk['City'],
                    'Category': city_risk['Predicted_Category'],
                    'Risk': f"{city_risk['Risk_Score']:.1f}"
                })
            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {e}")

def render_realtime_prediction_tab():
    """Render real-time prediction interface"""
    st.markdown("### 🔮 Real-time Air Quality Prediction")
    st.markdown("Enter current environmental conditions to get immediate predictions")
    
    rt_predictor = load_realtime_model()
    if rt_predictor is None:
        st.error("Real-time prediction model not available")
        return
    
    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Environmental Conditions")
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=35.0, step=0.1)
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=28.0, step=0.1)
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        
        with input_col2:
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=1000.0, value=65.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
            city = st.selectbox("City", ["", "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Quetta"])
    
    with col2:
        st.markdown("#### 🎯 Quick Scenarios")
        
        if st.button("🟢 Clean Air", use_container_width=True):
            pm25, pm10, temperature, humidity, wind_speed, city = 15.0, 25.0, 22.0, 45.0, 8.0, "Islamabad"
            st.rerun()
        
        if st.button("🟡 Moderate", use_container_width=True):
            pm25, pm10, temperature, humidity, wind_speed, city = 45.0, 75.0, 28.0, 65.0, 3.0, "Lahore"
            st.rerun()
        
        if st.button("🔴 High Pollution", use_container_width=True):
            pm25, pm10, temperature, humidity, wind_speed, city = 120.0, 180.0, 35.0, 80.0, 1.0, "Karachi"
            st.rerun()
    
    # Prediction
    if st.button("🔮 Predict Air Quality", type="primary", use_container_width=True):
        input_data = {"PM2.5": pm25, "PM10": pm10, "Temperature": temperature, "Humidity": humidity, "Wind_Speed": wind_speed}
        if city:
            input_data["City"] = city
        
        with st.spinner("Making prediction..."):
            result = rt_predictor.predict_from_input(input_data)
        
        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
            return
        
        # Results
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            category = result['predicted_category']
            color = get_aqi_color(category)
            st.markdown(f"**Predicted Category**")
            st.markdown(f"<h3 style='color: {color}; margin: 0;'>{category}</h3>", unsafe_allow_html=True)
        
        with res_col2:
            confidence = result['confidence']
            st.markdown(f"**Confidence**")
            st.markdown(f"<h3 style='color: #1f77b4; margin: 0;'>{confidence:.1%}</h3>", unsafe_allow_html=True)
        
        with res_col3:
            aqi_est = result.get('aqi_estimate', 'N/A')
            st.markdown(f"**Estimated AQI**")
            st.markdown(f"<h3 style='color: #ff9800; margin: 0;'>{aqi_est}</h3>", unsafe_allow_html=True)
        
        # Charts and recommendations
        chart_col, rec_col = st.columns([2, 1])
        
        with chart_col:
            st.markdown("#### 📈 Category Probabilities")
            prob_chart = create_probability_chart(result['all_probabilities'])
            st.plotly_chart(prob_chart, use_container_width=True)
        
        with rec_col:
            st.markdown("#### 💡 Recommendations")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                st.markdown(f"**{i}.** {rec}")

def render_batch_prediction_tab():
    """Render batch prediction interface"""
    st.markdown("### 📊 Batch Prediction")
    st.markdown("Upload CSV files for batch processing")
    
    rt_predictor = load_realtime_model()
    if rt_predictor is None:
        st.error("Model not available")
        return
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Simple column validation
            required_cols = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.info("Required columns: PM2.5, PM10, Temperature, Humidity, Wind_Speed")
                return
            
            if st.button("🔮 Process Predictions", type="primary"):
                input_list = batch_df.to_dict('records')
                
                with st.spinner(f"Processing {len(input_list)} predictions..."):
                    results = rt_predictor.batch_predict(input_list)
                
                results_data = []
                for i, result in enumerate(results):
                    if "error" not in result:
                        results_data.append({
                            'Scenario': i + 1,
                            'PM2.5': input_list[i]['PM2.5'],
                            'Predicted_Category': result['predicted_category'],
                            'Confidence': f"{result['confidence']:.1%}",
                            'AQI_Estimate': result.get('aqi_estimate', 'N/A')
                        })
                
                if results_data:
                    st.markdown("### 📊 Results")
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results", data=csv, file_name="predictions.csv", mime="text/csv")
        
        except Exception as e:
            st.error(f"Error: {e}")

def render_analytics_tab():
    """Render analytics interface"""
    st.markdown("### 📈 Analytics & Insights")
    
    df = load_data()
    rt_predictor = load_realtime_model()
    
    if df is None:
        st.error("Data not available")
        return
    
    # Model performance
    if rt_predictor:
        st.markdown("#### 🤖 Model Performance")
        model_info = rt_predictor.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model_info.get('model_name', 'Unknown'))
        with col2:
            st.metric("Features", model_info.get('feature_count', 'Unknown'))
        with col3:
            st.metric("Categories", len(model_info.get('categories', [])))
    
    # Data insights
    st.markdown("#### 📊 Data Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        aqi_dist = df['AQI_Category'].value_counts()
        fig = px.pie(values=aqi_dist.values, names=aqi_dist.index, title="AQI Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        fig = px.bar(x=city_aqi.index, y=city_aqi.values, title="Average AQI by City")
        st.plotly_chart(fig, use_container_width=True)

def render_footer():
    """Render footer"""
    st.markdown("---")
    st.markdown("### 🌟 About This Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📊 Historical Forecasting**\n\nTime-series predictions based on historical data")
    
    with col2:
        st.markdown("**🔮 Real-time Predictions**\n\nInstant predictions from current conditions")
    
    with col3:
        st.markdown("**📊 Batch Processing**\n\nProcess multiple scenarios via CSV upload")
    
    with col4:
        st.markdown("**📈 Advanced Analytics**\n\nModel insights and data analysis")
    
    st.markdown("---")
    st.markdown("**Model Performance:** 98.5% Accuracy | 23 Features | 6 AQI Categories")
    st.markdown("**Disclaimer:** This is a demonstration system. For official air quality information, please consult local environmental authorities.")

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🌫️ Pakistan Air Quality Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time air quality forecasting and prediction for major Pakistani cities</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical Forecast", "🔮 Real-time Prediction", "📊 Batch Prediction", "📈 Analytics"])
    
    with tab1:
        render_historical_forecast_tab()
    
    with tab2:
        render_realtime_prediction_tab()
    
    with tab3:
        render_batch_prediction_tab()
    
    with tab4:
        render_analytics_tab()
    
    render_footer()

if __name__ == "__main__":
    main()