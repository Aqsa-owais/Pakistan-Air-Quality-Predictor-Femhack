"""
Streamlit App for Air Quality Prediction
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

def render_realtime_prediction_tab():
    """Render real-time prediction interface"""
    st.header("🔮 Real-time Air Quality Prediction")
    st.markdown("Enter current environmental conditions to get immediate AQI predictions")
    
    # Load real-time model
    rt_predictor = load_realtime_model()
    if rt_predictor is None:
        st.error("Real-time prediction model not available")
        return
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Environmental Conditions")
        
        # Required inputs
        pm25 = st.number_input(
            "PM2.5 Concentration (μg/m³)", 
            min_value=0.0, 
            max_value=500.0, 
            value=35.0,
            step=0.1,
            help="Fine particulate matter concentration"
        )
        
        pm10 = st.number_input(
            "PM10 Concentration (μg/m³)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=65.0,
            step=0.1,
            help="Coarse particulate matter concentration"
        )
        
        temperature = st.number_input(
            "Temperature (°C)", 
            min_value=-10.0, 
            max_value=50.0, 
            value=28.0,
            step=0.1
        )
        
        humidity = st.number_input(
            "Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=65.0,
            step=0.1
        )
        
        wind_speed = st.number_input(
            "Wind Speed (km/h)", 
            min_value=0.0, 
            max_value=100.0, 
            value=5.0,
            step=0.1
        )
        
        # Optional inputs
        st.subheader("🌡️ Optional Parameters")
        
        pressure = st.number_input(
            "Atmospheric Pressure (hPa)", 
            min_value=900.0, 
            max_value=1100.0, 
            value=1013.25,
            step=0.1,
            help="Leave as default if unknown"
        )
        
        city = st.selectbox(
            "City (Optional)",
            ["", "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Quetta"],
            help="Select city for context"
        )
    
    with col2:
        st.subheader("🎯 Quick Scenarios")
        st.markdown("Click to load sample data:")
        
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            if st.button("🟢 Good Air", use_container_width=True):
                st.session_state.update({
                    'pm25': 15.0, 'pm10': 25.0, 'temperature': 22.0,
                    'humidity': 45.0, 'wind_speed': 8.0, 'city': 'Islamabad'
                })
                st.rerun()
        
        with col2b:
            if st.button("🟡 Moderate", use_container_width=True):
                st.session_state.update({
                    'pm25': 45.0, 'pm10': 75.0, 'temperature': 28.0,
                    'humidity': 65.0, 'wind_speed': 3.0, 'city': 'Lahore'
                })
                st.rerun()
        
        with col2c:
            if st.button("🔴 Unhealthy", use_container_width=True):
                st.session_state.update({
                    'pm25': 120.0, 'pm10': 180.0, 'temperature': 35.0,
                    'humidity': 80.0, 'wind_speed': 1.0, 'city': 'Karachi'
                })
                st.rerun()
        
        # Update inputs from session state
        if 'pm25' in st.session_state:
            pm25 = st.session_state.pm25
        if 'pm10' in st.session_state:
            pm10 = st.session_state.pm10
        if 'temperature' in st.session_state:
            temperature = st.session_state.temperature
        if 'humidity' in st.session_state:
            humidity = st.session_state.humidity
        if 'wind_speed' in st.session_state:
            wind_speed = st.session_state.wind_speed
        if 'city' in st.session_state:
            city = st.session_state.city
    
    # Prediction button
    if st.button("🔮 Predict Air Quality", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            "PM2.5": pm25,
            "PM10": pm10,
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind_Speed": wind_speed,
            "Pressure": pressure
        }
        
        if city:
            input_data["City"] = city
        
        # Make prediction
        with st.spinner("Making prediction..."):
            result = rt_predictor.predict_from_input(input_data)
        
        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
            return
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Main result
        col3, col4, col5 = st.columns(3)
        
        with col3:
            category = result['predicted_category']
            color_class = "good-quality" if "Good" in category else \
                         "moderate-quality" if "Moderate" in category else \
                         "unhealthy-quality"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Category</h3>
                <h2 class="{color_class}">{category}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            confidence = result['confidence']
            st.metric("Confidence", f"{confidence:.1%}")
            
            if result.get('aqi_estimate'):
                st.metric("Estimated AQI", result['aqi_estimate'])
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Input Summary</h4>
                <small>{result['input_summary']}</small>
                <br><br>
                <small>Predicted at: {result['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability chart
        col6, col7 = st.columns([2, 1])
        
        with col6:
            prob_chart = create_probability_chart(result['all_probabilities'])
            st.plotly_chart(prob_chart, use_container_width=True)
        
        with col7:
            st.subheader("💡 Health Recommendations")
            for i, rec in enumerate(result['recommendations'][:4], 1):
                st.markdown(f"{i}. {rec}")
        
        # Model info
        with st.expander("🤖 Model Information"):
            model_info = rt_predictor.get_model_info()
            col8, col9 = st.columns(2)
            
            with col8:
                st.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
                st.write(f"**Features:** {model_info.get('feature_count', 'Unknown')}")
                st.write(f"**Training Date:** {model_info.get('training_date', 'Unknown')}")
            
            with col9:
                st.write("**Categories:**")
                for cat in model_info.get('categories', []):
                    st.write(f"• {cat}")

def render_batch_prediction_tab():
    """Render batch prediction interface"""
    st.header("📊 Batch Prediction")
    st.markdown("Upload a CSV file or enter multiple scenarios for batch processing")
    
    rt_predictor = load_realtime_model()
    if rt_predictor is None:
        st.error("Real-time prediction model not available")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with environmental data",
        type=['csv'],
        help="CSV should contain columns: PM2.5, PM10, Temperature, Humidity, Wind_Speed"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(batch_df.head())
            
            # Validate required columns
            required_cols = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            if st.button("🔮 Process Batch Predictions", type="primary"):
                # Convert dataframe to list of dictionaries
                input_list = batch_df.to_dict('records')
                
                with st.spinner(f"Processing {len(input_list)} predictions..."):
                    results = rt_predictor.batch_predict(input_list)
                
                # Create results dataframe
                results_data = []
                for i, result in enumerate(results):
                    if "error" not in result:
                        results_data.append({
                            'Scenario': i + 1,
                            'PM2.5': input_list[i]['PM2.5'],
                            'PM10': input_list[i]['PM10'],
                            'Temperature': input_list[i]['Temperature'],
                            'Predicted_Category': result['predicted_category'],
                            'Confidence': f"{result['confidence']:.1%}",
                            'AQI_Estimate': result.get('aqi_estimate', 'N/A')
                        })
                
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    st.subheader("📊 Batch Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"aqi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        category_counts = results_df['Predicted_Category'].value_counts()
                        fig = px.pie(
                            values=category_counts.values,
                            names=category_counts.index,
                            title="Predicted Category Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Category Breakdown:**")
                        for category, count in category_counts.items():
                            percentage = (count / len(results_df)) * 100
                            st.write(f"• {category}: {count} ({percentage:.1f}%)")
                
                else:
                    st.error("No valid predictions generated")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        # Manual batch input
        st.subheader("✏️ Manual Batch Input")
        st.markdown("Enter multiple scenarios manually:")
        
        # Initialize session state for scenarios
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = []
        
        # Add scenario form
        with st.form("add_scenario"):
            st.write("**Add New Scenario:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pm25 = st.number_input("PM2.5", min_value=0.0, value=35.0, key="batch_pm25")
                pm10 = st.number_input("PM10", min_value=0.0, value=65.0, key="batch_pm10")
            
            with col2:
                temp = st.number_input("Temperature", value=28.0, key="batch_temp")
                humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=65.0, key="batch_humidity")
            
            with col3:
                wind = st.number_input("Wind Speed", min_value=0.0, value=5.0, key="batch_wind")
                city = st.selectbox("City", ["", "Karachi", "Lahore", "Islamabad"], key="batch_city")
            
            if st.form_submit_button("➕ Add Scenario"):
                scenario = {
                    "PM2.5": pm25, "PM10": pm10, "Temperature": temp,
                    "Humidity": humidity, "Wind_Speed": wind
                }
                if city:
                    scenario["City"] = city
                
                st.session_state.scenarios.append(scenario)
                st.success(f"Added scenario {len(st.session_state.scenarios)}")
                st.rerun()
        
        # Display current scenarios
        if st.session_state.scenarios:
            st.write(f"**Current Scenarios ({len(st.session_state.scenarios)}):**")
            scenarios_df = pd.DataFrame(st.session_state.scenarios)
            st.dataframe(scenarios_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔮 Predict All Scenarios", type="primary"):
                    with st.spinner("Processing scenarios..."):
                        results = rt_predictor.batch_predict(st.session_state.scenarios)
                    
                    # Display results similar to file upload
                    results_data = []
                    for i, result in enumerate(results):
                        if "error" not in result:
                            results_data.append({
                                'Scenario': i + 1,
                                'Predicted_Category': result['predicted_category'],
                                'Confidence': f"{result['confidence']:.1%}",
                                'AQI_Estimate': result.get('aqi_estimate', 'N/A')
                            })
                    
                    if results_data:
                        st.subheader("📊 Results")
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
            
            with col2:
                if st.button("🗑️ Clear All Scenarios"):
                    st.session_state.scenarios = []
                    st.rerun()

def render_historical_forecast_tab():
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
    st.markdown("**Real-time air quality forecasting and prediction for major Pakistani cities**")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Historical Forecast", 
        "🔮 Real-time Prediction", 
        "📊 Batch Prediction",
        "📈 Analytics"
    ])
    
    with tab1:
        render_historical_forecast_tab()
    
    with tab2:
        render_realtime_prediction_tab()
    
    with tab3:
        render_batch_prediction_tab()
    
    with tab4:
        render_analytics_tab()

def render_historical_forecast_tab():
    """Render historical forecast interface"""
    # Load data and model
    df = load_data()
    forecaster = load_model()
    
    if df is None or forecaster is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🎛️ Forecast Controls")
    
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

def render_analytics_tab():
    """Render analytics and insights"""
    st.header("📈 Analytics & Insights")
    
    df = load_data()
    rt_predictor = load_realtime_model()
    
    if df is None:
        st.error("Data not available for analytics")
        return
    
    # Model performance metrics
    if rt_predictor:
        st.subheader("🤖 Model Performance")
        model_info = rt_predictor.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model_info.get('model_name', 'Unknown'))
        with col2:
            st.metric("Features Used", model_info.get('feature_count', 'Unknown'))
        with col3:
            st.metric("Categories", len(model_info.get('categories', [])))
        
        # Performance details
        if 'performance' in model_info:
            perf = model_info['performance']
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Accuracy", f"{perf.get('accuracy', 0):.1%}")
            with col5:
                st.metric("F1-Score", f"{perf.get('f1_score', 0):.1%}")
    
    # Data insights
    st.subheader("📊 Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AQI distribution
        aqi_dist = df['AQI_Category'].value_counts()
        fig = px.pie(
            values=aqi_dist.values,
            names=aqi_dist.index,
            title="Historical AQI Category Distribution",
            color_discrete_map={
                'Good': '#4caf50',
                'Moderate': '#ff9800',
                'Unhealthy for Sensitive Groups': '#ff5722',
                'Unhealthy': '#f44336',
                'Very Unhealthy': '#9c27b0',
                'Hazardous': '#795548'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # City-wise average AQI
        city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=city_aqi.index,
            y=city_aqi.values,
            title="Average AQI by City",
            labels={'x': 'City', 'y': 'Average AQI'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal trends
    st.subheader("🌍 Seasonal Trends")
    df['Month'] = df['Date'].dt.month
    monthly_aqi = df.groupby('Month')['AQI'].mean()
    
    fig = px.line(
        x=monthly_aqi.index,
        y=monthly_aqi.values,
        title="Average AQI by Month",
        labels={'x': 'Month', 'y': 'Average AQI'}
    )
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Environmental Correlations")
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
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality metrics
    st.subheader("📋 Data Quality")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
    
    with col2:
        st.metric("Cities Covered", df['City'].nunique())
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    with col3:
        st.metric("Latest Data", df['Date'].max().strftime('%Y-%m-%d'))
        st.metric("Data Completeness", f"{100-missing_pct:.1f}%")

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** This comprehensive air quality prediction system uses machine learning to provide 
    both historical forecasting and real-time predictions for major Pakistani cities. The system combines 
    historical data analysis with immediate environmental condition assessment.
    
    **Features:**
    - 📊 Historical forecasting based on time-series data
    - 🔮 Real-time predictions from current conditions  
    - 📊 Batch processing for multiple scenarios
    - 📈 Advanced analytics and model insights
    
    **Disclaimer:** This is a demonstration system. For official air quality information, 
    please consult local environmental authorities.
    """)

if __name__ == "__main__":
    main()