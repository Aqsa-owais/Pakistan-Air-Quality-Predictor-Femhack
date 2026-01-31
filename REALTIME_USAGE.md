# Real-time Air Quality Prediction System

This system provides real-time air quality predictions using machine learning. You can input current environmental conditions and get immediate AQI category predictions with health recommendations.

## 🚀 Quick Start

### 1. Command Line Interface (Recommended)
```bash
python realtime_cli.py
```

Interactive CLI with commands:
- `predict` - Enter conditions manually
- `quick` - Use sample data
- `batch` - Process multiple scenarios
- `examples` - See input examples
- `help` - Show all commands

### 2. Web Interface
```bash
python realtime_web.py
```
Then open: http://localhost:5000

### 3. Python Code Integration
```python
from src.realtime_predictor import RealTimeAQIPredictor

# Load model
predictor = RealTimeAQIPredictor('models/aqi_predictor_xgboost.joblib')

# Make prediction
input_data = {
    "PM2.5": 65,
    "PM10": 95,
    "Temperature": 30,
    "Humidity": 70,
    "Wind_Speed": 2.5,
    "City": "Lahore"
}

result = predictor.predict_from_input(input_data)
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## 📊 Input Parameters

### Required Parameters
- **PM2.5**: Fine particulate matter concentration (μg/m³)
- **PM10**: Coarse particulate matter concentration (μg/m³)
- **Temperature**: Air temperature (°C)
- **Humidity**: Relative humidity (%)
- **Wind_Speed**: Wind speed (km/h)

### Optional Parameters
- **Pressure**: Atmospheric pressure (hPa)
- **City**: City name for context

## 🎯 Output Information

The system provides:
- **Predicted Category**: AQI category (Good, Moderate, Unhealthy, etc.)
- **Confidence**: Model confidence percentage
- **AQI Estimate**: Numerical AQI value
- **Health Recommendations**: Specific advice based on prediction
- **Category Probabilities**: Probability for each AQI category

## 📝 Example Scenarios

### Clean Air Day
```json
{
    "PM2.5": 15,
    "PM10": 25,
    "Temperature": 22,
    "Humidity": 45,
    "Wind_Speed": 8,
    "City": "Islamabad"
}
```

### Moderate Pollution
```json
{
    "PM2.5": 45,
    "PM10": 75,
    "Temperature": 28,
    "Humidity": 65,
    "Wind_Speed": 3,
    "City": "Lahore"
}
```

### High Pollution Alert
```json
{
    "PM2.5": 120,
    "PM10": 180,
    "Temperature": 35,
    "Humidity": 80,
    "Wind_Speed": 1,
    "City": "Karachi"
}
```

## 🔧 Features

### Real-time Prediction
- Instant predictions from current conditions
- No historical data required
- Synthetic feature generation for missing lag/rolling features

### Batch Processing
- Process multiple scenarios at once
- JSON file input support
- Comparative analysis

### Health Recommendations
- Category-specific health advice
- Sensitive group warnings
- Activity recommendations

### Web Interface
- User-friendly form input
- Visual probability charts
- Sample data buttons
- Responsive design

### CLI Interface
- Interactive command system
- Multiple input methods
- Formatted output display
- Help system

## ⚠️ Important Notes

1. **Model Limitations**: The current model may show bias toward certain categories due to training data distribution.

2. **Synthetic Features**: For real-time prediction without historical data, the system creates synthetic lag and rolling features based on current values.

3. **AQI Estimation**: The AQI estimate is calculated using a simplified formula based on PM2.5 values.

4. **Recommendations**: Health recommendations are general guidelines. Consult healthcare professionals for specific medical advice.

## 🛠️ Technical Details

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Features**: 23 engineered features including lag, rolling, seasonal, and interaction features
- **Categories**: 6 AQI categories from Good to Hazardous

### Feature Engineering
- Lag features (1, 2, 3, 7 days)
- Rolling statistics (3, 7, 14 day windows)
- Seasonal/cyclical encoding
- Weather-pollution interactions
- PM2.5/PM10 ratios

### Performance
- Training accuracy: ~98.5%
- F1-score: ~98.5%
- Real-time prediction: <1 second

## 🚀 Usage Examples

### CLI Demo
```bash
python demo_realtime.py
```

### Quick Test
```bash
python test_realtime.py
```

### Web Server
```bash
python realtime_web.py
# Open http://localhost:5000
```

## 📦 Dependencies

Required packages (install with `pip install -r requirements.txt`):
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- flask (for web interface)

## 🔮 Future Improvements

1. **Model Retraining**: Regular retraining with new data to reduce bias
2. **Historical Integration**: Use actual historical data when available
3. **Location-specific Models**: Train separate models for different cities
4. **Real-time Data Integration**: Connect to live air quality monitoring stations
5. **Mobile App**: Native mobile application for on-the-go predictions

## 📞 Support

For issues or questions:
1. Check the model is trained: `python run_pipeline.py`
2. Verify dependencies: `pip install -r requirements.txt`
3. Test basic functionality: `python test_realtime.py`

The system is designed to be user-friendly and provide immediate, actionable air quality insights for Pakistani cities.