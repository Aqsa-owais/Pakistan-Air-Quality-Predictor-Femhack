# CSV Upload Guide for Air Quality Predictions

## 🚀 Quick Start

The enhanced CSV upload feature now supports flexible column names and provides automatic mapping!

## 📊 Required Data Fields

Your CSV file needs these 5 essential fields (column names can vary):

### 1. **PM2.5 Concentration** (μg/m³)
**Accepted column names:**
- `PM2.5`, `pm2.5`, `PM25`, `pm25`, `PM_2_5`, `pm_2_5`

### 2. **PM10 Concentration** (μg/m³)
**Accepted column names:**
- `PM10`, `pm10`, `PM_10`, `pm_10`

### 3. **Temperature** (°C)
**Accepted column names:**
- `Temperature`, `temperature`, `temp`, `Temp`, `TEMP`

### 4. **Humidity** (%)
**Accepted column names:**
- `Humidity`, `humidity`, `RH`, `rh`, `relative_humidity`

### 5. **Wind Speed** (km/h)
**Accepted column names:**
- `Wind_Speed`, `wind_speed`, `WindSpeed`, `windspeed`, `Wind Speed`, `wind speed`

### 6. **City** (Optional)
**Accepted column names:**
- `City`, `city`, `location`, `Location`

## 📝 Sample CSV Formats

### Format 1: Standard Names
```csv
PM2.5,PM10,Temperature,Humidity,Wind_Speed,City
15,25,22,45,8,Islamabad
45,75,28,65,3,Lahore
120,180,35,80,1,Karachi
```

### Format 2: Alternative Names
```csv
pm25,pm10,temp,RH,wind_speed,location
15,25,22,45,8,Islamabad
45,75,28,65,3,Lahore
120,180,35,80,1,Karachi
```

### Format 3: Mixed Names
```csv
PM25,PM_10,temperature,humidity,WindSpeed,City
15,25,22,45,8,Islamabad
45,75,28,65,3,Lahore
120,180,35,80,1,Karachi
```

## 🔧 How the Smart Upload Works

### 1. **Automatic Detection**
- Upload your CSV file
- The system automatically detects column names
- Shows you the mapping it found

### 2. **Manual Mapping** (if needed)
- If some columns aren't detected, you can manually map them
- Select which column in your CSV corresponds to each required field
- The system remembers your mapping for processing

### 3. **Sample Download**
- If you're missing required columns, download the sample CSV
- Use it as a template for your own data

## 📊 Expected Results

After processing, you'll get:

### Results Table
| Scenario | PM2.5 | PM10 | Temperature | Predicted_Category | Confidence | AQI_Estimate |
|----------|-------|------|-------------|-------------------|------------|--------------|
| 1 | 15 | 25 | 22 | Moderate | 85.0% | 56 |
| 2 | 45 | 75 | 28 | Unhealthy for Sensitive Groups | 85.0% | 124 |
| 3 | 120 | 180 | 35 | Unhealthy | 80.8% | 184 |

### Summary Statistics
- Category distribution pie chart
- Percentage breakdown by AQI category
- Downloadable results CSV

## 🎯 Example Scenarios

### Clean Air Conditions
```csv
PM2.5,PM10,Temperature,Humidity,Wind_Speed,City
8,15,20,40,12,Islamabad
12,22,18,35,15,Murree
```
**Expected:** Good to Moderate categories

### Urban Pollution
```csv
PM2.5,PM10,Temperature,Humidity,Wind_Speed,City
45,75,28,65,3,Lahore
55,85,30,70,2,Faisalabad
```
**Expected:** Unhealthy for Sensitive Groups

### High Pollution Events
```csv
PM2.5,PM10,Temperature,Humidity,Wind_Speed,City
120,180,35,80,1,Karachi
200,280,38,85,0.5,Lahore
```
**Expected:** Unhealthy to Very Unhealthy

## ⚠️ Common Issues & Solutions

### Issue 1: "Missing required columns"
**Solution:** 
- Check your column names against the accepted names list
- Use the manual mapping feature
- Download and use the sample CSV template

### Issue 2: "Invalid values"
**Solution:**
- Ensure all values are numeric (no text in data columns)
- Remove any empty rows
- Check for special characters or formatting issues

### Issue 3: "Processing failed"
**Solution:**
- Verify data ranges are realistic:
  - PM2.5: 0-500 μg/m³
  - PM10: 0-1000 μg/m³
  - Temperature: -10 to 50°C
  - Humidity: 0-100%
  - Wind Speed: 0-100 km/h

## 📥 Download Sample Files

The system provides sample CSV files you can download:

1. **Basic Sample** - Standard format with 10 scenarios
2. **Extended Sample** - More scenarios with different cities
3. **Template** - Empty template for your own data

## 🔮 Processing Features

### Batch Size
- Process up to 1000 scenarios at once
- Larger files are processed in chunks
- Progress indicator shows processing status

### Output Options
- View results in interactive table
- Download results as CSV
- Summary statistics and visualizations
- Category distribution charts

## 💡 Tips for Best Results

1. **Data Quality:**
   - Use realistic values within expected ranges
   - Include city information when available
   - Ensure consistent units (μg/m³ for PM, °C for temperature)

2. **File Format:**
   - Save as CSV (comma-separated values)
   - Use UTF-8 encoding
   - Avoid special characters in column names

3. **Testing:**
   - Start with the sample CSV to test the system
   - Upload small batches first to verify format
   - Check results make sense for your input values

## 🚀 Advanced Features

### City-Specific Predictions
- Include city names for location-specific adjustments
- Supported cities: Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar, Quetta

### Confidence Levels
- Realistic confidence scores (70-95%)
- Higher confidence for clear-cut cases
- Lower confidence for borderline conditions

### Health Recommendations
- Category-specific health advice
- Sensitive group warnings
- Activity recommendations

The enhanced CSV upload system makes it easy to process multiple air quality scenarios with flexible data formats and comprehensive results!