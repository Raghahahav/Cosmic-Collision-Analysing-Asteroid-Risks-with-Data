# 🌌 Cosmic Collision Analyzer - Streamlit Web App

A professional, AI-powered web application for asteroid risk assessment using machine learning models.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## 🚀 Features

### 🎯 **Core Functionality**
- **Real-time Asteroid Risk Prediction** using trained ML models
- **Interactive Feature Input** for asteroid characteristics
- **Multi-Model Ensemble** predictions (XGBoost + Random Forest)
- **Professional Dark Theme** with cosmic aesthetics
- **Interactive Visualizations** using Plotly

### 📊 **Visualizations**
- **Prediction Probability Charts** comparing model outputs
- **Risk Factor Radar Chart** showing multi-dimensional risk analysis
- **Real-time Metrics** and confidence scores
- **Feature Importance Analysis**

### 🎨 **UI/UX Features**
- **Dark Cosmic Theme** with gradient effects
- **Responsive Design** for all screen sizes
- **Interactive Sidebar** for feature inputs
- **Professional Cards** and metric displays
- **Animated Loading** and status indicators

## 📋 Prerequisites

1. **Trained Models**: Ensure you have run the Jupyter notebook to generate pickle files
2. **Python Environment**: Python 3.8+ with required dependencies
3. **Model Files**: The following files should exist in `saved_models/`:
   - `xgboost_asteroid_model.pkl`
   - `random_forest_asteroid_model.pkl`
   - `feature_names.pkl`
   - `feature_scaler.pkl`
   - `model_metadata.pkl`

## 🛠️ Installation & Setup

### Method 1: Quick Start (Recommended)
```bash
# Windows
run_app.bat

# Linux/Mac
chmod +x run_app.sh
./run_app.sh
```

### Method 2: Manual Setup
```bash
# 1. Activate virtual environment
source ../.venv/bin/activate  # Linux/Mac
# OR
..\.venv\Scripts\activate.bat  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

## 🎮 How to Use

### 1. **Start the Application**
- Run the app using one of the methods above
- The app will automatically open in your default browser
- Default URL: `http://localhost:8501`

### 2. **Load Models**
- Click "🔄 Load Models" in the sidebar
- Verify all models are loaded successfully
- Green checkmarks indicate successful loading

### 3. **Input Asteroid Features**
Configure asteroid parameters in the sidebar:

#### 🌍 **Orbital Parameters**
- **Semi Major Axis**: Average distance from the Sun (AU)
- **Aphelion Distance**: Farthest point from the Sun (AU)
- **Jupiter Tisserand Invariant**: Orbital stability parameter

#### 🚀 **Approach Parameters**
- **Relative Velocity**: Speed relative to Earth (km/hr)
- **Velocity (mph)**: Speed in miles per hour

#### 📏 **Distance Measurements**
- **Miss Distance (AU)**: Closest approach in Astronomical Units
- **Miss Distance (Lunar)**: Distance in lunar distances
- **Miss Distance (km/miles)**: Distance in kilometers/miles

#### 🔄 **Additional Parameters**
- **Mean Motion**: Average angular speed (deg/day)
- **Mean Anomaly**: Position in orbit (degrees)

### 4. **Analyze Risk**
- Click "🚀 Analyze Asteroid Risk"
- View real-time predictions and analysis
- Explore interactive visualizations

## 📊 Understanding Results

### **Risk Classification**
- **🚨 POTENTIALLY HAZARDOUS**: Risk > 50%
- **🛡️ SAFE**: Risk ≤ 50%

### **Model Outputs**
- **Ensemble Prediction**: Average of all models
- **Individual Model Results**: XGBoost and Random Forest
- **Confidence Scores**: Model certainty levels
- **Risk Percentages**: Numerical risk assessment

### **Visualizations**
- **Probability Chart**: Model comparison
- **Risk Radar**: Multi-dimensional risk factors
- **Metrics Dashboard**: Key risk indicators

## 🎨 Customization

### **Theme Modifications**
Edit the CSS in `streamlit_app.py` to customize:
- Colors and gradients
- Card designs
- Typography
- Layout spacing

### **Feature Additions**
The modular design allows easy extension:
- Add new input features
- Integrate additional models
- Create custom visualizations
- Implement advanced analytics

## 🔧 Troubleshooting

### **Common Issues**

#### ❌ "No models found"
- **Cause**: Pickle files missing
- **Solution**: Run the Jupyter notebook to train and save models

#### ❌ "Feature mismatch error"
- **Cause**: Inconsistent preprocessing
- **Solution**: Retrain models or adjust feature inputs

#### ❌ "Module not found"
- **Cause**: Missing dependencies
- **Solution**: `pip install -r requirements.txt`

#### ❌ "Port already in use"
- **Cause**: Another Streamlit app running
- **Solution**: `streamlit run streamlit_app.py --server.port 8502`

### **Performance Optimization**
- Use caching for model loading: `@st.cache_resource`
- Optimize feature preprocessing
- Implement batch predictions for multiple asteroids

## 📁 File Structure

```
Cosmic-Collision-Analysing-Asteroid-Risks-with-Data/
├── streamlit_app.py              # Main application
├── run_app.bat                   # Windows launcher
├── run_app.sh                    # Linux/Mac launcher
├── requirements.txt              # Dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── saved_models/                # ML model files
│   ├── xgboost_asteroid_model.pkl
│   ├── random_forest_asteroid_model.pkl
│   ├── feature_names.pkl
│   ├── feature_scaler.pkl
│   └── model_metadata.pkl
└── STREAMLIT_README.md          # This file
```

## 🌟 Advanced Features

### **Model Management**
- Automatic model version detection
- Model performance metrics
- Feature importance analysis

### **Data Export**
- Prediction history
- Analysis reports
- Feature configurations

### **API Integration**
- REST API endpoints (future enhancement)
- Batch processing capabilities
- Real-time data feeds

## 🛡️ Security & Disclaimers

- **Educational Purpose**: This tool is for research and educational use
- **Not Official**: Not a replacement for NASA's official assessments
- **Data Privacy**: All inputs are processed locally
- **Model Limitations**: Predictions based on training data patterns

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add enhancements
4. Submit pull request

## 📧 Support

For issues or questions:
1. Check troubleshooting section
2. Review model training notebook
3. Verify file structure and dependencies

---

**Made with ❤️ for space enthusiasts and data scientists**

*Explore the cosmos, one prediction at a time* 🌌✨
