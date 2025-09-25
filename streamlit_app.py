import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌌 Cosmic Collision Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
        color: #ffffff;
    }
    
    .title-container {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        color: #a0a0a0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1e2139, #2a2d4a);
        border: 1px solid #3a3d5c;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .hazardous {
        background: linear-gradient(135deg, #ff4757, #ff3838);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .safe {
        background: linear-gradient(135deg, #2ed573, #1dd1a1);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #1a1d29;
    }
    
    .feature-importance {
        background: rgba(74, 144, 226, 0.1);
        border-left: 4px solid #4a90e2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AsteroidPredictor:
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.scaler = None
        self.model_dir = "saved_models"
        
    def load_models(self):
        """Load all saved models and preprocessors"""
        try:
            # Load XGBoost model
            xgb_path = os.path.join(self.model_dir, "xgboost_asteroid_model.pkl")
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    self.models['xgboost'] = pickle.load(f)
                st.success("✅ XGBoost model loaded")
            
            # Load Random Forest model
            rf_path = os.path.join(self.model_dir, "random_forest_asteroid_model.pkl")
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
                st.success("✅ Random Forest model loaded")
            
            # Load feature names
            features_path = os.path.join(self.model_dir, "feature_names.pkl")
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                st.success(f"✅ Feature names loaded ({len(self.feature_names)} features)")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                st.success("✅ Feature scaler loaded")
                
            return len(self.models) > 0
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return False
    
    def predict(self, features_df):
        """Make predictions using loaded models"""
        predictions = {}
        
        try:
            # Print debug information
            st.write(f"**Debug Info:**")
            st.write(f"Input features: {len(features_df.columns)} → {list(features_df.columns)}")
            
            if self.feature_names:
                st.write(f"Expected features: {len(self.feature_names)} → {self.feature_names}")
                
                # Create DataFrame with all expected features, filling missing ones with defaults
                aligned_features = pd.DataFrame(columns=self.feature_names)
                
                # Fill with available features
                for col in self.feature_names:
                    if col in features_df.columns:
                        aligned_features[col] = features_df[col]
                    else:
                        # Fill missing features with reasonable defaults
                        if 'velocity' in col.lower():
                            aligned_features[col] = 25000.0
                        elif 'distance' in col.lower():
                            aligned_features[col] = 0.1
                        elif 'energy' in col.lower():
                            aligned_features[col] = -5e7
                        elif 'mass' in col.lower():
                            aligned_features[col] = 1e15
                        elif 'time' in col.lower() or 'date' in col.lower():
                            aligned_features[col] = 1.2e12
                        else:
                            aligned_features[col] = 1.0
                
                features_df = aligned_features
                st.success(f"✅ Aligned to {len(features_df.columns)} features")
            
            # Apply scaling if available
            if self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform(features_df)
                    features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
                    st.success("✅ Applied feature scaling")
                except Exception as scale_error:
                    st.warning(f"⚠️ Scaling failed: {scale_error}")
            
            # XGBoost predictions
            if 'xgboost' in self.models:
                try:
                    xgb_pred = self.models['xgboost'].predict(features_df)[0]
                    xgb_prob = self.models['xgboost'].predict_proba(features_df)[0]
                    predictions['xgboost'] = {
                        'prediction': bool(xgb_pred),
                        'probability': float(xgb_prob[1]),
                        'confidence': float(max(xgb_prob))
                    }
                    st.success("✅ XGBoost prediction successful")
                except Exception as xgb_error:
                    st.error(f"❌ XGBoost error: {xgb_error}")
            
            # Random Forest predictions
            if 'random_forest' in self.models:
                try:
                    rf_pred = self.models['random_forest'].predict(features_df)[0]
                    rf_prob = self.models['random_forest'].predict_proba(features_df)[0]
                    predictions['random_forest'] = {
                        'prediction': bool(rf_pred),
                        'probability': float(rf_prob[1]),
                        'confidence': float(max(rf_prob))
                    }
                    st.success("✅ Random Forest prediction successful")
                except Exception as rf_error:
                    st.error(f"❌ Random Forest error: {rf_error}")
                
        except Exception as e:
            st.error(f"❌ General prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        return predictions

def create_feature_inputs():
    """Create input widgets for all asteroid features that models expect"""
    st.sidebar.markdown("## 🛸 Asteroid Features")
    
    features = {}
    
    # Basic approach data
    st.sidebar.markdown("### 📅 Approach Data")
    features['Epoch Date Close Approach'] = st.sidebar.number_input(
        "Epoch Date Close Approach", 
        min_value=7.889e11, max_value=1.473e12, value=1.2e12, step=1e10,
        help="Date of closest approach (Julian date)"
    )
    
    # Velocity parameters
    st.sidebar.markdown("### 🚀 Velocity Parameters")
    features['Relative Velocity km per sec'] = st.sidebar.number_input(
        "Relative Velocity (km/s)", 
        min_value=1.0, max_value=50.0, value=15.0, step=0.1,
        help="Speed relative to Earth in km/s"
    )
    
    features['Relative Velocity km per hr'] = st.sidebar.number_input(
        "Relative Velocity (km/hr)", 
        min_value=1000.0, max_value=200000.0, value=50000.0, step=1000.0,
        help="Speed relative to Earth in km/hr"
    )
    
    features['Miles per hour'] = st.sidebar.number_input(
        "Velocity (mph)", 
        min_value=500.0, max_value=120000.0, value=30000.0, step=500.0,
        help="Speed in miles per hour"
    )
    
    # Distance measurements
    st.sidebar.markdown("### 📏 Distance Measurements")
    features['Miss Dist.(Astronomical)'] = st.sidebar.number_input(
        "Miss Distance (AU)", 
        min_value=0.0001, max_value=0.5, value=0.1, step=0.001,
        help="Closest approach distance in Astronomical Units"
    )
    
    features['Miss Dist.(lunar)'] = st.sidebar.number_input(
        "Miss Distance (Lunar)", 
        min_value=0.1, max_value=200.0, value=20.0, step=0.1,
        help="Distance in lunar distances"
    )
    
    features['Miss Dist.(kilometers)'] = st.sidebar.number_input(
        "Miss Distance (km)", 
        min_value=10000.0, max_value=80000000.0, value=10000000.0, step=100000.0,
        help="Distance in kilometers"
    )
    
    features['Miss Dist.(miles)'] = st.sidebar.number_input(
        "Miss Distance (miles)", 
        min_value=6000.0, max_value=50000000.0, value=6000000.0, step=100000.0,
        help="Distance in miles"
    )
    
    # Orbital characteristics
    st.sidebar.markdown("### 🌍 Orbital Parameters")
    features['Jupiter Tisserand Invariant'] = st.sidebar.number_input(
        "Jupiter Tisserand Invariant", 
        min_value=2.0, max_value=9.0, value=3.5, step=0.1,
        help="Orbital parameter related to Jupiter's influence"
    )
    
    features['Epoch Osculation'] = st.sidebar.number_input(
        "Epoch Osculation", 
        min_value=2.45e6, max_value=2.46e6, value=2.455e6, step=1000.0,
        help="Reference epoch for orbital elements"
    )
    
    features['Semi Major Axis'] = st.sidebar.number_input(
        "Semi Major Axis (AU)", 
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help="Average distance from the Sun"
    )
    
    features['Asc Node Longitude'] = st.sidebar.number_input(
        "Ascending Node Longitude (deg)", 
        min_value=0.0, max_value=360.0, value=180.0, step=1.0,
        help="Longitude of ascending node"
    )
    
    features['Perihelion Arg'] = st.sidebar.number_input(
        "Perihelion Argument (deg)", 
        min_value=0.0, max_value=360.0, value=180.0, step=1.0,
        help="Argument of perihelion"
    )
    
    features['Aphelion Dist'] = st.sidebar.number_input(
        "Aphelion Distance (AU)", 
        min_value=0.5, max_value=10.0, value=2.0, step=0.1,
        help="Farthest distance from the Sun"
    )
    
    features['Perihelion Time'] = st.sidebar.number_input(
        "Perihelion Time", 
        min_value=2.45e6, max_value=2.46e6, value=2.455e6, step=1000.0,
        help="Time of perihelion passage"
    )
    
    features['Mean Anomaly'] = st.sidebar.number_input(
        "Mean Anomaly (degrees)", 
        min_value=0.0, max_value=360.0, value=180.0, step=1.0,
        help="Position in orbit"
    )
    
    features['Mean Motion'] = st.sidebar.number_input(
        "Mean Motion (deg/day)", 
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Average angular speed"
    )
    
    # Approach timing
    st.sidebar.markdown("### 📅 Approach Timing")
    features['approach_year'] = st.sidebar.number_input(
        "Approach Year", 
        min_value=1995, max_value=2030, value=2025, step=1,
        help="Year of closest approach"
    )
    
    features['approach_month'] = st.sidebar.number_input(
        "Approach Month", 
        min_value=1, max_value=12, value=6, step=1,
        help="Month of closest approach"
    )
    
    features['approach_day'] = st.sidebar.number_input(
        "Approach Day", 
        min_value=1, max_value=31, value=15, step=1,
        help="Day of closest approach"
    )
    
    # Orbital characteristics (text inputs for categories)
    st.sidebar.markdown("### 🔄 Additional Parameters")
    orbital_period_cat = st.sidebar.selectbox(
        "Orbital Period Category",
        ["Short", "Medium", "Long"],
        index=1,
        help="Orbital period classification"
    )
    features['Orbital Period'] = 1.0 if orbital_period_cat == "Medium" else (0.5 if orbital_period_cat == "Short" else 2.0)
    
    orbit_uncertainty = st.sidebar.selectbox(
        "Orbit Uncertainty",
        ["Low", "Medium", "High"],
        index=1,
        help="Uncertainty in orbital determination"
    )
    features['Orbit Uncertainity'] = 1.0 if orbit_uncertainty == "Medium" else (0.5 if orbit_uncertainty == "Low" else 2.0)
    
    # Advanced calculated features
    st.sidebar.markdown("### 🧮 Calculated Features")
    day_of_year = features['approach_month'] * 30 + features['approach_day']
    features['Day of Year'] = day_of_year
    
    features['Miss Distance to Semi Major Axis Ratio'] = features['Miss Dist.(Astronomical)'] / features['Semi Major Axis']
    
    features['Time Until Approach (days)'] = st.sidebar.number_input(
        "Time Until Approach (days)", 
        min_value=0, max_value=3650, value=365, step=1,
        help="Days until closest approach"
    )
    
    # Calculate eccentricity
    perihelion_dist = 2 * features['Semi Major Axis'] - features['Aphelion Dist']
    features['Eccentricity'] = (features['Aphelion Dist'] - perihelion_dist) / (features['Aphelion Dist'] + perihelion_dist)
    
    features['Orbital Period (years)'] = features['Semi Major Axis'] ** 1.5  # Kepler's third law
    
    # Velocity and energy calculations
    features['Average Orbital Velocity (m/s)'] = st.sidebar.number_input(
        "Average Orbital Velocity (m/s)", 
        min_value=10000.0, max_value=50000.0, value=25000.0, step=1000.0,
        help="Average orbital velocity"
    )
    
    features['Heliocentric Distance (m)'] = features['Semi Major Axis'] * 1.496e11  # AU to meters
    
    features['Escape Velocity (m/s)'] = st.sidebar.number_input(
        "Escape Velocity (m/s)", 
        min_value=1000.0, max_value=20000.0, value=5000.0, step=100.0,
        help="Escape velocity from asteroid"
    )
    
    features['Specific Orbital Energy (J/kg)'] = st.sidebar.number_input(
        "Specific Orbital Energy (J/kg)", 
        min_value=-1e8, max_value=0.0, value=-5e7, step=1e6,
        help="Specific orbital energy"
    )
    
    features['Specific Angular Momentum'] = st.sidebar.number_input(
        "Specific Angular Momentum", 
        min_value=1e12, max_value=1e15, value=1e13, step=1e11,
        help="Specific angular momentum"
    )
    
    features['Velocity at Perihelion (m/s)'] = st.sidebar.number_input(
        "Velocity at Perihelion (m/s)", 
        min_value=20000.0, max_value=60000.0, value=35000.0, step=1000.0,
        help="Velocity at closest point to Sun"
    )
    
    features['Velocity at Aphelion (m/s)'] = st.sidebar.number_input(
        "Velocity at Aphelion (m/s)", 
        min_value=5000.0, max_value=30000.0, value=15000.0, step=1000.0,
        help="Velocity at farthest point from Sun"
    )
    
    features['Synodic Period'] = st.sidebar.number_input(
        "Synodic Period", 
        min_value=100.0, max_value=2000.0, value=500.0, step=10.0,
        help="Synodic period with Earth"
    )
    
    # Additional derived features
    import datetime
    base_date = datetime.datetime(2025, 1, 1)
    approach_date = base_date + datetime.timedelta(days=features['Time Until Approach (days)'])
    features['next_approach_date'] = approach_date.timestamp()
    
    features['time_until_approach_days'] = features['Time Until Approach (days)']
    
    features['assumed_mass_kg'] = st.sidebar.number_input(
        "Assumed Mass (kg)", 
        min_value=1e12, max_value=1e18, value=1e15, step=1e12,
        help="Estimated asteroid mass"
    )
    
    features['velocity_m_s'] = features['Relative Velocity km per sec'] * 1000  # Convert to m/s
    
    features['energy_loss_gj'] = st.sidebar.number_input(
        "Energy Loss (GJ)", 
        min_value=0.0, max_value=1e12, value=1e9, step=1e8,
        help="Potential energy loss on impact"
    )
    
    return features

def create_visualization(predictions, features):
    """Create visualizations for predictions"""
    
    # Prediction probability chart
    fig_prob = go.Figure()
    
    models = []
    hazard_probs = []
    safe_probs = []
    
    for model_name, pred in predictions.items():
        models.append(model_name.replace('_', ' ').title())
        hazard_probs.append(pred['probability'] * 100)
        safe_probs.append((1 - pred['probability']) * 100)
    
    fig_prob.add_trace(go.Bar(
        name='Hazardous Probability',
        x=models,
        y=hazard_probs,
        marker_color='#ff4757',
        text=[f"{p:.1f}%" for p in hazard_probs],
        textposition='auto',
    ))
    
    fig_prob.add_trace(go.Bar(
        name='Safe Probability',
        x=models,
        y=safe_probs,
        marker_color='#2ed573',
        text=[f"{p:.1f}%" for p in safe_probs],
        textposition='auto',
    ))
    
    fig_prob.update_layout(
        title="🎯 Prediction Probabilities by Model",
        xaxis_title="Models",
        yaxis_title="Probability (%)",
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    # Risk assessment radar chart
    risk_factors = {
        'Velocity Risk': min(features.get('Relative Velocity km per hr', 0) / 200000 * 100, 100),
        'Distance Risk': max(0, 100 - features.get('Miss Dist.(Astronomical)', 0.5) * 200),
        'Size Risk': min(features.get('Semi Major Axis', 1.0) * 30, 100),
        'Orbital Risk': abs(features.get('Jupiter Tisserand Invariant', 3.5) - 3.5) * 20,
    }
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(risk_factors.values()),
        theta=list(risk_factors.keys()),
        fill='toself',
        marker_color='#ff6b6b',
        line_color='#ff4757',
        name='Risk Factors'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.2)'
            )
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="⚡ Risk Factor Analysis",
        height=400
    )
    
    return fig_prob, fig_radar

def main():
    # Title and header
    st.markdown('<h1 class="title-container">🌌 Cosmic Collision Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-Powered Asteroid Risk Assessment System</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = AsteroidPredictor()
    
    # Sidebar for model loading
    st.sidebar.markdown("## 🤖 Model Status")
    
    if st.sidebar.button("🔄 Load Models", type="primary"):
        with st.sidebar:
            with st.spinner("Loading models..."):
                model_loaded = predictor.load_models()
    else:
        model_loaded = predictor.load_models()
    
    if not model_loaded:
        st.error("❌ **No models found!** Please ensure pickle files are in the 'saved_models' directory.")
        st.info("💡 Run the model training notebook first to generate the pickle files.")
        return
    
    # Sample asteroids for quick testing
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎯 Quick Test")
    
    sample_asteroids = {
        "Custom": {},
        "High Risk Example": {
            "Miss Dist.(Astronomical)": 0.001,
            "Relative Velocity km per hr": 80000.0,
            "Semi Major Axis": 1.2,
            "Jupiter Tisserand Invariant": 3.0
        },
        "Low Risk Example": {
            "Miss Dist.(Astronomical)": 0.3,
            "Relative Velocity km per hr": 25000.0,
            "Semi Major Axis": 2.5,
            "Jupiter Tisserand Invariant": 4.5
        }
    }
    
    selected_sample = st.sidebar.selectbox(
        "Load Sample Asteroid",
        list(sample_asteroids.keys()),
        help="Select a pre-configured asteroid for quick testing"
    )
    
    # Create input features
    features = create_feature_inputs()
    
    # Override with sample values if selected
    if selected_sample != "Custom":
        sample_data = sample_asteroids[selected_sample]
        for key, value in sample_data.items():
            if key in features:
                features[key] = value
    
    # Show feature count
    st.sidebar.markdown(f"**Total Features:** {len(features)}")
    
    # Prediction button
    if st.sidebar.button("🚀 Analyze Asteroid Risk", type="primary"):
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Make predictions
        with st.spinner("🔮 Analyzing asteroid characteristics..."):
            predictions = predictor.predict(features_df)
        
        if predictions:
            # Main results area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("## 🎯 Prediction Results")
                
                # Calculate ensemble prediction
                if len(predictions) > 1:
                    avg_prob = np.mean([pred['probability'] for pred in predictions.values()])
                    ensemble_hazardous = avg_prob > 0.5
                else:
                    avg_prob = list(predictions.values())[0]['probability']
                    ensemble_hazardous = list(predictions.values())[0]['prediction']
                
                # Display main prediction
                if ensemble_hazardous:
                    st.markdown(f'<div class="hazardous">⚠️ POTENTIALLY HAZARDOUS ASTEROID<br>Risk Level: {avg_prob*100:.1f}%</div>', unsafe_allow_html=True)
                    st.error(f"🚨 **HIGH RISK DETECTED** - This asteroid shows characteristics consistent with potentially hazardous objects.")
                else:
                    st.markdown(f'<div class="safe">✅ SAFE ASTEROID<br>Risk Level: {avg_prob*100:.1f}%</div>', unsafe_allow_html=True)
                    st.success(f"🛡️ **LOW RISK** - This asteroid is classified as safe based on current parameters.")
                
                # Individual model results
                st.markdown("### 🤖 Model Breakdown")
                
                for model_name, pred in predictions.items():
                    with st.expander(f"📊 {model_name.replace('_', ' ').title()} Results"):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                "Prediction", 
                                "Hazardous" if pred['prediction'] else "Safe",
                                delta=f"{pred['probability']*100:.1f}% risk"
                            )
                        
                        with col_b:
                            st.metric(
                                "Confidence", 
                                f"{pred['confidence']*100:.1f}%"
                            )
                        
                        with col_c:
                            st.metric(
                                "Risk Score", 
                                f"{pred['probability']*100:.1f}%"
                            )
            
            with col2:
                st.markdown("## 📊 Risk Metrics")
                
                # Key metrics
                velocity_risk = features['Relative Velocity km per hr'] / 200000
                distance_risk = max(0, 1 - features['Miss Dist.(Astronomical)'] * 2)
                size_risk = features['Semi Major Axis'] / 5
                
                st.metric("🚀 Velocity Risk", f"{velocity_risk*100:.1f}%")
                st.metric("📏 Distance Risk", f"{distance_risk*100:.1f}%")
                st.metric("📐 Size Risk", f"{size_risk*100:.1f}%")
                
                # Quick stats
                st.markdown("### 📈 Quick Stats")
                st.write(f"**Approach Distance:** {features['Miss Dist.(lunar)']:.1f} lunar distances")
                st.write(f"**Relative Speed:** {features['Relative Velocity km per hr']:,.0f} km/hr")
                st.write(f"**Orbital Period:** ~{(features['Semi Major Axis']**1.5):.1f} years")
            
            # Visualizations
            st.markdown("## 📈 Analysis Visualizations")
            
            col_vis1, col_vis2 = st.columns(2)
            
            fig_prob, fig_radar = create_visualization(predictions, features)
            
            with col_vis1:
                st.plotly_chart(fig_prob, use_container_width=True)
            
            with col_vis2:
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Feature importance (if available)
            st.markdown("## 🔍 Feature Analysis")
            
            important_features = {
                'Miss Distance': features['Miss Dist.(Astronomical)'],
                'Relative Velocity': features['Relative Velocity km per hr'],
                'Semi Major Axis': features['Semi Major Axis'],
                'Jupiter Tisserand': features['Jupiter Tisserand Invariant']
            }
            
            for feat_name, feat_value in important_features.items():
                st.markdown(f"**{feat_name}:** {feat_value}")
    
    # Information sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.info("""
    This AI system uses machine learning models trained on NASA's asteroid database to assess collision risks.
    
    **Models Used:**
    - XGBoost Classifier
    - Random Forest Classifier
    
    **Risk Factors:**
    - Orbital characteristics
    - Approach distance
    - Relative velocity
    - Size indicators
    """)
    
    st.sidebar.markdown("## 🛡️ Disclaimer")
    st.sidebar.warning("""
    This tool is for educational and research purposes. 
    For official asteroid risk assessments, consult NASA's 
    Planetary Defense Coordination Office.
    """)

if __name__ == "__main__":
    main()
