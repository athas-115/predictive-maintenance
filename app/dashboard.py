"""
Streamlit Dashboard for Predictive Maintenance System

This dashboard provides:
- Interactive prediction interface
- Real-time risk assessment
- Feature visualization
- Model performance metrics
- SHAP explanations
"""

import os
import requests
import warnings
import sys
import contextlib

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

# Suppress warnings and stderr
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
warnings.filterwarnings('ignore')

# Context manager to suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output."""
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

import streamlit as st
import pandas as pd
import joblib
import json
import subprocess
import socket
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for predictions
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'last_input' not in st.session_state:
    st.session_state.last_input = None

def get_theme_css():
    """Return the single, modern dark theme CSS."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Styles - DARK THEME */
* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: #e2e8f0 !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp,
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
  color: #e2e8f0 !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Constrain content width */
.block-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}

/* Headings */
h1 {
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
  font-size: 2.25rem !important;
  margin-bottom: 0.5rem !important;
}
h2 { color: #f1f5f9 !important; font-weight: 600; font-size: 1.5rem !important; margin-top: 2rem !important; }
h3 { color: #e2e8f0 !important; font-weight: 600; font-size: 1.1rem !important; }

/* Metric cards */
[data-testid="stMetric"] {
  background: rgba(30, 41, 59, 0.8);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(100, 116, 139, 0.4);
  border-radius: 16px;
  padding: 1.5rem !important;
  min-height: 120px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-width: 0 !important;
}
[data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4); border-color: rgba(96, 165, 250, 0.6); }
[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.875rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 2rem !important; font-weight: 700 !important; }
/* Prevent truncated metric text (macOS rendering width variance) */
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    display: inline-block !important;
    max-width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    word-break: keep-all !important;
    line-height: 1.2 !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word !important;
    line-height: 1.2 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  border-right: 1px solid rgba(100, 116, 139, 0.4);
  color: #e2e8f0 !important;
  width: clamp(280px, 22vw, 340px) !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] label { color: #e2e8f0 !important; font-weight: 600 !important; font-size: 0.9rem !important; margin-bottom: 0.5rem !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] { width: 100% !important; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
  color: white !important;
  border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
  width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6); }

/* Sliders */
.stSlider [data-baseweb="slider-track"] { background: rgba(100, 116, 139, 0.3) !important; }
.stSlider [data-baseweb="slider-track-fill"] { background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important; }
.stSlider [data-baseweb="slider-thumb"] { background: #60a5fa !important; box-shadow: 0 2px 8px rgba(96, 165, 250, 0.4) !important; }

/* Selects and popovers */
.stSelectbox div[data-baseweb="select"] > div { color: #e2e8f0 !important; background: rgba(30, 41, 59, 0.95) !important; }
[data-baseweb="popover"], [role="listbox"], [role="option"] { background: rgba(30, 41, 59, 0.95) !important; color: #e2e8f0 !important; }
[role="option"]:hover { background: rgba(59, 130, 246, 0.3) !important; color: white !important; }
.stSlider [data-testid="stThumbValue"] { color: #e2e8f0 !important; background: rgba(30, 41, 59, 0.95) !important; border: 1px solid rgba(100, 116, 139, 0.4) !important; }

/* Expander */
.streamlit-expanderHeader { background: rgba(30, 41, 59, 0.6); border-radius: 12px; color: #e2e8f0 !important; font-weight: 600; border: 1px solid rgba(100, 116, 139, 0.4); }
.streamlit-expanderContent { background: rgba(30, 41, 59, 0.5) !important; color: #e2e8f0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: rgba(30, 41, 59, 0.5); border-radius: 8px; color: #e2e8f0 !important; padding: 12px 24px; font-weight: 500; border: 1px solid rgba(100, 116, 139, 0.3); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); color: white !important; border: none; }
[data-baseweb="tab-panel"], [data-baseweb="tab-panel"] * { color: #e2e8f0 !important; }

/* Code blocks */
code { background: rgba(30, 41, 59, 0.8) !important; color: #e2e8f0 !important; padding: 4px 8px !important; border-radius: 6px !important; font-family: 'Courier New', monospace !important; }
pre { background: rgba(30, 41, 59, 0.8) !important; color: #e2e8f0 !important; padding: 1rem !important; border-radius: 8px !important; border: 1px solid rgba(100, 116, 139, 0.4) !important; }
pre code { background: transparent !important; color: #e2e8f0 !important; }
.stMarkdown pre, .stMarkdown code { background: rgba(30, 41, 59, 0.8) !important; color: #e2e8f0 !important; }

/* Divider */
hr { border-color: rgba(100, 116, 139, 0.3); }

/* Columns */
[data-testid="column"] { min-width: 0 !important; padding: 0.5rem !important; }

/* Responsive */
@media (max-width: 1200px) {
  h1 { font-size: 2rem !important; }
  h2 { font-size: 1.35rem !important; }
  .stTabs [data-baseweb="tab"] { padding: 10px 14px; font-size: 0.95rem; }
}
@media (max-width: 992px) {
  .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
  [data-testid="stMetric"] { min-height: 100px; padding: 1rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
}
@media (max-width: 992px) {
    /* Stack columns on small screens for cleaner layout */
    [data-testid="column"] { flex: 1 1 100% !important; }
}
@media (max-width: 768px) {
  .stButton > button { padding: 0.5rem 1rem; font-size: 0.9rem; }
}
</style>
"""

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)


# ============================================================================
# Utility Functions
# ============================================================================

def is_port_in_use(port):
    """Check if a port is in use."""
    # Check if running in Docker (API service will be at 'api' hostname)
    # Otherwise check localhost for local development
    hosts_to_check = ['api', 'localhost'] if port == 8000 else ['localhost']

    for host in hosts_to_check:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)  # Add timeout to avoid hanging
                if s.connect_ex((host, port)) == 0:
                    return True
        except (socket.gaierror, socket.timeout):
            continue
    return False



@st.cache_resource
def load_model_and_preprocessors():
    """
    Load trained model and preprocessing artifacts.

    Returns:
        Dictionary with model and preprocessors
    """
    try:
        # Use absolute paths relative to script location
        script_dir = Path(__file__).parent.parent
        model_dir = script_dir / 'models'
        data_dir = script_dir / 'data' / 'processed'

        # Load LightGBM model (best performing: 97.8% accuracy, 98.7% ROC-AUC)
        model_path = model_dir / 'lightgbm.pkl'

        if not model_path.exists():
            st.error(f"LightGBM model not found at {model_path}. Please train the model first.")
            return None

        # Suppress stderr during model loading to hide joblib warnings
        with suppress_stderr():
            model = joblib.load(model_path)
        model_name = 'lightgbm'

        # Load scaler
        scaler = joblib.load(data_dir / 'scaler.pkl')

        # Load label encoder
        label_encoder = joblib.load(data_dir / 'label_encoder.pkl')

        # Load metadata
        with open(data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        return {
            'model': model,
            'model_name': model_name,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': metadata.get('feature_names', [])
        }

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_input(input_dict, scaler, label_encoder):
    """
    Preprocess user input.

    Args:
        input_dict: Dictionary with input values
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder

    Returns:
        Preprocessed feature array
    """
    # Encode type
    type_encoded = label_encoder.transform([input_dict['type']])[0]

    # Create feature array with proper column names
    feature_names = ['Air temperature [K]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Torque [Nm]',
                     'Tool wear [min]', 'Type_Encoded']

    features_df = pd.DataFrame([[
        input_dict['air_temperature'],
        input_dict['process_temperature'],
        input_dict['rotational_speed'],
        input_dict['torque'],
        input_dict['tool_wear'],
        type_encoded
    ]], columns=feature_names)

    # Scale features
    features_scaled = scaler.transform(features_df)

    return features_scaled


def calculate_risk_level(probability):
    """
    Calculate risk level based on probability.
    """
    if probability < 0.3:
        return "Low", "#00cc44"
    elif probability < 0.7:
        return "Medium", "#ffaa00"
    else:
        return "High", "#ff4444"


def create_gauge_chart(probability):
    """
    Create a responsive gauge chart for failure probability.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={
            'text': "Failure Probability",
            'font': {
                'size': 18,
                'color': '#e2e8f0',
                'family': 'Inter, sans-serif'
            }
        },
        delta={'reference': 50},
        number={'font': {'size': 36, 'color': '#e2e8f0'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickfont': {'size': 12, 'color': '#e2e8f0'}
            },
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 30], 'color': "#10b981"},
                {'range': [30, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#ef4444"}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        font={
            'color': '#e2e8f0',
            'size': 12
        },
        title={
            'font': {
                'color': '#e2e8f0'
            }
        }
    )
    return fig


def create_feature_importance_plot(model, feature_names):
    """
    Create feature importance plot.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # Create DataFrame
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        # Create horizontal bar plot
        fig = px.bar(df_importance, x='Importance', y='Feature',
                     title='Feature Importance',
                     color='Importance',
                     color_continuous_scale='Viridis',
                     orientation='h')

        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(30, 41, 59, 0.5)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            font={
                'color': '#e2e8f0',
                'size': 12
            },
            title={
                'font': {
                    'color': '#e2e8f0',
                    'size': 16
                }
            },
            xaxis={
                'gridcolor': 'rgba(100, 116, 139, 0.3)',
                'color': '#e2e8f0',
                'title': {
                    'font': {'color': '#e2e8f0'}
                },
                'tickfont': {'color': '#e2e8f0'}
            },
            yaxis={
                'gridcolor': 'rgba(100, 116, 139, 0.3)',
                'color': '#e2e8f0',
                'title': {
                    'font': {'color': '#e2e8f0'}
                },
                'tickfont': {'color': '#e2e8f0'}
            },
            coloraxis={
                'colorbar': {
                    'tickfont': {'color': '#e2e8f0'},
                    'title': {
                        'font': {'color': '#e2e8f0'}
                    }
                }
            }
        )
        return fig
    else:
        return None


def load_metrics():
    """
    Load model evaluation metrics.
    """
    try:
        # Use absolute path relative to script location
        script_dir = Path(__file__).parent.parent
        eval_dir = script_dir / 'evaluation'

        # Find available model results
        metrics_dict = {}

        for model_dir in eval_dir.iterdir():
            if model_dir.is_dir():
                metrics_file = model_dir / 'metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        metrics_dict[model_dir.name] = metrics

        return metrics_dict

    except Exception as e:
        st.warning(f"Could not load metrics: {str(e)}")
        return {}


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """
    Main dashboard function with responsive layout.
    """
    # ========== HEADER SECTION ==========
    header_container = st.container()
    with header_container:
        st.title("🔧 Predictive Maintenance Dashboard")
        st.markdown("**AI-Powered Machine Failure Prediction System**")

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    # ========== LOAD MODEL ==========
    model_data = load_model_and_preprocessors()
    if model_data is None:
        st.stop()

    # ========== SIDEBAR - INPUT PARAMETERS ==========
    with st.sidebar:
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        st.header("⚙️ Machine Parameters")
        st.markdown("Adjust the parameters below to predict machine failure risk.")
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        # Input fields with consistent styling
        input_dict = {}

        input_dict['type'] = st.selectbox(
            "Product Type",
            options=['L', 'M', 'H'],
            index=1,
            help="L: Low quality, M: Medium quality, H: High quality"
        )
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

        input_dict['air_temperature'] = st.slider(
            "Air Temperature (K)",
            min_value=290.0,
            max_value=310.0,
            value=298.1,
            step=0.1,
            help="Ambient air temperature in Kelvin"
        )
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

        input_dict['process_temperature'] = st.slider(
            "Process Temperature (K)",
            min_value=300.0,
            max_value=320.0,
            value=308.6,
            step=0.1,
            help="Process temperature in Kelvin"
        )
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    input_dict['rotational_speed'] = st.sidebar.slider(
        "Rotational Speed (RPM)",
        min_value=1000,
        max_value=3000,
        value=1551,
        step=10,
        help="Rotational speed in revolutions per minute"
    )
    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    input_dict['torque'] = st.sidebar.slider(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=42.8,
        step=0.1,
        help="Torque in Newton-meters"
    )
    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    input_dict['tool_wear'] = st.sidebar.slider(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=100,
        step=5,
        help="Tool wear time in minutes"
    )
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # Predict button with full width
    predict_button = st.sidebar.button(
        "🔮 Predict Failure Risk",
        type="primary",
        use_container_width=True,
        key="predict_button"
    )

    # ========== MAIN CONTENT AREA ==========
    # Status Metrics Row
    st.markdown("### 📌 System Status")
    metrics_container = st.container()
    with metrics_container:
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.metric(
                label="Model",
                value=model_data['model_name'].replace('_', ' ').title(),
                help="Currently loaded ML model"
            )
        with col2:
            st.metric(
                label="Features",
                value=len(model_data['feature_names']),
                help="Number of input features"
            )
        with col3:
            st.metric(
                label="Status",
                value="✅ Ready",
                help="System operational status"
            )

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # ========== PREDICTION SECTION ==========
    if predict_button:
        with st.spinner("🔄 Making prediction..."):
            # Preprocess
            features = preprocess_input(
                input_dict,
                model_data['scaler'],
                model_data['label_encoder']
            )

            # Predict
            prediction = model_data['model'].predict(features)[0]

            try:
                probability = model_data['model'].predict_proba(features)[0, 1]
            except:
                probability = float(prediction)

            risk_level, risk_color = calculate_risk_level(probability)

            # Store in session state
            st.session_state.prediction_result = {
                'prediction': prediction,
                'probability': probability,
                'risk_level': risk_level,
                'risk_color': risk_color
            }
            st.session_state.last_input = input_dict.copy()

    # ========== PREDICTION RESULTS DISPLAY ==========
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        prediction = result['prediction']
        probability = result['probability']
        risk_level = result['risk_level']
        risk_color = result['risk_color']

        # Results Section Header
        st.markdown("## 📊 Prediction Results")
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        # Metrics Row - Consistent height and spacing
        results_container = st.container()
        with results_container:
            col1, col2, col3, col4 = st.columns(4, gap="large")

            with col1:
                st.metric(
                    label="Prediction",
                    value="⚠️ Failure" if prediction == 1 else "✅ No\u00A0Failure",
                    help="Predicted outcome"
                )

            with col2:
                st.metric(
                    label="Failure Probability",
                    value=f"{probability*100:.1f}%",
                    help="Likelihood of failure"
                )

            with col3:
                st.metric(
                    label="Risk Level",
                    value=risk_level,
                    help="Risk assessment category"
                )

            with col4:
                confidence = probability if prediction == 1 else (1 - probability)
                st.metric(
                    label="Confidence",
                    value=f"{confidence*100:.1f}%",
                    help="Model confidence in prediction"
                )

        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

        # Risk Assessment Section with 50-50 layout
        st.markdown("### 🎯 Risk Assessment")
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

        risk_container = st.container()
        with risk_container:
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                # Gauge Chart - Centered
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                fig_gauge = create_gauge_chart(probability)
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                # Risk Interpretation Card
                st.markdown("#### 📋 Risk Interpretation")
                st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

                if risk_level == "Low":
                    st.success("""
                    **🟢 Low Risk Detected**

                    Machine is operating within normal parameters. Continue routine maintenance schedule.
                    """)
                elif risk_level == "Medium":
                    st.warning("""
                    **🟡 Medium Risk Detected**

                    Elevated risk levels observed. Monitor closely and consider scheduling preventive maintenance.
                    """)
                else:
                    st.error("""
                    **🔴 High Risk Detected**

                    Critical failure risk! Immediate attention required. Schedule urgent maintenance inspection.
                    """)

                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

                # Input Parameters Card
                st.markdown("#### ⚙️ Input Parameters")
                st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                import json as json_module
                display_input = st.session_state.last_input if st.session_state.last_input else input_dict
                st.code(json_module.dumps(display_input, indent=2), language='json')

        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

    # ========== MODEL INSIGHTS SECTION ==========
    st.markdown("## 📈 Model Insights")
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    insights_container = st.container()
    with insights_container:
        col1, col2 = st.columns([1, 1], gap="large")

        # Feature Importance Card
        with col1:
            st.markdown("### 📊 Feature Importance")
            st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

            fig_importance = create_feature_importance_plot(
                model_data['model'],
                [
                    'Air Temperature',
                    'Process Temperature',
                    'Rotational Speed',
                    'Torque',
                    'Tool Wear',
                    'Type'
                ]
            )

            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True, key="feature_importance_chart")
            else:
                st.info("📌 Feature importance not available for this model type.")

        # Model Performance Card
        with col2:
            st.markdown("### 🎯 Model Performance")
            st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

            # Load metrics if available
            metrics_dict = load_metrics()

            if metrics_dict and model_data['model_name'] in metrics_dict:
                metrics = metrics_dict[model_data['model_name']]['metrics']

                # Create metrics DataFrame
                df_metrics = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Score': [
                        metrics.get('accuracy', 0),
                        metrics.get('precision', 0),
                        metrics.get('recall', 0),
                        metrics.get('f1_score', 0),
                        metrics.get('roc_auc', 0)
                    ]
                })

                # Create bar chart
                fig_metrics = px.bar(
                    df_metrics,
                    x='Metric',
                    y='Score',
                    title='Performance Metrics',
                    color='Score',
                    color_continuous_scale='Blues',
                    range_y=[0, 1]
                )

                fig_metrics.update_layout(
                    height=400,
                    showlegend=False,
                    paper_bgcolor='rgba(30, 41, 59, 0.5)',
                    plot_bgcolor='rgba(30, 41, 59, 0.5)',
                    font={
                        'color': '#e2e8f0',
                        'size': 12
                    },
                    title={
                        'font': {
                            'color': '#e2e8f0',
                            'size': 16
                        }
                    },
                    xaxis={
                        'gridcolor': 'rgba(100, 116, 139, 0.3)',
                        'color': '#e2e8f0',
                        'title': {
                            'font': {'color': '#e2e8f0'}
                        },
                        'tickfont': {'color': '#e2e8f0'}
                    },
                    yaxis={
                        'gridcolor': 'rgba(100, 116, 139, 0.3)',
                        'color': '#e2e8f0',
                    'title': {
                        'font': {'color': '#e2e8f0'}
                    },
                    'tickfont': {'color': '#e2e8f0'}
                },
                coloraxis={
                    'colorbar': {
                        'tickfont': {'color': '#e2e8f0'},
                        'title': {
                            'font': {'color': '#e2e8f0'}
                        }
                    }
                }
            )
                st.plotly_chart(fig_metrics, use_container_width=True, key="performance_chart")
            else:
                st.info("📌 Run evaluation script to see performance metrics.")

    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

    # ========== FOOTER SECTION ==========
    footer_container = st.container()
    with footer_container:
        st.markdown("### 📚 About This Dashboard")
        st.info("""
        **Predictive Maintenance System** provides real-time machine failure predictions using advanced machine learning.

        **✨ Key Features:**
        - 🎯 Interactive parameter adjustment
        - 📊 Real-time risk assessment
        - 🔍 Model interpretability with feature importance
        - 📈 Comprehensive performance metrics

        **🤖 Model:** Trained on the AI4I 2020 Predictive Maintenance Dataset with 97.8% accuracy.
        """)

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # ========== SIDEBAR FOOTER ==========
    with st.sidebar:
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 📚 Documentation")
        st.markdown("- [API Documentation](http://localhost:8000/docs)")
        st.markdown("- [GitHub Repository](https://github.com/athas-115/predictive-maintenance)")
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

        
        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

        # API Health Check
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("**API Health**")
        with col2:
            if is_port_in_use(8000):
                st.markdown("🟢")
            else:
                st.markdown("🔴")

                st.link_button(
            "🔍 Check API",
            "https://predictive-maintenance-api-2s8i.onrender.com/health",
            use_container_width=True
        )

        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

        

        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Last Updated:** 2026-06-15")


# ============================================================================
# Run Dashboard
# ============================================================================

if __name__ == "__main__":
    main()
