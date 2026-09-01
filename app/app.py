"""
FastAPI REST API for Predictive Maintenance System

This API provides:
- /predict endpoint for failure prediction
- /health endpoint for health checks
- /models endpoint for available models
- Pydantic validation for inputs
- Comprehensive error handling
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import List
import json

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class MachineInput(BaseModel):
    """
    Input schema for machine parameters.
    """
    type: str = Field(..., description="Product type: L (Low), M (Medium), or H (High)")
    air_temperature: float = Field(..., description="Air temperature in Kelvin", ge=250, le=350)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", ge=250, le=350)
    rotational_speed: int = Field(..., description="Rotational speed in RPM", ge=0, le=5000)
    torque: float = Field(..., description="Torque in Nm", ge=0, le=100)
    tool_wear: int = Field(..., description="Tool wear in minutes", ge=0, le=300)

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v.upper() not in ['L', 'M', 'H']:
            raise ValueError('Type must be L, M, or H')
        return v.upper()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 100
            }
        }
    )


class PredictionOutput(BaseModel):
    """
    Output schema for prediction results.
    """
    prediction: int = Field(..., description="Predicted failure: 0 (No Failure) or 1 (Failure)")
    failure_probability: float = Field(..., description="Probability of failure (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_used: str = Field(..., description="Name of the model used for prediction")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "prediction": 1,
                "failure_probability": 0.85,
                "risk_level": "High",
                "confidence": 0.85,
                "model_used": "xgboost_optimized"
            }
        }
    )


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str
    model_loaded: bool
    version: str

    model_config = ConfigDict(
        protected_namespaces=()
    )


class BatchPredictionInput(BaseModel):
    """
    Input schema for batch predictions.
    """
    instances: List[MachineInput]


# ============================================================================
# Global Model State
# ============================================================================

class ModelState:
    """
    Global state for loaded models and preprocessors.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.model_name = None
        self.is_loaded = False


model_state = ModelState()


# ============================================================================
# Lifespan Context Manager (Replaces startup/shutdown events)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Predictive Maintenance API...")

    try:
        # Paths - use absolute paths
        script_dir = Path(__file__).parent.parent
        model_dir = script_dir / 'models'
        data_dir = script_dir / 'data' / 'processed'

        # Try to load the best model (xgboost_optimized or xgboost)
        model_candidates = ['xgboost_optimized.pkl', 'xgboost.pkl', 'random_forest.pkl']

        for model_file in model_candidates:
            model_path = model_dir / model_file
            if model_path.exists():
                model_state.model = joblib.load(model_path)
                model_state.model_name = model_file.replace('.pkl', '')
                logger.info(f"Loaded model: {model_state.model_name}")
                break

        if model_state.model is None:
            logger.warning("No model found. Please train a model first.")
        else:
            # Load scaler
            scaler_path = data_dir / 'scaler.pkl'
            if scaler_path.exists():
                model_state.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler")

            # Load label encoder
            encoder_path = data_dir / 'label_encoder.pkl'
            if encoder_path.exists():
                model_state.label_encoder = joblib.load(encoder_path)
                logger.info("Loaded label encoder")

            # Load metadata
            metadata_path = data_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_state.feature_names = metadata.get('feature_names', [])
                    logger.info(f"Loaded feature names: {model_state.feature_names}")

            model_state.is_loaded = True
            logger.info("API startup complete - Model ready for predictions")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        model_state.is_loaded = False

    yield

    # Shutdown
    logger.info("Shutting down Predictive Maintenance API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Predictive Maintenance API",
    description="REST API for machine failure prediction using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def preprocess_input(input_data: MachineInput) -> np.ndarray:
    """
    Preprocess input data to match training format.

    Args:
        input_data: Machine input parameters

    Returns:
        Preprocessed feature array
    """
    # Encode type
    type_encoded = model_state.label_encoder.transform([input_data.type])[0]

    # Create feature array
    features = np.array([
        input_data.air_temperature,
        input_data.process_temperature,
        input_data.rotational_speed,
        input_data.torque,
        input_data.tool_wear,
        type_encoded
    ]).reshape(1, -1)

    # Scale features
    features_scaled = model_state.scaler.transform(features)

    return features_scaled


def calculate_risk_level(probability: float) -> str:
    """
    Calculate risk level based on failure probability.

    Args:
        probability: Failure probability (0-1)

    Returns:
        Risk level string
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API status.

    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy" if model_state.is_loaded else "unhealthy",
        model_loaded=model_state.is_loaded,
        version="1.0.0"
    )


@app.get("/models", tags=["Models"])
async def get_models():
    """
    Get information about available models.

    Returns:
        Model information
    """
    if not model_state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "current_model": model_state.model_name,
        "feature_names": model_state.feature_names,
        "model_type": str(type(model_state.model).__name__)
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: MachineInput):
    """
    Make a prediction for a single machine instance.

    Args:
        input_data: Machine parameters

    Returns:
        Prediction results with failure probability and risk level
    """
    # Check if model is loaded
    if not model_state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Preprocess input
        features = preprocess_input(input_data)

        # Make prediction
        prediction = int(model_state.model.predict(features)[0])

        # Get probability
        try:
            probability = float(model_state.model.predict_proba(features)[0, 1])
        except:
            # If model doesn't support predict_proba, use binary prediction
            probability = float(prediction)

        # Calculate risk level
        risk_level = calculate_risk_level(probability)

        # Calculate confidence (using probability as proxy)
        confidence = probability if prediction == 1 else (1 - probability)

        logger.info(f"Prediction made: {prediction}, Probability: {probability:.4f}, Risk: {risk_level}")

        return PredictionOutput(
            prediction=prediction,
            failure_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            model_used=model_state.model_name
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Make predictions for multiple machine instances.

    Args:
        input_data: List of machine parameters

    Returns:
        List of prediction results
    """
    if not model_state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        predictions = []

        for instance in input_data.instances:
            features = preprocess_input(instance)
            prediction = int(model_state.model.predict(features)[0])

            try:
                probability = float(model_state.model.predict_proba(features)[0, 1])
            except:
                probability = float(prediction)

            risk_level = calculate_risk_level(probability)
            confidence = probability if prediction == 1 else (1 - probability)

            predictions.append({
                "input": instance.dict(),
                "prediction": prediction,
                "failure_probability": probability,
                "risk_level": risk_level,
                "confidence": confidence
            })

        logger.info(f"Batch prediction completed for {len(predictions)} instances")

        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """
    Get API statistics and model information.

    Returns:
        Statistics dictionary
    """
    if not model_state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_name": model_state.model_name,
        "feature_count": len(model_state.feature_names) if model_state.feature_names else 0,
        "features": model_state.feature_names,
        "api_version": "1.0.0"
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """
    Handle ValueError exceptions.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid input", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handle general exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Run the FastAPI application.
    """
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
