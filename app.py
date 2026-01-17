"""
FastAPI Application for Student Performance Predictor
Provides API endpoints for model predictions with an interactive UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path
from train_model import load_model, predict

app = FastAPI(
    title="Student Performance Predictor",
    description="Predict student performance based on study habits and other factors",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model_data = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    global model_data, model_loaded
    try:
        model_file = Path("student_performance_model.pkl")
        if model_file.exists():
            model_data = load_model("student_performance_model.pkl")
            model_loaded = True
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model file not found at startup")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


class PredictionInput(BaseModel):
    """Input schema for prediction requests."""
    hours_studied: float
    previous_scores: float
    extracurricular_activities: int
    sleep_hours: float
    sample_question_papers_practiced: int
    
    class Config:
        schema_extra = {
            "example": {
                "hours_studied": 7.0,
                "previous_scores": 85.0,
                "extracurricular_activities": 1,
                "sleep_hours": 8.0,
                "sample_question_papers_practiced": 5
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""
    predicted_performance_index: float
    message: str


@app.get("/")
async def root():
    """Serve the main UI page."""
    return FileResponse("index.html")


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the trained model."""
    if not model_loaded:
        return {"error": "Model not loaded. Please train first."}
    
    return {
        "status": "Model loaded successfully",
        "test_mse": model_data.get('mse'),
        "test_r2_score": model_data.get('r2_score'),
        "features": model_data.get('feature_names'),
        "parameters": len(model_data.get('w', []))
    }


@app.post("/api/predict", response_model=PredictionOutput)
async def make_prediction(input_data: PredictionInput):
    """
    Make a prediction based on input features.
    
    Returns:
        Predicted performance index and a message
    """
    if not model_loaded:
        return {"predicted_performance_index": 0, "message": "Model not loaded"}
    
    try:
        # Prepare features in the correct order
        features = [
            input_data.hours_studied,
            input_data.previous_scores,
            input_data.extracurricular_activities,
            input_data.sleep_hours,
            input_data.sample_question_papers_practiced
        ]
        
        # Make prediction
        prediction = predict(model_data, features)
        
        # Cap prediction at 100
        prediction = min(prediction, 100)
        
        return {
            "predicted_performance_index": round(prediction, 2),
            "message": f"Predicted performance index: {prediction:.2f}"
        }
    except Exception as e:
        return {"predicted_performance_index": 0, "message": f"Error: {str(e)}"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
