# Pseudo Student Performance Predictor

A machine learning application that predicts student performance based on study habits and other factors. Built with gradient descent, saved with joblib, and exposed via a FastAPI web UI.

## Features

✅ **Custom Linear Regression** - Implemented from scratch using gradient descent  
✅ **Model Persistence** - Saved with joblib for easy loading and deployment  
✅ **Organized Functions** - Modular training code in `train_model.py`  
✅ **FastAPI Web Service** - REST API with automatic documentation  
✅ **Interactive UI** - Modern, responsive web interface

## Project Structure

```
Student Performance Predictor/
├── Student_Performance.csv      # Dataset
├── Working Model.ipynb          # Interactive Jupyter notebook (for exploration)
├── train_model.py               # Organized functions for training and prediction
├── app.py                       # FastAPI application
├── index.html                   # Web UI
├── requirements.txt             # Python dependencies
├── student_performance_model.pkl # Saved model (generated after training)
└── README.md                    # This file
```

## Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify you have the dataset:**
   - Ensure `Student_Performance.csv` is in the same directory

## Usage

### Step 1: Train the Model

Run the training script to train the model and save it:

```bash
python train_model.py
```

This will:

- Load the CSV data
- Preprocess and scale features
- Train a linear regression model using gradient descent
- Evaluate on the test set
- Save the model to `student_performance_model.pkl`

### Step 2: Start the FastAPI Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload
```

The server will start at `http://localhost:8000`

### Step 3: Use the Web UI

Open your browser and navigate to:

- **UI**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)

## API Endpoints

### GET `/api/health`

Check if the API is running and model is loaded.

### GET `/api/model-info`

Get information about the trained model (MSE, R² score, features).

### POST `/api/predict`

Make a prediction for a student's performance.

**Request Body:**

```json
{
  "hours_studied": 7.0,
  "previous_scores": 85.0,
  "extracurricular_activities": 1,
  "sleep_hours": 8.0,
  "sample_question_papers_practiced": 5
}
```

**Response:**

```json
{
  "predicted_performance_index": 72.5,
  "message": "Predicted performance index: 72.50"
}
```

## Module: train_model.py

Organized functions for the entire ML pipeline:

- **`load_and_prepare_data(csv_path)`** - Load CSV and extract features/target
- **`preprocess_data(x, y)`** - One-hot encode, split, and scale data
- **`compute_cost(X, y, w, b)`** - Compute MSE cost
- **`compute_gradient(X, y, w, b)`** - Compute gradients
- **`gradient_descent(X, y, w_in, b_in, alpha, num_iters)`** - Train using gradient descent
- **`train_model(csv_path, alpha, num_iters)`** - Full training pipeline
- **`save_model(model_data, filepath)`** - Save model with joblib
- **`load_model(filepath)`** - Load saved model
- **`predict(model_data, features)`** - Make predictions with loaded model

## Example: Using the Functions Programmatically

```python
from train_model import train_model, save_model, load_model, predict

# Train the model
model_data = train_model("Student_Performance.csv")

# Save it
save_model(model_data, "my_model.pkl")

# Load it later
model = load_model("my_model.pkl")

# Make predictions
features = [7.0, 85.0, 1, 8.0, 5]
prediction = predict(model, features)
print(f"Predicted performance: {prediction:.2f}")
```

## Jupyter Notebook

The `Working Model.ipynb` notebook includes:

- Data loading and exploration
- Feature engineering and preprocessing
- Cost function definition
- Gradient descent implementation
- Model training and evaluation
- Model saving with joblib

Run it with:

```bash
jupyter notebook "Working Model.ipynb"
```

## Technical Details

### Model Architecture

- **Algorithm**: Linear Regression with Gradient Descent
- **Features**: Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Sample Question Papers Practiced
- **Preprocessing**: One-hot encoding, StandardScaler normalization
- **Split**: 80/20 train-test split
- **Loss Function**: Mean Squared Error (MSE)

### Hyperparameters

- Learning rate (alpha): 0.001
- Iterations: 10,000
- Test size: 0.2

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- fastapi
- uvicorn
- pydantic

## License

This project is for educational purposes.
