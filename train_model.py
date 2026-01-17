"""
Student Performance Predictor - Model Training Module
Organized functions for training and saving the model.
"""

import pandas as pd
import numpy as np
import copy
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_and_prepare_data(csv_path):
    """Load data from CSV and prepare features and target."""
    df = pd.read_csv(csv_path)
    
    features = ["Hours Studied", "Previous Scores", "Extracurricular Activities", 
                "Sleep Hours", "Sample Question Papers Practiced"]
    x = df[features]
    y = df[["Performance Index"]]
    
    return x, y, df


def preprocess_data(x, y, test_size=0.2, random_state=42):
    """One-hot encode features, convert target to numeric, and split/scale data."""
    # One-hot encode
    X_encoded = pd.get_dummies(x, drop_first=True)
    
    # Convert y to numeric
    y_numeric = y.astype(float).values.reshape(-1)
    
    # Train-test split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_numeric, test_size=test_size, random_state=random_state
    )
    
    # Scale AFTER split (no data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, X_encoded.columns.tolist()


def compute_cost(X, y, w, b):
    """Compute the cost (MSE) for linear regression."""
    m = X.shape[0]
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y.reshape(-1)) ** 2)
    return cost


def compute_gradient(X, y, w, b):
    """Compute gradients for weights and bias."""
    m = X.shape[0]
    predictions = X.dot(w) + b
    error = predictions - y.reshape(-1)
    dj_dw = (1 / m) * X.T.dot(error)
    dj_db = (1 / m) * np.sum(error)
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha=0.001, num_iters=10000, verbose=True):
    """
    Perform gradient descent to train the model.
    
    Args:
        X: Training features
        y: Training target
        w_in: Initial weights
        b_in: Initial bias
        alpha: Learning rate
        num_iters: Number of iterations
        verbose: Print cost every 100 iterations
    
    Returns:
        w: Trained weights
        b: Trained bias
        J_history: Cost history
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            J_history.append(cost)
            if verbose:
                print(f"Iteration {i:4d}: Cost {cost:.6f}")

    return w, b, J_history


def train_model(csv_path, alpha=0.001, num_iters=10000):
    """
    Full pipeline: load data, preprocess, and train the model.
    
    Args:
        csv_path: Path to the CSV file
        alpha: Learning rate
        num_iters: Number of iterations
    
    Returns:
        Dictionary containing trained weights, bias, scaler, and feature names
    """
    print("Loading data...")
    x, y, df = load_and_prepare_data(csv_path)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(x, y)
    
    print("Training model...")
    m, n = X_train.shape
    initial_w = np.zeros(n)
    initial_b = 0.0
    
    w_final, b_final, J_history = gradient_descent(X_train, y_train, initial_w, 
                                                    initial_b, alpha, num_iters)
    
    # Evaluate on test set
    y_pred = X_test.dot(w_final) + b_final
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Training Complete!")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R² Score: {r2:.6f}")
    
    model_data = {
        'w': w_final,
        'b': b_final,
        'scaler': scaler,
        'feature_names': feature_names,
        'mse': mse,
        'r2_score': r2
    }
    
    return model_data


def save_model(model_data, filepath='student_performance_model.pkl'):
    """Save the trained model to a file using joblib."""
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='student_performance_model.pkl'):
    """Load a trained model from a file."""
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_data


def predict(model_data, features):
    """
    Make predictions using the trained model.
    
    Args:
        model_data: Dictionary containing trained model
        features: Input features (should match feature_names)
    
    Returns:
        Prediction value
    """
    w = model_data['w']
    b = model_data['b']
    scaler = model_data['scaler']
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = features_scaled.dot(w) + b
    
    return prediction[0]


if __name__ == "__main__":
    # Train and save the model
    model_data = train_model("Student_Performance.csv")
    save_model(model_data)
