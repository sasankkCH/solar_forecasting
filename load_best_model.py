import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def preprocess_data(data):
    """Enhanced data preprocessing with feature engineering and outlier handling"""
    # Feature engineering
    data['radiation_temp'] = data['shortwave_radiation_backwards_sfc'] * data['temperature_2_m_above_gnd']
    data['cloud_impact'] = data['total_cloud_cover_sfc'] * data['shortwave_radiation_backwards_sfc']
    data['wind_power'] = data['wind_speed_10_m_above_gnd'] * data['wind_gust_10_m_above_gnd']
    data['humidity_temp'] = data['relative_humidity_2_m_above_gnd'] * data['temperature_2_m_above_gnd']
    
    # List of numerical columns for outlier removal
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    
    # Remove outliers using IQR method
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return data

def plot_predictions(y_true, y_pred, fold, set_name='Validation', save_path=RESULTS_DIR):
    """Plot prediction scatter plots"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Power Generation (kW)')
    plt.ylabel('Predicted Power Generation (kW)')
    plt.title(f'{set_name} Predictions vs Actual Values (Fold {fold})')
    
    # Add R² score to plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{set_name.lower()}_predictions_fold_{fold}.png'))
    plt.close()

def save_metrics(metrics_dict, save_path=RESULTS_DIR):
    """Save metrics to CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    df_metrics = pd.DataFrame([metrics_dict])
    df_metrics.to_csv(os.path.join(save_path, 'evaluation_metrics.csv'), index=False)
    
    # Also save as readable text file
    with open(os.path.join(save_path, 'evaluation_metrics.txt'), 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f'{key}: {value}\n')

def load_and_predict():
    # Load the data
    print("Loading data...")
    data_path = os.path.join(DATA_DIR, 'solarpowergeneration.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    data = pd.read_csv(data_path)
    
    # Preprocess data using the same function as training
    print("Preprocessing data...")
    data = preprocess_data(data)
    
    # Prepare features and target
    X = data.iloc[:, :-1].values  # All columns except the last one
    y = data.iloc[:, -1].values.reshape(-1, 1)  # Last column is the target
    
    # Initialize metrics storage
    all_metrics = {
        'rmse': [], 'mae': [], 'r2': []
    }
    
    # K-fold cross validation
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models_loaded = 0
    
    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\nProcessing fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features and target
        scaler_X = RobustScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        
        scaler_y = RobustScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # Load the model for this fold
        model_path = os.path.join(MODELS_DIR, f'best_model_fold_{fold}.h5')
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            continue
            
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model for fold {fold}")
            models_loaded += 1
        except Exception as e:
            print(f"Error loading model for fold {fold}: {str(e)}")
            continue
        
        # Make predictions
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        all_metrics['mae'].append(mae)
        all_metrics['rmse'].append(rmse)
        all_metrics['r2'].append(r2)
        
        # Plot predictions
        plot_predictions(y_val, y_pred, fold)
        
        # Save fold predictions
        fold_df = pd.DataFrame({
            'Actual_Power': y_val.flatten(),
            'Predicted_Power': y_pred.flatten(),
            'Absolute_Error': np.abs(y_val.flatten() - y_pred.flatten())
        })
        fold_df.to_csv(os.path.join(RESULTS_DIR, f'evaluation_predictions_fold_{fold}.csv'), index=False)
        
        # Print fold metrics
        print(f"\nFold {fold} Performance:")
        print(f"Mean Absolute Error: {mae:.2f} kW")
        print(f"Root Mean Square Error: {rmse:.2f} kW")
        print(f"R² Score: {r2:.4f}")
    
    # Calculate and save final metrics
    if models_loaded > 0:
        final_metrics = {
            'mean_mae': np.mean(all_metrics['mae']),
            'std_mae': np.std(all_metrics['mae']),
            'mean_rmse': np.mean(all_metrics['rmse']),
            'std_rmse': np.std(all_metrics['rmse']),
            'mean_r2': np.mean(all_metrics['r2']),
            'std_r2': np.std(all_metrics['r2']),
            'models_evaluated': models_loaded
        }
        
        save_metrics(final_metrics)
        
        print(f"\nSuccessfully evaluated {models_loaded} models.")
        print("\nFinal Mean Metrics:")
        print(f"Mean MAE: {final_metrics['mean_mae']:.4f} ± {final_metrics['std_mae']:.4f}")
        print(f"Mean RMSE: {final_metrics['mean_rmse']:.4f} ± {final_metrics['std_rmse']:.4f}")
        print(f"Mean R²: {final_metrics['mean_r2']:.4f} ± {final_metrics['std_r2']:.4f}")
    else:
        print("\nNo models were successfully loaded and evaluated.")
        print("Please ensure that:")
        print("1. The model files exist in the 'models' directory")
        print("2. The model files are named 'best_model_fold_X.h5' where X is the fold number")
        print("3. You have run the training script (improved_solar_forecasting.py) first")

if __name__ == "__main__":
    load_and_predict()
