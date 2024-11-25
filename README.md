# Solar Power Forecasting

## Project Overview
This project focuses on solar power forecasting using advanced machine learning models. It involves data preprocessing, model training, and evaluation to predict solar power generation more accurately.

## Directory Structure
- **data/**: Contains the dataset used for training and evaluation
  - `solarpowergeneration.csv`: Main dataset containing solar power generation data and weather parameters

- **models/**: Stores trained machine learning models
  - Contains trained model files (*.h5) for each cross-validation fold
  - Models are saved in TensorFlow's HDF5 format
  - Each fold's best model is preserved for ensemble predictions

- **results/**: Holds results from model evaluations, including:
  - Model Architecture: `enhanced_model_architecture.png`
  - Training History: Training progress plots for each fold
  - Prediction Visualizations: Scatter plots comparing actual vs predicted values
  - Metrics: CSV and TXT files containing detailed performance metrics
  - Fold-wise Predictions: Individual prediction results for each cross-validation fold

## Key Scripts
- **solar_forecasting.py**: 
  - Preprocesses the data with feature engineering and outlier removal.
  - Trains a neural network model with improved architecture and regularization.
  - Evaluates model performance using metrics like RMSE and R².
  - Includes functions for plotting training history and predictions.

- **load_best_model.py**: 
  - Loads a pre-trained model and preprocesses new data.
  - Makes predictions and visualizes results with scatter plots.

## Model Features
- **Data Preprocessing**:
  - Feature engineering including radiation-temperature interactions
  - Cloud cover impact calculations
  - Wind power computations
  - Humidity-temperature relationships
  - Robust outlier detection and removal using IQR method

- **Model Architecture**:
  - Deep neural network with optimized layers
  - Batch normalization for training stability
  - Dropout layers for preventing overfitting
  - LeakyReLU activation functions
  - L1-L2 regularization for better generalization

- **Training Process**:
  - K-fold cross-validation for robust evaluation
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau
  - Automatic model checkpointing
  - Comprehensive metrics tracking

## Performance Metrics
The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) Score
- Detailed validation metrics for each fold

## Usage
To run the scripts, ensure that all dependencies are installed and execute the Python files in the command line:
```bash
python solar_forecasting.py
python load_best_model.py
```

## Dependencies
The project requires the following Python packages:
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Authors and Acknowledgments
This project was developed by CH Sasank. Special thanks to contributors and any external resources or datasets used.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation
If you use this project in your research or work, please cite it as:
```
@software{solar_forecasting,
  author = {CH Sasank},
  title = {Solar Power Forecasting},
  year = {2024},
  url = {https://github.com/sasankkCH/solar_forecasting}
}
