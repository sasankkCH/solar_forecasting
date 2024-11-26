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
  - Model Architecture: Saved model architecture files (*.h5)
  - Training History: Training progress plots for each fold
  - Prediction Visualizations: Scatter plots comparing actual vs predicted values
  - Metrics: CSV and TXT files containing detailed performance metrics
  - Fold-wise Predictions: Individual prediction results for each cross-validation fold

## Key Features
- **Advanced Data Preprocessing**:
  - Feature engineering including radiation-temperature interactions
  - Cloud cover impact calculations
  - Wind power computations
  - Humidity-temperature relationships
  - Robust outlier detection and removal using IQR method

- **Optimized Model Architecture**:
  - Deep neural network with three dense layers (64-128-64 neurons)
  - Batch normalization for training stability
  - Dropout layers (0.3, 0.3, 0.2) for preventing overfitting
  - LeakyReLU activation functions
  - L1-L2 regularization for better generalization

- **Robust Training Process**:
  - 5-fold cross-validation for reliable evaluation
  - Early stopping with patience of 20 epochs
  - Learning rate reduction on plateau
  - Model checkpointing for best weights
  - Comprehensive metrics tracking

## Performance Metrics
Latest model performance (averaged across 5 folds):
- Training RMSE: 42.16 ± 3.61
- Test RMSE: 45.75 ± 3.05
- Training MAE: 30.44 ± 2.96
- Test MAE: 32.94 ± 2.70
- Training R²: 0.9725 ± 0.0047
- Test R²: 0.9677 ± 0.0033

These metrics demonstrate:
- Excellent predictive accuracy (R² > 0.96)
- Strong generalization (small gap between training and test metrics)
- Consistent performance (low standard deviations)

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python solar_forecasting.py
```

3. Load and use a trained model:
```python
import tensorflow as tf
model = tf.keras.models.load_model('path_to_model.h5')
```

## Dependencies
- TensorFlow >= 2.0
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Author
CH Sasank

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
