# Gold Price Forecasting Dataset Analysis

## 📊 Overview

This project performs comprehensive analysis and forecasting of gold prices using multiple machine learning approaches. The analysis includes exploratory data analysis (EDA), feature engineering, and predictions using three different models: Linear Regression, Random Forest, and LSTM (Long Short-Term Memory) neural networks.

## 🎯 Features

- **Data Exploration**: Statistical analysis and visualization of historical gold price data
- **Feature Engineering**: Creation of lag features, moving averages, and rolling statistics
- **Multiple Models**: 
  - Linear Regression (baseline model)
  - Random Forest Regressor (ensemble method)
  - LSTM Neural Network (deep learning approach)
- **Model Comparison**: Comprehensive evaluation using RMSE, MAE, and R² metrics
- **Visualizations**: 
  - Price trends over time
  - Price distribution
  - Moving averages
  - Daily returns
  - Model predictions comparison

## 📁 Project Structure

```
Gold-price-predictions/
├── gold_price_analysis.py  # Main analysis script
├── gold_prices.csv          # Historical gold price dataset
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sacdart/Gold-price-predictions.git
cd Gold-price-predictions
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the analysis script:
```bash
python gold_price_analysis.py
```

The script will:
1. Load and explore the gold price dataset
2. Perform feature engineering
3. Train three different forecasting models
4. Compare model performance
5. Generate visualizations

### Output Files

After running the script, you'll get:
- `gold_price_analysis.png`: EDA visualizations including price trends, distribution, moving averages, and daily returns
- `model_predictions.png`: Comparison of predictions from all three models

## 📈 Dataset

The dataset (`gold_prices.csv`) contains historical gold prices with the following structure:
- **Date**: Trading date
- **Price**: Gold price in USD per ounce

The dataset covers approximately 2 years of daily gold price data from 2020 to 2021.

## 🤖 Models

### 1. Linear Regression
A simple baseline model that establishes a linear relationship between features and gold prices.

### 2. Random Forest
An ensemble learning method that uses multiple decision trees to improve prediction accuracy and reduce overfitting.

### 3. LSTM Neural Network
A deep learning model specifically designed for time series forecasting, capable of learning long-term dependencies in the data.

## 📊 Evaluation Metrics

All models are evaluated using:
- **RMSE (Root Mean Squared Error)**: Measures the average prediction error
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **R² Score**: Proportion of variance in the dependent variable predictable from independent variables

## 🛠️ Technical Details

### Feature Engineering
- **Lag Features**: Previous 5 days' prices (Lag_1 to Lag_5)
- **Moving Averages**: 30-day and 90-day moving averages
- **Rolling Statistics**: 7-day and 14-day rolling mean and standard deviation

### Data Split
- Training: 80% of the data
- Testing: 20% of the data

### LSTM Architecture
- 2 LSTM layers (50 units each)
- Dropout layers (0.2) for regularization
- 2 Dense layers for output
- Adam optimizer with MSE loss

## 📝 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**sacdart**

## 🙏 Acknowledgments

- Historical gold price data used for analysis
- Scikit-learn and TensorFlow communities for excellent machine learning libraries