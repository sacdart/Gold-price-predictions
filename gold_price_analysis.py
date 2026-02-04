"""
Gold Price Forecasting Dataset Analysis
This script performs comprehensive analysis and forecasting of gold prices using various machine learning models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GoldPriceAnalyzer:
    """
    A comprehensive class for analyzing and forecasting gold prices.
    """
    
    # Define feature columns as class constant
    FEATURE_COLS = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 
                    'MA_30', 'MA_90', 'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_14']
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with data path.
        
        Args:
            data_path (str): Path to the CSV file containing gold price data
        """
        self.data_path = data_path
        self.data = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date')
        self.data.set_index('Date', inplace=True)
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        print(f"\nMissing values: {self.data.isnull().sum().sum()}")
        
        # Date range
        print(f"\nDate range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Number of trading days: {len(self.data)}")
        
        return self.data.describe()
    
    def visualize_data(self):
        """Create visualizations for the dataset."""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gold Price Analysis', fontsize=16, fontweight='bold')
        
        # Time series plot
        axes[0, 0].plot(self.data.index, self.data['Price'], linewidth=2, color='gold')
        axes[0, 0].set_title('Gold Price Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution plot
        axes[0, 1].hist(self.data['Price'], bins=50, edgecolor='black', alpha=0.7, color='goldenrod')
        axes[0, 1].set_title('Price Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Price (USD)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Moving averages
        self.data['MA_30'] = self.data['Price'].rolling(window=30).mean()
        self.data['MA_90'] = self.data['Price'].rolling(window=90).mean()
        axes[1, 0].plot(self.data.index, self.data['Price'], label='Price', linewidth=1, alpha=0.7)
        axes[1, 0].plot(self.data.index, self.data['MA_30'], label='30-Day MA', linewidth=2)
        axes[1, 0].plot(self.data.index, self.data['MA_90'], label='90-Day MA', linewidth=2)
        axes[1, 0].set_title('Price with Moving Averages', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price (USD)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Daily returns
        self.data['Returns'] = self.data['Price'].pct_change()
        axes[1, 1].plot(self.data.index, self.data['Returns'], linewidth=1, alpha=0.6, color='darkgreen')
        axes[1, 1].set_title('Daily Returns', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Returns (%)')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gold_price_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'gold_price_analysis.png'")
        plt.close()
        
    def feature_engineering(self):
        """Create features for modeling."""
        print("\nPerforming feature engineering...")
        
        # Remove rows with NaN values from moving averages
        self.data = self.data.dropna()
        
        # Create lag features
        for i in range(1, 6):
            self.data[f'Lag_{i}'] = self.data['Price'].shift(i)
        
        # Rolling statistics
        self.data['Rolling_Mean_7'] = self.data['Price'].rolling(window=7).mean()
        self.data['Rolling_Std_7'] = self.data['Price'].rolling(window=7).std()
        self.data['Rolling_Mean_14'] = self.data['Price'].rolling(window=14).mean()
        
        # Drop NaN values created by feature engineering
        self.data = self.data.dropna()
        
        print(f"Features created. New shape: {self.data.shape}")
        print(f"Feature columns: {list(self.data.columns)}")
        
        return self.data
    
    def prepare_data_for_modeling(self):
        """Prepare data for machine learning models."""
        print("\nPreparing data for modeling...")
        
        # Use class constant for feature columns
        X = self.data[self.FEATURE_COLS].values
        y = self.data['Price'].values
        
        # Split the data (80-20 split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train and evaluate Linear Regression model."""
        print("\n" + "-"*60)
        print("LINEAR REGRESSION MODEL")
        print("-"*60)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluation
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"Training RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return model, y_pred_test
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train and evaluate Random Forest model."""
        print("\n" + "-"*60)
        print("RANDOM FOREST MODEL")
        print("-"*60)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluation
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"Training RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Feature importance - use class constant
        feature_importance = pd.DataFrame({
            'Feature': self.FEATURE_COLS,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.to_string(index=False))
        
        return model, y_pred_test
    
    def prepare_lstm_data(self, sequence_length=60):
        """Prepare data for LSTM model."""
        print("\nPreparing data for LSTM...")
        
        # Use only price for LSTM
        data = self.data['Price'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"LSTM Training set shape: {X_train.shape}")
        print(f"LSTM Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """Train and evaluate LSTM model."""
        print("\n" + "-"*60)
        print("LSTM MODEL")
        print("-"*60)
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("Training LSTM model...")
        # Train model (store history for potential future use)
        _ = model.fit(X_train, y_train, batch_size=32, epochs=20, 
                      validation_split=0.1, verbose=0)
        
        # Predictions
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)
        
        # Inverse transform to get actual prices
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_train_actual = self.scaler.inverse_transform(y_pred_train)
        y_pred_test_actual = self.scaler.inverse_transform(y_pred_test)
        
        # Evaluation
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))
        train_mae = mean_absolute_error(y_train_actual, y_pred_train_actual)
        test_mae = mean_absolute_error(y_test_actual, y_pred_test_actual)
        train_r2 = r2_score(y_train_actual, y_pred_train_actual)
        test_r2 = r2_score(y_test_actual, y_pred_test_actual)
        
        print(f"Training RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return model, y_pred_test_actual, y_test_actual
    
    def compare_models(self, lr_pred, rf_pred, lstm_pred, y_test, lstm_y_test):
        """Compare predictions from different models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Calculate metrics for each model
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        lstm_rmse = np.sqrt(mean_squared_error(lstm_y_test, lstm_pred))
        
        lr_r2 = r2_score(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        lstm_r2 = r2_score(lstm_y_test, lstm_pred)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'LSTM'],
            'RMSE': [lr_rmse, rf_rmse, lstm_rmse],
            'R² Score': [lr_r2, rf_r2, lstm_r2]
        })
        
        print("\n" + comparison.to_string(index=False))
        
        # Find best model
        best_model_idx = comparison['RMSE'].idxmin()
        best_model = comparison.iloc[best_model_idx]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        
        return comparison
    
    def visualize_predictions(self, y_test, lr_pred, rf_pred, lstm_pred, lstm_y_test):
        """Visualize predictions from all models."""
        print("\nGenerating prediction visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Predictions Comparison', fontsize=16, fontweight='bold')
        
        # Get test dates
        test_dates = self.data.index[-len(y_test):]
        lstm_test_dates = self.data.index[-len(lstm_y_test):]
        
        # Linear Regression
        axes[0, 0].plot(test_dates, y_test, label='Actual', linewidth=2, color='blue')
        axes[0, 0].plot(test_dates, lr_pred, label='Predicted', linewidth=2, 
                       color='red', alpha=0.7)
        axes[0, 0].set_title('Linear Regression Predictions', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Random Forest
        axes[0, 1].plot(test_dates, y_test, label='Actual', linewidth=2, color='blue')
        axes[0, 1].plot(test_dates, rf_pred, label='Predicted', linewidth=2, 
                       color='green', alpha=0.7)
        axes[0, 1].set_title('Random Forest Predictions', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price (USD)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # LSTM
        axes[1, 0].plot(lstm_test_dates, lstm_y_test, label='Actual', linewidth=2, color='blue')
        axes[1, 0].plot(lstm_test_dates, lstm_pred, label='Predicted', linewidth=2, 
                       color='purple', alpha=0.7)
        axes[1, 0].set_title('LSTM Predictions', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price (USD)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # All models comparison
        axes[1, 1].plot(test_dates, y_test, label='Actual', linewidth=2.5, color='blue')
        axes[1, 1].plot(test_dates, lr_pred, label='Linear Regression', 
                       linewidth=1.5, alpha=0.6, linestyle='--')
        axes[1, 1].plot(test_dates, rf_pred, label='Random Forest', 
                       linewidth=1.5, alpha=0.6, linestyle='--')
        # Note: LSTM has different test size, so we align it
        if len(lstm_y_test) <= len(y_test):
            axes[1, 1].plot(lstm_test_dates, lstm_pred, label='LSTM', 
                           linewidth=1.5, alpha=0.6, linestyle='--')
        axes[1, 1].set_title('All Models Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Price (USD)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
        print("Predictions visualization saved as 'model_predictions.png'")
        plt.close()


def main():
    """Main function to run the gold price analysis."""
    print("="*60)
    print("GOLD PRICE FORECASTING ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = GoldPriceAnalyzer('gold_prices.csv')
    
    # Load and explore data
    analyzer.load_data()
    analyzer.explore_data()
    analyzer.visualize_data()
    
    # Feature engineering
    analyzer.feature_engineering()
    
    # Prepare data for traditional ML models
    X_train, X_test, y_train, y_test = analyzer.prepare_data_for_modeling()
    
    # Train models
    lr_model, lr_pred = analyzer.train_linear_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_pred = analyzer.train_random_forest(X_train, X_test, y_train, y_test)
    
    # Prepare and train LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = analyzer.prepare_lstm_data()
    lstm_model, lstm_pred, lstm_y_test = analyzer.train_lstm(
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm
    )
    
    # Compare models
    analyzer.compare_models(lr_pred, rf_pred, lstm_pred, y_test, lstm_y_test)
    
    # Visualize predictions
    analyzer.visualize_predictions(y_test, lr_pred, rf_pred, lstm_pred, lstm_y_test)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - gold_price_analysis.png: Exploratory data analysis visualizations")
    print("  - model_predictions.png: Model predictions comparison")


if __name__ == "__main__":
    main()
